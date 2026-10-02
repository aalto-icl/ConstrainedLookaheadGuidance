"""Minimal FLUX implementation of constrained Look-ahead guidance."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch
from diffusers import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from PIL import Image

from .config import CLAGConfig
from .rewards import CLIPContrastiveReward


def _seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


class CLAGFluxPipeline:
    """Inversion-free FLUX editor with constrained Look-ahead updates."""

    def __init__(self, config: CLAGConfig, device: str = "cuda") -> None:
        config.validate()
        self.config = config
        self.device = torch.device(device)
        self._cpu = torch.device("cpu")
        self._pipe: Optional[FluxPipeline] = None
        self._reward: Optional[CLIPContrastiveReward] = None

    @property
    def pipe(self) -> FluxPipeline:
        self.preload()
        assert self._pipe is not None
        return self._pipe

    @property
    def reward(self) -> CLIPContrastiveReward:
        self.preload()
        assert self._reward is not None
        return self._reward

    def preload(self) -> None:
        if self._pipe is not None:
            return

        print(f"[init] loading reward model: {self.config.clip_model_id}")
        self._reward = CLIPContrastiveReward(self.config.clip_model_id, self.device)

        print(f"[init] loading FLUX pipeline: {self.config.model_id}")
        pipe = FluxPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=torch.bfloat16,
        )
        if self.config.cpu_offload:
            pipe.transformer = pipe.transformer.to(self.device)
            pipe.vae = pipe.vae.to(self.device)
        else:
            pipe = pipe.to(self.device)

        for module in (pipe.transformer, pipe.vae):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        if self.config.gradient_checkpointing:
            pipe.transformer.enable_gradient_checkpointing()
        self._pipe = pipe

    def _resolve_image_size(self, image: Image.Image) -> tuple[int, int]:
        pipe = self.pipe
        original_width, original_height = image.size
        default_size = int(pipe.default_sample_size * pipe.vae_scale_factor)
        alignment = int(pipe.vae_scale_factor * 2)
        max_side = max(original_height, original_width)
        scale = min(1.0, default_size / float(max_side))
        height = max(alignment, int(round(original_height * scale)) // alignment * alignment)
        width = max(alignment, int(round(original_width * scale)) // alignment * alignment)
        return height, width

    def _encode_image(
        self,
        image: Image.Image,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pipe = self.pipe
        dtype = pipe.transformer.dtype
        pixels = pipe.image_processor.preprocess(image, height=height, width=width)
        pixels = pixels.to(device=self.device, dtype=dtype)
        with torch.no_grad():
            latent = pipe.vae.encode(pixels).latent_dist.sample()
        latent = (latent - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        batch, channels, latent_height, latent_width = latent.shape
        packed = pipe._pack_latents(
            latent, batch, channels, latent_height, latent_width
        )
        image_ids = pipe._prepare_latent_image_ids(
            batch,
            latent_height // 2,
            latent_width // 2,
            self.device,
            dtype,
        )
        return packed, image_ids

    def _decode_latent(
        self,
        latent: torch.Tensor,
        height: int,
        width: int,
        checkpoint: bool = False,
    ) -> torch.Tensor:
        pipe = self.pipe
        unpacked = pipe._unpack_latents(latent, height, width, pipe.vae_scale_factor)
        unpacked = unpacked / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
        vae_dtype = next(pipe.vae.parameters()).dtype

        def decode(value: torch.Tensor) -> torch.Tensor:
            return pipe.vae.decode(value.to(dtype=vae_dtype), return_dict=False)[0]

        if checkpoint:
            decoded = torch.utils.checkpoint.checkpoint(
                decode, unpacked, use_reentrant=False
            )
        else:
            decoded = decode(unpacked)
        return decoded.clamp(-1.0, 1.0)

    def _prepare_scheduler(self, image_seq_len: int) -> torch.Tensor:
        pipe = self.pipe
        steps = self.config.num_steps
        sigmas = np.linspace(1.0, 1.0 / steps, steps)
        mu = calculate_shift(
            image_seq_len,
            pipe.scheduler.config.get("base_image_seq_len", 256),
            pipe.scheduler.config.get("max_image_seq_len", 4096),
            pipe.scheduler.config.get("base_shift", 0.5),
            pipe.scheduler.config.get("max_shift", 1.15),
        )
        retrieve_timesteps(
            pipe.scheduler, steps, self.device, None, sigmas, mu=mu
        )
        return pipe.scheduler.sigmas.clone()

    def _predict_velocity(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        image_ids: torch.Tensor,
        guidance_scale: float,
        checkpoint: bool,
    ) -> torch.Tensor:
        pipe = self.pipe
        expanded_timestep = timestep.expand(latent.shape[0]).to(latent.dtype)
        guidance = None
        if pipe.transformer.config.guidance_embeds:
            guidance = torch.full(
                (latent.shape[0],),
                guidance_scale,
                device=latent.device,
                dtype=torch.float32,
            )

        def forward(
            value: torch.Tensor,
            time: torch.Tensor,
            prompt: torch.Tensor,
            pooled: torch.Tensor,
            txt_ids: torch.Tensor,
            img_ids: torch.Tensor,
            guidance_value: torch.Tensor,
        ) -> torch.Tensor:
            return pipe.transformer(
                hidden_states=value,
                timestep=time / 1000,
                guidance=guidance_value,
                pooled_projections=pooled,
                encoder_hidden_states=prompt,
                txt_ids=txt_ids,
                img_ids=img_ids,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]

        if checkpoint:
            return torch.utils.checkpoint.checkpoint(
                forward,
                latent,
                expanded_timestep,
                prompt_embeds,
                pooled_prompt_embeds,
                text_ids,
                image_ids,
                guidance,
                use_reentrant=False,
            )
        return forward(
            latent,
            expanded_timestep,
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
            image_ids,
            guidance,
        )

    def _euler_step(
        self,
        latent: torch.Tensor,
        sigma: float,
        next_sigma: float,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        image_ids: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        velocity = self._predict_velocity(
            latent,
            timestep,
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
            image_ids,
            guidance_scale,
            checkpoint=False,
        )
        return latent + velocity.to(latent.dtype) * (next_sigma - sigma)

    @staticmethod
    def _normalize(gradient: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        gradient = gradient.detach().float()
        return (gradient / (gradient.norm() + 1e-8)).to(dtype)

    @staticmethod
    def _fidelity_reward(prediction: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        return -((prediction - source.detach()) ** 2).mean()

    def _edit_reward_at(
        self,
        latent: torch.Tensor,
        sigma: float,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        image_ids: torch.Tensor,
        source_pixels: torch.Tensor,
        source_prompt: str,
        edit_prompt: str,
        height: int,
        width: int,
    ) -> torch.Tensor:
        velocity = self._predict_velocity(
            latent,
            timestep,
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
            image_ids,
            self.config.guidance_scale,
            checkpoint=self.config.gradient_checkpointing,
        )
        prediction = latent - sigma * velocity
        decoded = self._decode_latent(
            prediction,
            height,
            width,
            checkpoint=self.config.gradient_checkpointing,
        )
        return self.reward(decoded, source_prompt, edit_prompt)

    def _fidelity_reward_at(
        self,
        latent: torch.Tensor,
        source_latent: torch.Tensor,
        sigma: float,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        image_ids: torch.Tensor,
        checkpoint: Optional[bool] = None,
    ) -> torch.Tensor:
        if checkpoint is None:
            checkpoint = self.config.gradient_checkpointing
        velocity = self._predict_velocity(
            latent,
            timestep,
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
            image_ids,
            self.config.source_guidance_scale,
            checkpoint=checkpoint,
        )
        prediction = latent - sigma * velocity
        return self._fidelity_reward(prediction, source_latent)

    def _lookahead_proposal(
        self,
        latent: torch.Tensor,
        source_latent: torch.Tensor,
        sigma: float,
        timestep: torch.Tensor,
        edit_condition: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        source_condition: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        image_ids: torch.Tensor,
        source_pixels: torch.Tensor,
        source_prompt: str,
        edit_prompt: str,
        height: int,
        width: int,
        edit_gradient: torch.Tensor,
        fidelity_gradient: torch.Tensor,
    ) -> torch.Tensor:
        config = self.config
        edit_direction = self._normalize(edit_gradient, latent.dtype)
        fidelity_direction = self._normalize(fidelity_gradient, latent.dtype)

        edit_probe = (
            latent.detach() + config.edit_probe_step_size * edit_direction
        ).requires_grad_(True)
        edit_reward = self._edit_reward_at(
            edit_probe,
            sigma,
            timestep,
            *edit_condition,
            image_ids,
            source_pixels,
            source_prompt,
            edit_prompt,
            height,
            width,
        )
        edit_lookahead = torch.autograd.grad(edit_reward, edit_probe)[0].detach()
        del edit_reward, edit_probe

        fidelity_probe = (
            latent.detach() + config.fidelity_probe_step_size * fidelity_direction
        ).requires_grad_(True)
        fidelity_reward = config.fidelity_weight * self._fidelity_reward_at(
            fidelity_probe,
            source_latent,
            sigma,
            timestep,
            *source_condition,
            image_ids,
        )
        fidelity_lookahead = torch.autograd.grad(
            fidelity_reward, fidelity_probe
        )[0].detach()
        del fidelity_reward, fidelity_probe

        lookahead_gradient = edit_lookahead.float() + fidelity_lookahead.float()
        direction = self._normalize(lookahead_gradient, latent.dtype)
        return latent.detach() + config.update_step_size * direction

    def _fidelity_value(
        self,
        latent: torch.Tensor,
        source_latent: torch.Tensor,
        sigma: float,
        timestep: torch.Tensor,
        source_condition: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        image_ids: torch.Tensor,
    ) -> float:
        with torch.no_grad():
            reward = self._fidelity_reward_at(
                latent,
                source_latent,
                sigma,
                timestep,
                *source_condition,
                image_ids,
                checkpoint=False,
            )
        return float(reward.item())

    def _project_fidelity(
        self,
        latent: torch.Tensor,
        source_latent: torch.Tensor,
        sigma: float,
        timestep: torch.Tensor,
        source_condition: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        image_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, float, int]:
        config = self.config
        projected = latent.detach()
        reward_value = float("nan")

        for projection_index in range(config.max_projection_steps):
            variable = projected.requires_grad_(True)
            reward = self._fidelity_reward_at(
                variable,
                source_latent,
                sigma,
                timestep,
                *source_condition,
                image_ids,
            )
            distance = -reward
            reward_value = float(reward.item())
            if distance.item() <= config.fidelity_epsilon:
                return projected.detach(), reward_value, projection_index

            gradient = torch.autograd.grad(distance, variable)[0]
            direction = self._normalize(gradient, projected.dtype)
            projected = (
                projected.detach() - config.projection_step_size * direction
            )

        return projected.detach(), reward_value, config.max_projection_steps

    def _encode_prompt(
        self,
        prompt: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pipe = self.pipe
        encode_device = self._cpu if self.config.cpu_offload else self.device
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
            prompt,
            prompt,
            device=encode_device,
            num_images_per_prompt=1,
        )
        dtype = pipe.transformer.dtype
        return (
            prompt_embeds.to(device=self.device, dtype=dtype),
            pooled_prompt_embeds.to(device=self.device, dtype=dtype),
            text_ids.to(device=self.device),
        )

    def edit(
        self,
        image: Image.Image,
        source_prompt: str,
        edit_prompt: str,
    ) -> Image.Image:
        """Edit one image using the paper's inversion-free Look-ahead method."""
        config = self.config
        _seed_everything(config.seed, config.deterministic)
        pipe = self.pipe
        height, width = self._resolve_image_size(image)
        source_latent, image_ids = self._encode_image(image, height, width)
        with torch.no_grad():
            source_pixels = self._decode_latent(source_latent, height, width).detach()

        source_condition = self._encode_prompt(source_prompt)
        edit_condition = self._encode_prompt(edit_prompt)
        torch.cuda.empty_cache()

        sigmas = self._prepare_scheduler(source_latent.shape[1])
        timesteps = pipe.scheduler.timesteps
        sigma_initial = float(sigmas[0])
        noise = torch.randn_like(source_latent)
        latent = (1.0 - sigma_initial) * source_latent + sigma_initial * noise
        print(f"[forward] sigma={sigma_initial:.4f}")

        for step in range(config.num_steps):
            sigma = float(sigmas[step])
            next_sigma = float(sigmas[step + 1])
            timestep = timesteps[step].to(self.device)

            with torch.no_grad():
                updated = self._euler_step(
                    latent,
                    sigma,
                    next_sigma,
                    timestep,
                    *edit_condition,
                    image_ids,
                    config.guidance_scale,
                )

            print(f"[step {step + 1:02d}/{config.num_steps}]")
            for inner_step in range(config.inner_steps):
                edit_variable = updated.detach().requires_grad_(True)
                edit_reward = self._edit_reward_at(
                    edit_variable,
                    sigma,
                    timestep,
                    *edit_condition,
                    image_ids,
                    source_pixels,
                    source_prompt,
                    edit_prompt,
                    height,
                    width,
                )
                edit_gradient = torch.autograd.grad(edit_reward, edit_variable)[0].detach()
                edit_value = float(edit_reward.item())

                fidelity_variable = updated.detach().requires_grad_(True)
                fidelity_reward = self._fidelity_reward_at(
                    fidelity_variable,
                    source_latent,
                    sigma,
                    timestep,
                    *source_condition,
                    image_ids,
                )
                fidelity_gradient = torch.autograd.grad(
                    fidelity_reward, fidelity_variable
                )[0].detach()

                proposal = self._lookahead_proposal(
                    updated,
                    source_latent,
                    sigma,
                    timestep,
                    edit_condition,
                    source_condition,
                    image_ids,
                    source_pixels,
                    source_prompt,
                    edit_prompt,
                    height,
                    width,
                    edit_gradient,
                    fidelity_gradient,
                )
                proposal_fidelity = self._fidelity_value(
                    proposal,
                    source_latent,
                    sigma,
                    timestep,
                    source_condition,
                    image_ids,
                )

                projection_steps = 0
                if proposal_fidelity < -config.fidelity_epsilon:
                    updated, final_fidelity, projection_steps = self._project_fidelity(
                        proposal,
                        source_latent,
                        sigma,
                        timestep,
                        source_condition,
                        image_ids,
                    )
                else:
                    updated = proposal
                    final_fidelity = proposal_fidelity

                print(
                    f"  inner={inner_step + 1}/{config.inner_steps} "
                    f"edit={edit_value:.4f} fidelity={final_fidelity:.4f} "
                    f"projections={projection_steps}"
                )

            latent = updated.detach()

        with torch.no_grad():
            decoded = self._decode_latent(latent, height, width)
        return pipe.image_processor.postprocess(decoded, output_type="pil")[0]
