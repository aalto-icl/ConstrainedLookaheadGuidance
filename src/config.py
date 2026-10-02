"""Configuration for the paper's Look-ahead model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CLAGConfig:
    """Look-ahead configuration with the paper defaults."""

    model_id: str = "black-forest-labs/FLUX.1-dev"
    clip_model_id: str = "openai/clip-vit-large-patch14"
    num_steps: int = 28
    inner_steps: int = 3
    guidance_scale: float = 3.5
    source_guidance_scale: float = 0.0

    # alpha, beta, and gamma in the paper.
    edit_probe_step_size: float = 5.0
    fidelity_probe_step_size: float = 5.0
    update_step_size: float = 5.0
    fidelity_weight: float = 1.0

    # The paper writes r_fid >= -epsilon; the implementation uses D_fid <= epsilon.
    fidelity_epsilon: float = 0.1
    projection_step_size: float = 5.0
    max_projection_steps: int = 3

    gradient_checkpointing: bool = True
    cpu_offload: bool = True
    seed: int = 42
    deterministic: bool = False

    def validate(self) -> None:
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if self.inner_steps <= 0:
            raise ValueError("inner_steps must be positive")
        if self.fidelity_epsilon <= 0:
            raise ValueError("fidelity_epsilon must be positive")
        if self.max_projection_steps <= 0:
            raise ValueError("max_projection_steps must be positive")
