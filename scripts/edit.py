#!/usr/bin/env python3
"""Edit one image with CLAG."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src import CLAGConfig, CLAGFluxPipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Constrained Look-ahead editing with FLUX.1-dev"
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--source-prompt", required=True)
    parser.add_argument("--edit-prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--clip-model-id", default="openai/clip-vit-large-patch14")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-steps", type=int, default=28)
    parser.add_argument("--inner-steps", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--gamma", type=float, default=5.0)
    parser.add_argument("--fidelity-weight", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--projection-step-size", type=float, default=5.0)
    parser.add_argument("--max-projection-steps", type=int, default=3)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--no-cpu-offload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CLAGConfig(
        model_id=args.model_id,
        clip_model_id=args.clip_model_id,
        seed=args.seed,
        num_steps=args.num_steps,
        inner_steps=args.inner_steps,
        edit_probe_step_size=args.alpha,
        fidelity_probe_step_size=args.beta,
        update_step_size=args.gamma,
        fidelity_weight=args.fidelity_weight,
        fidelity_epsilon=args.epsilon,
        projection_step_size=args.projection_step_size,
        max_projection_steps=args.max_projection_steps,
        guidance_scale=args.guidance_scale,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        cpu_offload=not args.no_cpu_offload,
    )
    editor = CLAGFluxPipeline(config)
    image = Image.open(args.image).convert("RGB")
    result = editor.edit(image, args.source_prompt, args.edit_prompt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    print(f"Saved edited image to {args.output}")


if __name__ == "__main__":
    main()
