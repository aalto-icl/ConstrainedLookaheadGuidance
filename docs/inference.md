# Inference

[Back to README](../README.md)

Complete the [Environment Setup](environment_setup.md) first.

## Edit an Image

```bash
python scripts/edit.py \
  --image assets/steak.png \
  --source-prompt "A steak accompanied by a side of leaf salad." \
  --edit-prompt "A bread roll accompanied by a side of leaf salad." \
  --output outputs/steak_to_bread_roll.png
```

`--source-prompt` describes the input image and `--edit-prompt` describes the
desired result. Additional example images are provided in `assets/`.

## Hyperparameters

The defaults match the settings used in the paper.

- `--num-steps`: Number of FLUX denoising steps.
- `--inner-steps`: Number of Look-ahead optimization updates performed at each
  denoising step.
- `--alpha`: Step size used to construct the edit-oriented probe state.
- `--beta`: Step size used to construct the fidelity-oriented probe state.
- `--gamma`: Step size of the Look-ahead update committed to the sampling
  trajectory.
- `--fidelity-weight`: Weight applied to the fidelity reward in the Look-ahead
  objective.
- `--epsilon`: Maximum allowed latent fidelity distance from the source. Larger
  values relax the constraint and generally permit stronger edits.
- `--projection-step-size`: Step size of each fidelity correction update.
- `--max-projection-steps`: Maximum number of fidelity corrections applied to
  a violating proposal.
- `--guidance-scale`: Guidance scale used by FLUX for the edit condition.
- `--seed`: Random seed used for inversion-free forward initialization.
- `--no-gradient-checkpointing`: Disable gradient checkpointing to reduce
  runtime at the cost of higher GPU memory usage.
- `--no-cpu-offload`: Keep the complete FLUX pipeline on the GPU, including
  the text encoders.
