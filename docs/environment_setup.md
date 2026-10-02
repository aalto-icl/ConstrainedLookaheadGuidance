# Environment Setup

[Back to README](../README.md)

## Python Environment

Create an isolated Python environment and install PyTorch for the CUDA version
available on your system. Then install the remaining dependencies:

```bash
conda create -n clag python=3.10
conda activate clag
pip install torch==2.4.1 torchvision==0.19.1
pip install -r requirements.txt
```

## Model Weights

CLAG uses the following pretrained models:

- [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) as the
  flow-matching backbone. This is a gated model; accept its license on the model
  page and authenticate with `huggingface-cli login` before running CLAG.
- [CLIP ViT-L/14](https://huggingface.co/openai/clip-vit-large-patch14) for the
  differentiable contrastive editing reward.

The weights are downloaded automatically from Hugging Face on the first run
and stored in the standard Hugging Face cache. Existing local model directories
can instead be supplied through `--model-id` and `--clip-model-id`.

Next: [Inference](inference.md)
