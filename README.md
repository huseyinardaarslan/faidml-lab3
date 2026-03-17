# Tiny-ImageNet Training Starter

This repository contains a starter deep learning pipeline for Tiny-ImageNet with:

- dataset download and loading
- simple CNN model
- training script
- evaluation script
- class-wise sample visualization

## Project Structure

```text
faidml-lab3/
├── checkpoints/
├── data/
│   └── tiny_imagenet.py
├── dataset/
├── models/
│   ├── __init__.py
│   └── simple_cnn.py
├── utils/
│   ├── training.py
│   └── visualization.py
├── train.py
├── eval.py
└── requirements.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python3 train.py --epochs 1 --batch-size 32
```

Notes:
- Dataset is automatically downloaded to `dataset/` if not present.
- Checkpoint is saved to `checkpoints/tiny_imagenet_cnn.pth`.

## Evaluate

```bash
python3 eval.py --checkpoint checkpoints/tiny_imagenet_cnn.pth
```

## Useful Arguments

- `--dataset-root`: dataset folder (default: `dataset`)
- `--batch-size`: mini-batch size (default: `32`)
- `--num-workers`: dataloader workers (default: `0`)
- `--epochs`: number of training epochs (train only)
- `--lr`: learning rate (train only)
