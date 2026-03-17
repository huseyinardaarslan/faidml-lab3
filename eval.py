import argparse

import torch
import torch.nn as nn

from data.tiny_imagenet import create_dataloaders, download_and_extract_tiny_imagenet
from models import TinyImageNetCNN
from utils.training import evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, default="dataset")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/tiny_imagenet_cnn.pth",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    download_and_extract_tiny_imagenet(extract_dir=args.dataset_root)
    _, val_loader, class_names = create_dataloaders(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if val_loader is None:
        raise RuntimeError("Validation loader is not available for evaluation.")

    model = TinyImageNetCNN(num_classes=len(class_names)).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc = evaluate(
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        device=device,
    )
    print(f"Evaluation - loss: {val_loss:.4f}, acc: {val_acc:.4f}")


if __name__ == "__main__":
    main()
