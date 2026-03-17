import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim

from data.tiny_imagenet import create_dataloaders, download_and_extract_tiny_imagenet
from models import TinyImageNetCNN
from utils.training import evaluate, train_one_epoch
from utils.visualization import plot_one_example_per_class


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, default="dataset")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("checkpoints", exist_ok=True)

    dataset_root = args.dataset_root
    download_and_extract_tiny_imagenet(extract_dir=dataset_root)

    train_loader, val_loader, class_names = create_dataloaders(
        dataset_root=dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = TinyImageNetCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"Number of classes: {len(class_names)}")
    print(f"Number of samples: {len(train_loader.dataset)}")
    print(f"Validation loader available: {val_loader is not None}")
    print(f"Device: {device}")

    plot_one_example_per_class(train_loader, class_names, max_classes=10)

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        print(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"train loss: {train_loss:.4f}, train acc: {train_acc:.4f}"
        )

        if val_loader is not None:
            val_loss, val_acc = evaluate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
            )
            print(f"Validation - loss: {val_loss:.4f}, acc: {val_acc:.4f}")

    checkpoint_path = "checkpoints/tiny_imagenet_cnn.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": len(class_names),
        },
        checkpoint_path,
    )
    print(f"Checkpoint saved: {checkpoint_path}")


if __name__ == "__main__":
    main()
