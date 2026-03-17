import os
from io import BytesIO
from zipfile import ZipFile

import requests
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder


TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def download_and_extract_tiny_imagenet(
    url=TINY_IMAGENET_URL,
    extract_dir="dataset",
):
    os.makedirs(extract_dir, exist_ok=True)
    extracted_root = os.path.join(extract_dir, "tiny-imagenet-200")

    if os.path.isdir(extracted_root):
        print(f"Dataset already exists at: {extracted_root}")
        return extracted_root

    response = requests.get(url, timeout=120)
    response.raise_for_status()

    with ZipFile(BytesIO(response.content)) as zip_file:
        zip_file.extractall(extract_dir)

    print("Download and extraction complete!")
    return extracted_root


def get_default_transform(image_size=224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class TinyImageNetValDataset(Dataset):
    def __init__(self, val_dir, class_to_idx, transform=None):
        self.transform = transform
        images_dir = os.path.join(val_dir, "images")
        annotations_path = os.path.join(val_dir, "val_annotations.txt")
        self.samples = []

        with open(annotations_path, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split("\t")
                image_name = parts[0]
                class_name = parts[1]
                if class_name in class_to_idx:
                    image_path = os.path.join(images_dir, image_name)
                    self.samples.append((image_path, class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def create_dataloaders(dataset_root="dataset", batch_size=32, num_workers=0):
    transform = get_default_transform()
    base_path = os.path.join(dataset_root, "tiny-imagenet-200")
    train_path = os.path.join(base_path, "train")
    val_path = os.path.join(base_path, "val")

    train_dataset = ImageFolder(root=train_path, transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = None
    if os.path.isdir(val_path):
        val_dataset = TinyImageNetValDataset(
            val_dir=val_path,
            class_to_idx=train_dataset.class_to_idx,
            transform=transform,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    return train_loader, val_loader, train_dataset.classes
