import matplotlib.pyplot as plt
import numpy as np


def denormalize(image_tensor):
    image = image_tensor.detach().cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    return np.clip(image, 0, 1)


def plot_one_example_per_class(dataloader, class_names, max_classes=10):
    rows, cols = 2, 5
    fig, axes = plt.subplots(rows, cols, figsize=(15, 7))
    axes = axes.flatten()
    found_classes = {}

    for images, labels in dataloader:
        for i in range(len(labels)):
            label_idx = labels[i].item()
            if label_idx not in found_classes:
                found_classes[label_idx] = images[i]
            if len(found_classes) == max_classes:
                break
        if len(found_classes) == max_classes:
            break

    for i, (label_idx, img_tensor) in enumerate(found_classes.items()):
        img = denormalize(img_tensor)
        axes[i].imshow(img)
        axes[i].set_title(f"Class: {class_names[label_idx]}")
        axes[i].axis("off")

    for j in range(len(found_classes), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()
