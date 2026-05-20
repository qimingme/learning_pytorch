import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST

from data import create_preview_loader


def show_image_batch(
        dataset: FashionMNIST,
        data_loader: DataLoader,
        rows: int = 4,
        cols: int = 16,
) -> None:
    """展示一个 batch 的图片和标签。"""

    # 可视化数据是训练前很有价值的一步：
    # 它能帮助我们确认图片尺寸、颜色通道、标签名称都符合预期。
    images: torch.Tensor  # size: [batch_size, 1, 28, 28]
    labels: torch.Tensor
    images, labels = next(iter(data_loader))
    images: np.ndarray = images.squeeze(1).numpy()  # size: (batch_size, 28, 28)
    labels: np.ndarray = labels.numpy()

    plt.figure(figsize=(12, 5))
    for index in np.arange(min(len(labels), rows * cols)):
        label_index: int = int(labels[index])

        plt.subplot(rows, cols, int(index) + 1)
        plt.imshow(images[index], cmap=plt.cm.gray)
        plt.title(dataset.classes[label_index], size=8)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def main() -> None:
    # 脚本入口：随机取一批训练图片并显示出来。
    dataset: FashionMNIST
    preview_loader: DataLoader
    dataset, preview_loader = create_preview_loader()
    show_image_batch(dataset, preview_loader)


if __name__ == "__main__":
    main()
