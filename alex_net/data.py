from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import FashionMNIST

from config import (
    DATA_DIR,
    EVAL_BATCH_SIZE,
    IMAGE_SIZE,
    RANDOM_SEED,
    TRAIN_BATCH_SIZE,
    VALIDATION_RATIO,
)


def build_transform() -> transforms.Compose:
    """定义图像预处理流程。"""

    # 神经网络接收的是张量，不是 PIL 图片。
    # Resize 保证输入尺寸统一，ToTensor 会把像素值缩放到 0~1。
    return transforms.Compose(
        [
            transforms.Resize(size=IMAGE_SIZE),
            transforms.ToTensor(),
        ]
    )


def create_train_val_loaders(
        data_dir: Path = DATA_DIR,
        batch_size: int = TRAIN_BATCH_SIZE,
        validation_ratio: float = VALIDATION_RATIO,
        num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """创建训练集和验证集的数据加载器。"""

    # 训练集用于更新模型参数；验证集用于观察模型是否真的学会了规律。
    # download=True 表示如果本地没有数据集，torchvision 会自动下载。
    dataset = FashionMNIST(
        root=str(data_dir),
        train=True,
        transform=build_transform(),
        download=True,
    )

    # 从训练数据中切出一部分作为验证集。
    # 固定随机种子可以让每次切分结果一致，方便复现实验。
    val_size: int = int(len(dataset) * validation_ratio)
    train_size: int = len(dataset) - val_size
    split_generator: torch.Generator = torch.Generator().manual_seed(RANDOM_SEED)

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=split_generator,
    )

    # DataLoader 负责按 batch 取数据。
    # 训练时 shuffle=True 可以打乱样本顺序，让模型更稳定地学习。
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    # 验证集不参与参数更新，保持固定顺序即可。
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader


def create_test_loader(
        data_dir: Path = DATA_DIR,
        batch_size: int = EVAL_BATCH_SIZE,
        num_workers: int = 0,
) -> DataLoader:
    """创建测试集的数据加载器。"""

    # 测试集只在训练完成后使用，用来评估模型在未见过数据上的表现。
    test_dataset = FashionMNIST(
        root=str(data_dir),
        train=False,
        transform=build_transform(),
        download=True,
    )

    return DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


def create_preview_loader(
        data_dir: Path = DATA_DIR,
        batch_size: int = 64,
        num_workers: int = 0,
) -> tuple[FashionMNIST, DataLoader]:
    """创建用于可视化预览的数据加载器。"""

    # 这个加载器只用于展示图片，帮助确认数据读取和标签是否正常。
    train_dataset = FashionMNIST(
        root=str(data_dir),
        train=True,
        transform=build_transform(),
        download=True,
    )

    preview_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    return train_dataset, preview_loader
