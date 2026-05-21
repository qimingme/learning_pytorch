from pathlib import Path

import torch

DATA_DIR: Path = Path("../data")
CHECKPOINT_DIR: Path = Path("../checkpoints")
MODEL_PATH: Path = CHECKPOINT_DIR / "alexnet_fashion_mnist.pth"

# FashionMNIST 是 28x28 的灰度图，一共有 10 个类别。
IMAGE_SIZE: int = 227
NUM_CLASSES: int = 10

NUM_EPOCHS: int = 20
TRAIN_BATCH_SIZE: int = 32
EVAL_BATCH_SIZE: int = 256
LEARNING_RATE: float = 0.001
VALIDATION_RATIO: float = 0.2
RANDOM_SEED: int = 42


def get_device() -> torch.device:
    """选择当前电脑上可用的最快训练设备。"""

    # CUDA 常见于 NVIDIA 显卡；如果可用，通常训练最快。
    if torch.cuda.is_available():
        return torch.device("cuda")

    # MPS 是 Apple Silicon Mac 上的 GPU 后端。
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    # 没有 GPU 时使用 CPU，速度慢一些，但最稳定。
    return torch.device("cpu")


if __name__ == "__main__":
    device: torch.device = get_device()
    print(device)
