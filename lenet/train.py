import copy
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from config import LEARNING_RATE, MODEL_PATH, NUM_EPOCHS, get_device
from data import create_train_val_loaders
from lenet import LeNet

HistoryItem = dict[str, int | float]
TrainingHistory = list[HistoryItem]


def train_one_epoch(
        model: nn.Module,
        data_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
) -> tuple[float, float]:
    """训练一个 epoch，并返回平均损失和准确率。"""

    # train 模式会启用训练相关行为，例如 Dropout、BatchNorm 的训练状态。
    # 这个模型里暂时没有这些层，但保留写法有助于形成正确习惯。
    model.train()

    running_loss: float = 0.0
    correct_count: int = 0
    sample_count: int = 0

    for images, labels in data_loader:
        images: torch.Tensor
        labels: torch.Tensor

        # 数据和模型必须放在同一个设备上，否则无法计算。
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播：模型根据图片输出每个类别的分数。
        outputs: torch.Tensor = model(images)
        loss: torch.Tensor = criterion(outputs, labels)
        predictions: torch.Tensor = torch.argmax(outputs, dim=1)

        # 反向传播三步：清空旧梯度、计算新梯度、更新参数。
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 用样本数量加权累计，最后得到整个 epoch 的平均指标。
        batch_size: int = images.size(0)
        running_loss += loss.item() * batch_size
        correct_count += int(torch.sum(predictions == labels).item())
        sample_count += batch_size

    return running_loss / sample_count, correct_count / sample_count


def validate_one_epoch(
        model: nn.Module,
        data_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
) -> tuple[float, float]:
    """验证一个 epoch，并返回平均损失和准确率。"""

    # eval 模式表示现在只评估模型，不训练模型。
    model.eval()

    running_loss: float = 0.0
    correct_count: int = 0
    sample_count: int = 0

    # 验证阶段不需要梯度，关闭梯度能节省显存和计算时间。
    with torch.no_grad():
        for images, labels in data_loader:
            images: torch.Tensor
            labels: torch.Tensor

            images = images.to(device)
            labels = labels.to(device)

            outputs: torch.Tensor = model(images)
            loss: torch.Tensor = criterion(outputs, labels)
            predictions: torch.Tensor = torch.argmax(outputs, dim=1)

            batch_size: int = images.size(0)
            running_loss += loss.item() * batch_size
            correct_count += int(torch.sum(predictions == labels).item())
            sample_count += batch_size

    return running_loss / sample_count, correct_count / sample_count


def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = NUM_EPOCHS,
        learning_rate: float = LEARNING_RATE,
        model_path: Path = MODEL_PATH,
) -> tuple[nn.Module, TrainingHistory]:
    """完整训练流程：训练、验证、保存验证集表现最好的模型。"""

    device: torch.device = get_device()
    model = model.to(device)

    # CrossEntropyLoss 适合多分类任务，输入是原始分数 logits，标签是类别编号。
    criterion: nn.Module = nn.CrossEntropyLoss()

    # Adam 是常用优化器，入门项目中收敛通常比较稳定。
    optimizer: Optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc: float = 0.0
    best_model_weights: dict[str, Any] = copy.deepcopy(model.state_dict())
    history: TrainingHistory = []
    start_time: float = time.time()

    for epoch in range(num_epochs):
        # 每个 epoch 都先训练，再用验证集检查泛化表现。
        train_loss: float
        train_acc: float
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss: float
        val_acc: float
        val_loss, val_acc = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
        )

        # history 用于后续画出损失和准确率曲线。
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
            }
        )

        # 只保存验证集准确率最高的一版参数，避免最后一轮刚好退步。
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())

        elapsed_time: float = time.time() - start_time
        print(
            "Epoch {}/{} | "
            "train loss: {:.4f}, train acc: {:.4f} | "
            "val loss: {:.4f}, val acc: {:.4f} | "
            "time: {:.0f}m {:.0f}s".format(
                epoch + 1,
                num_epochs,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                elapsed_time // 60,
                elapsed_time % 60,
            )
        )

    # 保存模型前先确保 checkpoints 目录存在。
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_model_weights, model_path)
    print("Best validation accuracy: {:.4f}".format(best_val_acc))
    print("Saved model weights to:", model_path)

    model.load_state_dict(best_model_weights)
    return model, history


def plot_training_history(history: TrainingHistory) -> None:
    """绘制训练过程中的损失和准确率曲线。"""

    # 延迟导入 matplotlib，避免只导入训练函数时就触发绘图库初始化。
    import matplotlib.pyplot as plt

    epochs: list[int | float] = [item["epoch"] for item in history]

    train_losses: list[int | float] = [item["train_loss"] for item in history]
    val_losses: list[int | float] = [item["val_loss"] for item in history]
    train_accuracies: list[int | float] = [item["train_acc"] for item in history]
    val_accuracies: list[int | float] = [item["val_acc"] for item in history]

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, "ro-", label="Train loss")
    plt.plot(epochs, val_losses, "bs-", label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, "ro-", label="Train acc")
    plt.plot(epochs, val_accuracies, "bs-", label="Val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


def main() -> None:
    # 脚本入口：准备数据、创建模型、开始训练、展示曲线。
    train_loader: DataLoader
    val_loader: DataLoader
    train_loader, val_loader = create_train_val_loaders()

    model: LeNet = LeNet()

    history: TrainingHistory
    _, history = train_model(model, train_loader, val_loader)

    plot_training_history(history)


if __name__ == "__main__":
    main()
