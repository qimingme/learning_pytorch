from pathlib import Path

import torch

from config import MODEL_PATH, get_device
from data import create_test_loader
from alex_net import AlexNet


def load_model(model_path=MODEL_PATH, device=None):
    """从文件加载训练好的模型。"""

    if device is None:
        device = get_device()

    # 评估前必须先完成训练，生成对应的权重文件。
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            "Model checkpoint not found. Run train.py first to create {}.".format(model_path)
        )

    # 先加载到 CPU，再移动到当前设备，可以避免跨设备保存/加载时的常见问题。
    model = AlexNet()
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model


def evaluate_model(model, data_loader, device=None):
    """在测试集上计算分类准确率。"""

    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    correct_count = 0
    sample_count = 0

    # 测试阶段不更新参数，只统计预测是否正确。
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            correct_count += torch.sum(predictions == labels).item()
            sample_count += images.size(0)

    return correct_count / sample_count


def main():
    # 脚本入口：加载最佳模型，并在测试集上报告最终准确率。
    device = get_device()
    model = load_model(MODEL_PATH, device)
    test_loader = create_test_loader()
    test_acc = evaluate_model(model, test_loader, device)

    print("Test accuracy: {:.4f}".format(test_acc))


if __name__ == "__main__":
    main()
