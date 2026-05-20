import torch
from torch import nn

from config import NUM_CLASSES


class LeNet(nn.Module):
    """用于 1x28x28 灰度图片分类的 LeNet 模型。"""

    feature_extractor: nn.Sequential
    classifier: nn.Sequential

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()

        # 特征提取部分负责从图片中提取边缘、纹理等局部模式。
        # 卷积层学习特征，Sigmoid 引入非线性，池化层缩小特征图并减少计算量。
        self.feature_extractor = nn.Sequential(
            # 5x5 卷积(6)，填充2
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            # 2x2 平均池化层，步幅2
            nn.AvgPool2d(kernel_size=2, stride=2),
            # 5x5 卷积(16)，填充0
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.Sigmoid(),
            # 2x2 平均池化层，步幅2
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        # 分类部分把卷积得到的特征转换为 10 个类别的分数。
        # 最后一层不加 Softmax，因为 CrossEntropyLoss 内部已经包含这一步。
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=16 * 5 * 5, out_features=120),
            nn.Sigmoid(),
            nn.Linear(in_features=120, out_features=84),
            nn.Sigmoid(),
            nn.Linear(in_features=84, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """定义一次前向传播：输入图片，输出每个类别的原始分数。"""

        features: torch.Tensor = self.feature_extractor(x)
        logits: torch.Tensor = self.classifier(features)
        return logits


if __name__ == "__main__":
    # 直接运行这个文件时，做一次简单的形状检查。
    # 模型输入应为 [batch, channel, height, width]，输出应为 [batch, num_classes]。
    model: LeNet = LeNet()
    sample_input: torch.Tensor = torch.randn(1, 1, 28, 28)
    sample_output: torch.Tensor = model(sample_input)

    print(model)
    print("Input shape:", tuple(sample_input.shape))
    print("Output shape:", tuple(sample_output.shape))
