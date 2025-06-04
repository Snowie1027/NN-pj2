import torch
import torch.nn as nn
import torch.nn.functional as F

class FullyConnected(nn.Module):
    """
    全连接层封装（等价于 nn.Linear）

    参数:
    - in_features (int): 输入特征维度
    - out_features (int): 输出特征维度
    """
    def __init__(self, in_features, out_features):
        super(FullyConnected, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.fc(x)


class Conv2D(nn.Module):
    """
    2D 卷积层封装（等价于 nn.Conv2d）

    参数:
    - in_channels (int): 输入通道数
    - out_channels (int): 输出通道数
    - kernel_size (int 或 tuple): 卷积核大小
    - stride (int): 步长，默认 1
    - padding (int): 填充大小，默认 0
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding)

    def forward(self, x):
        return self.conv(x)


class MaxPool2D(nn.Module):
    """
    2D 最大池化层封装（等价于 nn.MaxPool2d）

    参数:
    - kernel_size (int 或 tuple): 池化核大小
    - stride (int): 步长，默认与 kernel_size 相同
    - padding (int): 填充大小，默认 0
    """
    def __init__(self, kernel_size, stride=None, padding=0):
        super(MaxPool2D, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.pool(x)


class ReLU(nn.Module):
    """
    ReLU 激活函数封装（等价于 F.relu）
    """
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        return F.relu(x)


class Sigmoid(nn.Module):
    """
    Sigmoid 激活函数封装（等价于 torch.sigmoid）
    """
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        return torch.sigmoid(x)


class Tanh(nn.Module):
    """
    Tanh 激活函数封装（等价于 torch.tanh）
    """
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        return torch.tanh(x)


# 测试模块功能
if __name__ == '__main__':
    # 构造一个模拟图像输入（batch_size=1, channels=3, height=32, width=32）
    x = torch.randn(1, 3, 32, 32)

    # 初始化模块
    conv = Conv2D(3, 16, kernel_size=3, padding=1)
    pool = MaxPool2D(kernel_size=2, stride=2)
    relu = ReLU()

    # 前向传播
    out = conv(x)
    print(f"Conv2D 输出 shape: {out.shape}")

    out = pool(out)
    print(f"MaxPool2D 输出 shape: {out.shape}")

    out = relu(out)
    print(f"ReLU 输出 shape: {out.shape}")

    # 展平后连接全连接层
    out_flat = out.view(out.size(0), -1)
    fc = FullyConnected(out_flat.size(1), 10)
    out = fc(out_flat)
    print(f"FullyConnected 输出 shape: {out.shape}")
