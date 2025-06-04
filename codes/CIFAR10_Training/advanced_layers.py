import torch
import torch.nn as nn
import torch.nn.functional as F
from data_loader import get_cifar10_dataloaders

class BatchNorm2D(nn.Module):
    """
    2D 批归一化层封装（等价于 nn.BatchNorm2d）

    参数:
    - num_features (int): 特征通道数
    """
    def __init__(self, num_features):
        super(BatchNorm2D, self).__init__()
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        return self.bn(x)


class Dropout2D(nn.Module):
    """
    Dropout 层封装（用于 2D 特征图，等价于 nn.Dropout）

    参数:
    - p (float): 每个元素被置为 0 的概率（默认 0.5）
    """
    def __init__(self, p=0.5):
        super(Dropout2D, self).__init__()
        self.p = p

    def forward(self, x):
        # 检查输入是否是 4D（batch_size, channels, height, width）
        if x.dim() == 4:
            # 使用 nn.Dropout 针对每个通道进行 Dropout
            return nn.Dropout(self.p)(x)
        else:
            # 如果是非4D输入（2D或其他维度），直接使用 Dropout
            return nn.Dropout(self.p)(x)

class ResidualBlock(nn.Module):
    """
    基本残差块（ResNet 基础结构）

    参数:
    - in_channels (int): 输入通道数
    - out_channels (int): 输出通道数
    - stride (int): 第一个卷积层的步长，默认 1
    - downsample (nn.Module 或 None): 下采样模块，用于匹配输入输出维度
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        identity = x  # 残差连接

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)  # 对输入进行下采样匹配尺寸

        out += identity  # 残差连接
        out = self.relu(out)
        return out


class SEBlock(nn.Module):
    """
    通道注意力机制：Squeeze-and-Excitation Block

    参数:
    - channel (int): 输入通道数
    - reduction (int): 压缩比，默认为 16
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)        # Squeeze：全局平均池化
        y = self.fc(y).view(b, c, 1, 1)        # Excitation：全连接权重生成注意力
        return x * y.expand_as(x)              # 缩放原特征图（通道加权）

class SimpleCNN(nn.Module):
    """
    网络结构类 SimpleCNN
    """
    def __init__(self, use_bn=False, use_dropout=False):
        super(SimpleCNN, self).__init__()
        layers = []

        # 第一个卷积层
        layers.append(nn.Conv2d(3, 64, kernel_size=3, padding=1))
        if use_bn:
            layers.append(BatchNorm2D(64))
        layers.append(nn.ReLU(inplace=True))
        if use_dropout:
            layers.append(Dropout2D(0.3))

        # 第二个卷积层
        layers.append(nn.Conv2d(64, 128, kernel_size=3, padding=1))
        if use_bn:
            layers.append(BatchNorm2D(128))
        layers.append(nn.ReLU(inplace=True))
        if use_dropout:
            layers.append(Dropout2D(0.3))

        layers.append(nn.AdaptiveAvgPool2d(1))  # [B, 128, 1, 1]

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(128, 10)  # 假设是 CIFAR-10

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.classifier(x)

def evaluate_model(model, train_loader, test_loader, epochs=10):
    '''
    评估函数
    '''
    import torch.optim as optim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 测试准确率
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    acc = 100.0 * correct / total
    return acc

# 模块功能测试
if __name__ == '__main__':
    x = torch.randn(1, 64, 32, 32)

    print("BatchNorm2D:")
    bn = BatchNorm2D(64)
    print(bn(x).shape)  # -> torch.Size([1, 64, 32, 32])

    print("Dropout2D:")
    drop = Dropout2D(p=0.3)
    print(drop(x).shape)  # -> torch.Size([1, 64, 32, 32])

    print("ResidualBlock:")
    block = ResidualBlock(64, 64)
    print(block(x).shape)  # -> torch.Size([1, 64, 32, 32])

    print("SEBlock:")
    se = SEBlock(64)
    print(se(x).shape)  # -> torch.Size([1, 64, 32, 32])
    
    # 基础模型：无BN无Dropout
    model_plain = SimpleCNN(use_bn=False, use_dropout=False)

    # 启用BN
    model_bn = SimpleCNN(use_bn=True, use_dropout=False)

    # 启用BN + Dropout
    model_bn_dropout = SimpleCNN(use_bn=True, use_dropout=True)
    
    train_loader, test_loader, classes = get_cifar10_dataloaders(batch_size=128, num_workers=2)

    acc_plain = evaluate_model(model_plain, train_loader, test_loader, epochs=10)
    acc_bn = evaluate_model(model_bn, train_loader, test_loader, epochs=10)
    acc_bn_dropout = evaluate_model(model_bn_dropout, train_loader, test_loader, epochs=10)

    print("无BN/Dropout模型准确率:", acc_plain)
    print("启用BatchNorm模型准确率:", acc_bn)
    print("启用BatchNorm+Dropout模型准确率:", acc_bn_dropout)
