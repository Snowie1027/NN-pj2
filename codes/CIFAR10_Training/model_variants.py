import torch
import torch.nn as nn
import torch.nn.functional as F
from basic_layers import Conv2D, MaxPool2D, FullyConnected, ReLU
from advanced_layers import ResidualBlock, Dropout2D, BatchNorm2D, SEBlock
from data_loader import get_cifar10_dataloaders
from advanced_layers import evaluate_model

import torch.nn as nn

class CNNCustom(nn.Module):
    """
    自定义卷积神经网络模型，支持残差块、批归一化、注意力机制和多种激活函数选择。

    参数:
    - num_filters (tuple): 两个卷积层的输出通道数，格式为 (conv1_out_channels, conv2_out_channels)，默认(32, 64)
    - fc_neurons (int): 全连接层的神经元数量，默认128
    - dropout_rate (float): Dropout 概率，默认0.3
    - attention (bool): 是否使用注意力机制（SEBlock），默认 False
    - use_bn (bool): 是否使用批归一化，默认 True
    - use_resblock (bool): 是否使用残差块，默认 True
    - activation (str): 激活函数名称，支持 'ReLU', 'LeakyReLU', 'GELU'，默认 'ReLU'

    方法:
    - forward(x): 前向传播，输入张量 x，返回网络输出
    """
    def __init__(
        self,
        num_filters=(32, 64),
        fc_neurons=128,
        dropout_rate=0.3,
        attention=False,
        use_bn=True,
        use_resblock=True,
        activation='ReLU'   # 新增激活函数选择参数
    ):
        super(CNNCustom, self).__init__()
        conv1_out_channels, conv2_out_channels = num_filters

        self.conv1 = Conv2D(3, conv1_out_channels, kernel_size=3, padding=1)
        self.pool1 = MaxPool2D(2, 2)

        self.conv2 = Conv2D(conv1_out_channels, conv2_out_channels, kernel_size=3, padding=1)
        self.pool2 = MaxPool2D(2, 2)

        self.use_resblock = use_resblock
        if use_resblock:
            self.resblock1 = ResidualBlock(conv2_out_channels, conv2_out_channels)

        self.use_bn = use_bn
        if use_bn:
            self.bn = BatchNorm2D(conv2_out_channels)

        self.dropout = Dropout2D(dropout_rate)

        self.use_attention = attention
        if attention:
            self.attention = SEBlock(conv2_out_channels)

        self.fc1 = FullyConnected(conv2_out_channels * 8 * 8, fc_neurons)
        self.fc2 = FullyConnected(fc_neurons, 10)

        # 激活函数选择
        activation = activation.lower()
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(negative_slope=0.01)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.pool1(x)
        x = self.activation(self.conv2(x))
        x = self.pool2(x)

        if self.use_resblock:
            x = self.resblock1(x)
        if self.use_bn:
            x = self.bn(x)

        x = self.dropout(x)

        if self.use_attention:
            x = self.attention(x)

        x = x.flatten(1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class CustomCNN(torch.nn.Module):
    """
    自定义卷积神经网络模型，支持残差块、批归一化、注意力机制和多种激活函数选择。

    参数:
    - num_filters (tuple): 两个卷积层的输出通道数，格式为 (conv1_out_channels, conv2_out_channels)，默认(32, 64)
    - fc_neurons (int): 全连接层的神经元数量，默认128
    - dropout_rate (float): Dropout 概率，默认0.3
    - attention (bool): 是否使用注意力机制（SEBlock），默认 False
    - use_bn (bool): 是否使用批归一化，默认 True
    - use_resblock (bool): 是否使用残差块，默认 True
    - activation (str): 激活函数名称，支持 'ReLU', 'LeakyReLU', 'GELU'，默认 'ReLU'

    方法:
    - forward(x): 前向传播，输入张量 x，返回网络输出
    """

    def __init__(self, conv1_out_channels=32, conv2_out_channels=64, fc_units=128, dropout_rate=0.3, attention=False):
        super(CustomCNN, self).__init__()
        
        # 定义基本卷积层、池化层和激活层
        self.conv1 = Conv2D(3, conv1_out_channels, kernel_size=3, padding=1)
        self.pool1 = MaxPool2D(2, 2)
        self.relu = ReLU()

        # 第二层卷积
        self.conv2 = Conv2D(conv1_out_channels, conv2_out_channels, kernel_size=3, padding=1)
        self.pool2 = MaxPool2D(2, 2)
        
        # 残差块，BatchNorm 和 Dropout
        self.resblock1 = ResidualBlock(conv2_out_channels, conv2_out_channels)
        self.bn = BatchNorm2D(conv2_out_channels)
        self.dropout = Dropout2D(dropout_rate)

        # 是否使用注意力机制
        if attention:
            self.attention = SEBlock(conv2_out_channels)
        
        # 全连接层
        self.fc1 = FullyConnected(conv2_out_channels * 8 * 8, fc_units)
        self.fc2 = FullyConnected(fc_units, 10)  # CIFAR-10 是 10 类分类问题

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.resblock1(x)
        x = self.bn(x)
        x = self.dropout(x)

        if hasattr(self, 'attention'):
            x = self.attention(x)

        x = x.flatten(1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    def extract_feature_maps(self, x):
        fmap1 = self.relu(self.conv1(x))
        x = self.pool1(fmap1)
        fmap2 = self.relu(self.conv2(x))
        x = self.pool2(fmap2)
        return [fmap1, fmap2]

def get_loss_function(name='cross_entropy', l2_lambda=0.0, model=None, label_smoothing=0.0):
    """
    获取损失函数，支持 L2 正则和 Label Smoothing。

    参数:
    - name (str): 损失函数名称：'cross_entropy'、'label_smoothing'、'mse'
    - l2_lambda (float): L2 正则化系数，>0 时启用
    - model (nn.Module): 用于添加 L2 项
    - label_smoothing (float): 平滑系数，0 表示不启用

    返回:
    - loss_fn (function): (output, target) → loss
    """
    name = name.lower()

    if name == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif name == 'label_smoothing':
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    elif name == 'mse':
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss function: {name}")

    def loss_fn(output, target):
        base_loss = criterion(output, target)
        if l2_lambda > 0 and model is not None:
            l2_reg = sum(torch.norm(p) ** 2 for p in model.parameters() if p.requires_grad)
            return base_loss + l2_lambda * l2_reg
        else:
            return base_loss

    return loss_fn

def train_for_epochs(model, loss_fn, optimizer, train_loader, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu', num_epochs=10):
    model.to(device)
    train_acc_list = []
    test_acc_list = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total, correct = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        train_acc_list.append(train_acc)
        test_acc = evaluate_model(model, train_loader, test_loader)
        test_acc_list.append(test_acc)

        avg_loss = running_loss / total
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - Train Acc: {train_acc:.2f}% - Test Acc: {test_acc:.2f}%")
    
    return train_acc_list, test_acc_list

def test_selected_configurations(train_loader, test_loader, num_epochs=10):
    """
    训练模型若干个 epoch，并在每个 epoch 结束后评估测试集准确率。

    参数:
    - model (nn.Module): 待训练的神经网络模型
    - loss_fn (function): 损失函数，格式 (outputs, targets) -> loss
    - optimizer (torch.optim.Optimizer): 优化器实例
    - train_loader (DataLoader): 训练数据加载器
    - test_loader (DataLoader): 测试数据加载器
    - device (str): 设备标识，'cuda' 或 'cpu'，默认自动选择
    - num_epochs (int): 训练轮数，默认10

    返回:
    - train_acc_list (list of float): 每个 epoch 的训练集准确率列表
    - test_acc_list (list of float): 每个 epoch 的测试集准确率列表
    """
    configurations = [
        {
            "name": "CrossEntropy_NoL2",
            "loss_name": "cross_entropy",
            "l2_lambda": 0.0,
        },
        {
            "name": "LabelSmoothing_NoL2",
            "loss_name": "label_smoothing",
            "label_smoothing": 0.1,
            "l2_lambda": 0.0,
        },
        {
            "name": "CrossEntropy_L2",
            "loss_name": "cross_entropy",
            "l2_lambda": 1e-4,
        },
    ]

    for cfg in configurations:
        print(f"\n Training: {cfg['name']}")
        model = CNNCustom(
            attention=False,
            dropout_rate=0.0,
            use_bn=False,
            use_resblock=False
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        loss_fn = get_loss_function(
            name=cfg["loss_name"],
            l2_lambda=cfg.get("l2_lambda", 0.0),
            model=model,
            label_smoothing=cfg.get("label_smoothing", 0.0)
        )

        train_acc_list, test_acc_list = train_for_epochs(model, loss_fn, optimizer, train_loader, test_loader, num_epochs=num_epochs)

        print(f"\nFinal Test Accuracy for {cfg['name']}: {test_acc_list[-1]:.2f}%\n")

# 测试模型与损失函数
if __name__ == '__main__':
    # 构造测试输入（CIFAR-10 图像大小）
    x = torch.randn(4, 3, 32, 32)  # batch_size=4
    y = torch.tensor([1, 2, 3, 4], dtype=torch.long)

    # 初始化模型
    model = CNNCustom(num_filters=(32, 64), fc_neurons=256)
    out = model(x)
    print(f"Output shape: {out.shape}")

    # CIFAR-10 加载
    train_loader, test_loader, classes = get_cifar10_dataloaders(batch_size=256, num_workers=4)

    # 测试三种配置，绘制测试精度随 epoch 变化
    test_selected_configurations(train_loader, test_loader, num_epochs=10)

    model_relu = CNNCustom(activation='ReLU')
    model_leaky = CNNCustom(activation='LeakyReLU')
    model_gelu = CNNCustom(activation='GELU')

    train_loader, test_loader, classes = get_cifar10_dataloaders(batch_size=128, num_workers=2)

    acc_relu = evaluate_model(model_relu, train_loader, test_loader, epochs=25)
    acc_leaky = evaluate_model(model_leaky, train_loader, test_loader, epochs=25)
    acc_gelu = evaluate_model(model_gelu, train_loader, test_loader, epochs=25)

    print("启用relu模型准确率:", acc_relu)
    print("启用leaky模型准确率:", acc_leaky)
    print("启用gelu模型准确率:", acc_gelu)
