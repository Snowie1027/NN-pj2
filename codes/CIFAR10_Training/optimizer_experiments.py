import torch
import torch.nn as nn
import torch.nn.functional as F
from basic_layers import Conv2D, MaxPool2D, ReLU, FullyConnected
from advanced_layers import ResidualBlock, Dropout2D, BatchNorm2D, SEBlock
from data_loader import get_cifar10_dataloaders

class FullModel(nn.Module):
    """
    基于卷积、残差块和SE注意力机制的图像分类模型。

    参数:
    - num_classes (int): 输出类别数，默认10。
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.block1 = nn.Sequential(
            Conv2D(3, 64, 3, 1, 1),
            BatchNorm2D(64),
            ReLU(),
            Dropout2D(0.1)
        )
        self.block2 = nn.Sequential(
            ResidualBlock(64, 64),
            SEBlock(64),
            MaxPool2D(2)
        )
        self.block3 = nn.Sequential(
            ResidualBlock(64, 128, stride=2, downsample=nn.Sequential(
                nn.Conv2d(64, 128, 1, 2),
                nn.BatchNorm2d(128)
            )),
            SEBlock(128),
            MaxPool2D(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4, 256),  # 修正输入维度
            ReLU(),
            Dropout2D(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)

class SimpleSGD:
    """
    自定义简单带动量的 SGD 优化器。

    参数:
    - params: 模型参数迭代器。
    - lr (float): 学习率。
    - momentum (float): 动量系数。
    """
    def __init__(self, params, lr=0.01, momentum=0.0):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.v = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            self.v[i] = self.momentum * self.v[i] - self.lr * p.grad
            p.data += self.v[i]

def run_all_optimizers(num_epochs=10, batch_size=128, lr=0.001, optimizer_choice='adam'):
    """
    训练并测试 FullModel，支持三种优化器选择。

    参数:
    - num_epochs (int): 训练轮数。
    - batch_size (int): 批大小。
    - lr (float): 学习率。
    - optimizer_choice (str): 优化器选择，'adam', 'sgd' 或 'custom'。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 获取数据加载器
    trainloader, testloader, classes = get_cifar10_dataloaders(batch_size=batch_size, num_workers=2)

    # --- 模型 ---
    model = FullModel().to(device)
    criterion = nn.CrossEntropyLoss()

    # --- 选择优化器 ---
    if optimizer_choice == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_choice == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_choice == 'custom':
        optimizer = SimpleSGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError("Unknown optimizer: choose from 'adam', 'sgd', 'custom'.")

    # --- 训练 ---
    for epoch in range(num_epochs):
        model.train()
        correct, total, running_loss = 0, 0, 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        acc = 100. * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(trainloader):.4f} | Train Acc: {acc:.2f}%")

    # --- 测试 ---
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    print(f"Test Accuracy ({optimizer_choice}): {100.*correct/total:.2f}%")

    # 保存模型
    torch.save(model.state_dict(), f'model_{optimizer_choice}.pth')
    print(f"✅ 模型保存为 model_{optimizer_choice}.pth")


# 测试数据调用
if __name__ == '__main__':
    print("\n torch.optim.Adam")
    run_all_optimizers(optimizer_choice='adam')

    print("\n torch.optim.SGD")
    run_all_optimizers(optimizer_choice='sgd')

    print("\n 自定义 SimpleSGD")
    run_all_optimizers(optimizer_choice='custom')
