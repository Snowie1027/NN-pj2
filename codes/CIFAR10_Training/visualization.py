import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model_variants import CustomCNN
import os
import numpy as np
import pandas as pd
import copy

def visualize_first_layer_filters(checkpoint_path, save_dir="filters"):
    """
    可视化模型第一层卷积滤波器权重。

    参数:
    - checkpoint_path (str): 模型权重文件路径，包含模型参数及状态字典。
    - save_dir (str): 保存滤波器图像的文件夹，默认"filters"。
    """
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    full_params = checkpoint.get('params', {})
    allowed_keys = ['conv1_out_channels', 'conv2_out_channels', 'fc_units', 'dropout_rate', 'attention']
    model_params = {k: v for k, v in full_params.items() if k in allowed_keys}

    model = CustomCNN(**model_params)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    filters = model.conv1.conv.weight.data.cpu().numpy()
    filters = (filters - filters.min()) / (filters.max() - filters.min() + 1e-5)

    num_filters = filters.shape[0]
    num_cols = 8
    num_rows = (num_filters + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 1.5, num_rows * 1.5))

    for i in range(num_filters):
        row, col = divmod(i, num_cols)
        ax = axes[row][col] if num_rows > 1 else axes[col]
        filter_img = filters[i]
        if filter_img.shape[0] == 3:
            filter_img = np.transpose(filter_img, (1, 2, 0))
        else:
            filter_img = filter_img[0]
        ax.imshow(filter_img)
        ax.axis('off')

    for j in range(i + 1, num_rows * num_cols):
        row, col = divmod(j, num_cols)
        ax = axes[row][col] if num_rows > 1 else axes[col]
        ax.axis('off')

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "conv1_filters.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Filters saved to {save_path}")
    plt.show()


def visualize_loss_landscape(model, dataloader, criterion, steps=20, epsilon=1.0):
    """
    基于两个随机方向，计算并可视化模型参数空间的损失曲面（Loss Landscape）。

    参数:
    - model (torch.nn.Module): 训练好的模型实例。
    - dataloader (torch.utils.data.DataLoader): 用于计算损失的数据加载器。
    - criterion (torch.nn.Module): 损失函数实例（如CrossEntropyLoss）。
    - steps (int): 在两个方向上采样的步数，默认20。
    - epsilon (float): 参数空间扰动幅度，默认1.0。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    model.eval()
    weights = torch.nn.utils.parameters_to_vector(model.parameters()).detach()

    # 生成两个单位向量方向
    direction1 = torch.randn_like(weights)
    direction1 /= direction1.norm()
    direction2 = torch.randn_like(weights)
    direction2 /= direction2.norm()

    loss_surface = np.zeros((steps, steps))

    x_range = np.linspace(-epsilon, epsilon, steps)
    y_range = np.linspace(-epsilon, epsilon, steps)

    for i, alpha in enumerate(x_range):
        for j, beta in enumerate(y_range):
            perturbed_weights = weights + alpha * direction1 + beta * direction2
            # 应用扰动后的权重
            new_model = copy.deepcopy(model)
            torch.nn.utils.vector_to_parameters(perturbed_weights, new_model.parameters())

            total_loss = 0.0
            total_samples = 0

            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = new_model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)

            loss_surface[i, j] = total_loss / total_samples

    # 可视化 Loss Surface
    X, Y = np.meshgrid(x_range, y_range)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, loss_surface, cmap='viridis')
    ax.set_xlabel('Direction 1')
    ax.set_ylabel('Direction 2')
    ax.set_zlabel('Loss')
    plt.title("Loss Landscape")
    plt.tight_layout()
    plt.show()

def plot_loss_accuracy(csv_path, save_dir="figures"):
    """
    可视化模型损失及准确率。
    """
    # 读取日志数据
    df = pd.read_csv(csv_path)

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    epochs = df['epoch']
    train_loss = df['loss']
    train_acc = df['accuracy']
    test_acc = df['test_accuracy']

    # Loss 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_loss, label='Train Loss', color='blue', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.legend()
    loss_path = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(loss_path)
    plt.close()
    print(f"Loss curve saved to {loss_path}")

    # Accuracy 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_acc, label='Train Accuracy', color='green', linewidth=2)
    plt.plot(epochs, test_acc, label='Test Accuracy', color='orange', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Train/Test Accuracy Curve')
    plt.grid(True)
    plt.legend()
    acc_path = os.path.join(save_dir, "accuracy_curve.png")
    plt.savefig(acc_path)
    plt.close()
    print(f"Accuracy curve saved to {acc_path}")

    # Loss + Accuracy 综合图
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 左 y 轴：Loss
    ax1.plot(epochs, train_loss, 'b-', label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # 右 y 轴：Accuracy
    ax2 = ax1.twinx()
    ax2.plot(epochs, train_acc, 'g--', label='Train Acc')
    ax2.plot(epochs, test_acc, 'r--', label='Test Acc')
    ax2.set_ylabel('Accuracy (%)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title("Training Loss & Accuracy Over Epochs")
    fig.tight_layout()
    fig.legend(loc="upper right", bbox_to_anchor=(1, 0.9))
    both_path = os.path.join(save_dir, "loss_accuracy_combined.png")
    plt.savefig(both_path)
    plt.close()
    print(f"Combined loss & accuracy plot saved to {both_path}")

if __name__ == "__main__":

    # 模型路径
    ckpt_path = './1/model_results/model_batch256_lr0.01_conv132_conv2128_fc128_dropout0.3_attentionTrue_optimizeradam.pth'

    # 可视化滤波器
    visualize_first_layer_filters(ckpt_path)

    # 加载模型并构建
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    params = {k: v for k, v in checkpoint.get("params", {}).items()
              if k in ['conv1_out_channels', 'conv2_out_channels', 'fc_units', 'dropout_rate', 'attention']}
    model = CustomCNN(**params)

    model.load_state_dict(checkpoint['model_state_dict'])

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 加载 CIFAR-10 验证集
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    val_dataset = CIFAR10(root='./data', train=False, transform=transform, download=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # 损失函数
    criterion = torch.nn.CrossEntropyLoss()

    # 可视化 Loss Landscape
    visualize_loss_landscape(model, val_loader, criterion)

    csv_file = r'.\saved_models\log_model_batch256_lr0.001_conv132_conv2128_fc128_dropout0.3_attentionTrue_optimizeradam.csv'
    plot_loss_accuracy(csv_file)
