import torch
import torchvision
import torchvision.transforms as transforms

def get_cifar10_dataloaders(batch_size=4, num_workers=2, data_dir='./data'):
    """
    下载并加载 CIFAR-10 数据集。

    参数:
    - batch_size (int): 每个 batch 的样本数。
    - num_workers (int): 加载数据使用的线程数。
    - data_dir (str): 数据集保存目录。

    返回:
    - trainloader: 训练集 DataLoader
    - testloader: 测试集 DataLoader
    - classes: CIFAR-10 的类别标签
    """
    
    # 数据增强与归一化（训练和测试均使用）
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


    # 加载训练集
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform)
    
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    # 加载测试集
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform)
    
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # CIFAR-10 类别
    classes = (
        'plane', 'car', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    )

    return trainloader, testloader, classes


# 测试数据加载
if __name__ == '__main__':
    trainloader, testloader, classes = get_cifar10_dataloaders()
    print(f"Trainloader length: {len(trainloader)} batches")
    print(f"Testloader length: {len(testloader)} batches")
    print(f"Classes: {classes}")
