import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import get_cifar10_dataloaders
from basic_layers import Conv2D, MaxPool2D, ReLU, FullyConnected
from advanced_layers import ResidualBlock, Dropout2D, BatchNorm2D, SEBlock
from model_variants import CustomCNN, get_loss_function
from optimizer_experiments import SimpleSGD  # 导入自定义优化器
import torch.optim as optim  # 导入 PyTorch 优化器
import pandas as pd
import os

# 创建文件夹（如果不存在）
model_save_dir = './1/model_results'
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

# 定义保存模型的函数
def save_model(model, model_name, params):
    """
    保存训练好的模型和相关参数到指定路径。

    参数:
    - model: 训练好的模型
    - model_name (str): 模型文件的保存名称
    - params (dict): 包含训练参数的字典
    """
    model_path = os.path.join(model_save_dir, model_name)
    
    # 将模型的状态字典和超参数保存为字典
    torch.save({
        'model_state_dict': model.state_dict(),
        'params': params
    }, model_path)
    
    print(f"模型已保存至 {model_path}")

# 选择优化器的函数
def get_optimizer(model, optimizer_choice='adam', lr=0.001, momentum=0.9):
    if optimizer_choice == 'adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer_choice == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer_choice == 'custom':
        return SimpleSGD(model.parameters(), lr=lr, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer choice: {optimizer_choice}")

def extract_feature_maps(self, x):
    fmap1 = self.conv1(x)
    fmap2 = self.conv2(self.pool(self.relu(fmap1)))
    return [fmap1, fmap2]

# 在超参数搜索和训练过程中，保存模型
def hyperparameter_search(batch_sizes, lr_values, num_epochs_values, 
                          conv1_out_channels_values, conv2_out_channels_values, 
                          fc_units_values, dropout_values, attention_values, 
                          optimizer_choices, model_save_dir="saved_models"):

    os.makedirs(model_save_dir, exist_ok=True)

    best_accuracy = 0
    best_params = {}

    # 加载 CIFAR-10
    trainloader, testloader, classes = get_cifar10_dataloaders(batch_size=32)

    results = []  # 保存最终结果

    for batch_size in batch_sizes:
        for lr in lr_values:
            for num_epochs in num_epochs_values:
                for conv1_out_channels in conv1_out_channels_values:
                    for conv2_out_channels in conv2_out_channels_values:
                        for fc_units in fc_units_values:
                            for dropout_rate in dropout_values:
                                for attention in attention_values:
                                    for optimizer_choice in optimizer_choices:

                                        print(f"\n--- Training with batch_size={batch_size}, lr={lr}, "
                                              f"epochs={num_epochs}, conv1={conv1_out_channels}, "
                                              f"conv2={conv2_out_channels}, fc={fc_units}, "
                                              f"dropout={dropout_rate}, attention={attention}, "
                                              f"optimizer={optimizer_choice} ---")

                                        model = CustomCNN(conv1_out_channels, conv2_out_channels, fc_units, dropout_rate, attention).to(device)
                                        criterion = get_loss_function('cross_entropy', l2_lambda=1e-4, model=model)
                                        optimizer = get_optimizer(model, optimizer_choice=optimizer_choice, lr=lr)

                                        train_logs = []

                                        for epoch in range(num_epochs):
                                            model.train()
                                            running_loss = 0.0
                                            correct, total = 0, 0

                                            for inputs, labels in trainloader:
                                                inputs, labels = inputs.to(device), labels.to(device)
                                                optimizer.zero_grad()

                                                outputs = model(inputs)
                                                loss = criterion(outputs, labels)
                                                loss.backward()

                                                # 保存梯度信息ex
                                                for name, param in model.named_parameters():
                                                    if param.grad is not None:
                                                        grad_path = os.path.join(model_save_dir, f"grads_model_epoch{epoch+1}_{name.replace('.', '_')}.pt")
                                                        torch.save(param.grad.cpu(), grad_path)

                                                optimizer.step()
                                                running_loss += loss.item()
                                                _, predicted = outputs.max(1)
                                                total += labels.size(0)
                                                correct += predicted.eq(labels).sum().item()

                                            train_accuracy = 100. * correct / total
                                            print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(trainloader):.4f} | Train Accuracy: {train_accuracy:.2f}%")

                                            # 测试阶段
                                            model.eval()
                                            correct, total = 0, 0
                                            with torch.no_grad():
                                                for inputs, labels in testloader:
                                                    inputs, labels = inputs.to(device), labels.to(device)
                                                    outputs = model(inputs)
                                                    _, predicted = outputs.max(1)
                                                    total += labels.size(0)
                                                    correct += predicted.eq(labels).sum().item()

                                            test_accuracy = 100. * correct / total

                                            # 保存 train 日志
                                            train_logs.append({
                                                'epoch': epoch + 1,
                                                'loss': running_loss / len(trainloader),
                                                'accuracy': train_accuracy,
                                                'test_accuracy': test_accuracy
                                            })

                                            # 每 5 个 epoch 保存特征图
                                            if epoch % 5 == 0:
                                                with torch.no_grad():
                                                    sample_inputs, _ = next(iter(trainloader))
                                                    sample_inputs = sample_inputs[:1].to(device)
                                                    feature_maps = model.extract_feature_maps(sample_inputs)
                                                    for idx, fmap in enumerate(feature_maps):
                                                        fmap_path = os.path.join(model_save_dir, f"featuremap_layer{idx}_epoch{epoch}_model_{optimizer_choice}.pt")
                                                        torch.save(fmap.cpu(), fmap_path)

                                        print(f"Final Test Accuracy: {test_accuracy:.2f}%")

                                        # 保存最终模型
                                        model_name = (f"model_batch{batch_size}_lr{lr}_conv1{conv1_out_channels}_"
                                                      f"conv2{conv2_out_channels}_fc{fc_units}_dropout{dropout_rate}_"
                                                      f"attention{attention}_optimizer{optimizer_choice}.pth")
                                        torch.save(model.state_dict(), os.path.join(model_save_dir, model_name))

                                        # 保存 train_logs 日志 CSV
                                        log_df = pd.DataFrame(train_logs)
                                        log_path = os.path.join(model_save_dir, f"log_{model_name.replace('.pth', '.csv')}")
                                        log_df.to_csv(log_path, index=False)

                                        # 记录每次搜索结果
                                        results.append({
                                            "batch_size": batch_size,
                                            "lr": lr,
                                            "num_epochs": num_epochs,
                                            "conv1_out_channels": conv1_out_channels,
                                            "conv2_out_channels": conv2_out_channels,
                                            "fc_units": fc_units,
                                            "dropout_rate": dropout_rate,
                                            "attention": attention,
                                            "optimizer": optimizer_choice,
                                            "test_accuracy": test_accuracy
                                        })

                                        # 保存最佳模型
                                        if test_accuracy > best_accuracy:
                                            best_accuracy = test_accuracy
                                            best_params = {
                                                "batch_size": batch_size,
                                                "lr": lr,
                                                "num_epochs": num_epochs,
                                                "conv1_out_channels": conv1_out_channels,
                                                "conv2_out_channels": conv2_out_channels,
                                                "fc_units": fc_units,
                                                "dropout_rate": dropout_rate,
                                                "attention": attention,
                                                "optimizer": optimizer_choice
                                            }
                                            # 保存最佳权重
                                            torch.save(model.state_dict(), os.path.join(model_save_dir, 'best_model_weights.pth'))

    print(f"\nBest Accuracy: {best_accuracy:.2f}%")
    print(f"Best Hyperparameters: {best_params}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(model_save_dir, 'hyperparameter_search_results.csv'), index=False)

    return best_params, best_accuracy

# 主函数
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设置要搜索的超参数范围
    batch_sizes = [256]
    lr_values = [0.001]
    num_epochs_values = [30]
    conv1_out_channels_values = [32]
    conv2_out_channels_values = [128]
    fc_units_values = [128]
    dropout_values = [0.3]
    attention_values = [True]
    optimizer_choices = ['adam']

    # 调用超参数搜索函数
    best_params, best_accuracy = hyperparameter_search(
        batch_sizes, lr_values, num_epochs_values, 
        conv1_out_channels_values, conv2_out_channels_values, 
        fc_units_values, dropout_values, attention_values, optimizer_choices
    )