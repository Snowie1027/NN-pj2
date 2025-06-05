import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
from IPython import display

from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm
from data.loaders import get_cifar_loader

# ## Constants (parameters) initialization
device_id = [0,1,2,3]
num_workers = 4
batch_size = 512

# add our package dir to path 
module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')

# Make sure you are using the right device.
device_id = device_id
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name(0))



# This function is used to calculate the accuracy of model classification
def get_accuracy(model, data_loader):
    ## --------------------
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total
    ## --------------------

# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0, device='gpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# We use this function to complete the entire
# training process. In order to plot the loss landscape,
# you need to record the loss value of each step.
# Of course, as before, you can test your model
# after drawing a training round and save the curve
# to observe the training
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None):
    model.to(device)
    learning_curve = [np.nan] * epochs_n
    train_accuracy_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)
    losses_list = []
    grads = []
    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()

        correct_train = 0
        total_train = 0

        loss_list = []  # use this to record the loss value of each step
        grad = []  # use this to record the loss gradient of each step
        learning_curve[epoch] = 0  # maintain this to plot the training curve

        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            # You may need to record some variable values here
            # if you want to get loss gradient, use
            # grad = model.classifier[4].weight.grad.clone()
            ## --------------------
            loss.backward()
            # 获取中间某一层的梯度，例如 classifier[4] 是 Linear 层
            if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
                if len(model.classifier) > 4 and hasattr(model.classifier[4], 'weight'):
                    grad.append(model.classifier[4].weight.grad.clone().cpu().numpy())
            optimizer.step()
            # 累加损失
            learning_curve[epoch] += loss.item()
            loss_list.append(loss.item())

            pred = torch.argmax(prediction, dim=1)
            correct_train += (pred == y).sum().item()
            total_train += y.size(0)
            ## --------------------
        
        train_accuracy = correct_train / total_train
        train_accuracy_curve[epoch] = train_accuracy

        losses_list.append(loss_list)
        grads.append(grad)
        display.clear_output(wait=True)
        f, axes = plt.subplots(1, 2, figsize=(15, 3))

        learning_curve[epoch] /= batches_n
        axes[0].plot(learning_curve)

        # Test your model and save figure here (not required)
        # remember to use model.eval()
        ## --------------------
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                out_val = model(x_val)
                pred_val = torch.argmax(out_val, dim=1)
                correct_val += (pred_val == y_val).sum().item()
                total_val += y_val.size(0)
        val_accuracy = correct_val / total_val
        val_accuracy_curve[epoch] = val_accuracy

        if val_accuracy > max_val_accuracy and best_model_path is not None:
            torch.save(model.state_dict(), best_model_path)
            max_val_accuracy = val_accuracy
            max_val_accuracy_epoch = epoch

        axes[1].plot(train_accuracy_curve, label='Train Acc')
        axes[1].plot(val_accuracy_curve, label='Val Acc')
        axes[1].legend()
        # plt.show()

        print(f"[Epoch {epoch+1}/{epochs_n}] Loss: {learning_curve[epoch]:.4f} | "
              f"Train Acc: {train_accuracy:.4f} | Val Acc: {val_accuracy:.4f}")
        ## --------------------

    return losses_list, grads, train_accuracy_curve, val_accuracy_curve

# Use this function to plot the final loss landscape,
# fill the area between the two curves can use plt.fill_between()
def plot_loss_landscape(min_curve, max_curve):
    plt.figure(figsize=(12, 6))
    steps = list(range(len(min_curve)))
    plt.plot(steps, max_curve, label='Max Loss', color='red')
    plt.plot(steps, min_curve, label='Min Loss', color='blue')
    plt.fill_between(steps, min_curve, max_curve, color='purple', alpha=0.3)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss Landscape')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(figures_path, "loss_landscape.png"))
    plt.close()

os.makedirs(figures_path, exist_ok=True)

def plot_all_loss_landscapes(loss_curves, save_path):
    steps = np.arange(len(next(iter(loss_curves.values()))['min']))
    plt.figure(figsize=(14, 7))

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']  # 给前几个模型定义颜色
    for i, (model_name, curves) in enumerate(loss_curves.items()):
        color = colors[i % len(colors)]
        min_curve = curves['min']
        max_curve = curves['max']
        
        plt.plot(steps, max_curve, label=f'{model_name} Max Loss', linestyle='--', color=color)
        plt.plot(steps, min_curve, label=f'{model_name} Min Loss', linestyle='-', color=color)
        plt.fill_between(steps, min_curve, max_curve, alpha=0.2, color=color)

    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Loss Landscape Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    print(f"Saving combined loss landscape to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.close()

if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import torch.nn as nn
    import torch

    # 创建输出目录
    loss_save_path = './VGG_BatchNorm/loss_path'
    train_accs_path = './VGG_BatchNorm/train_accs_path'
    test_accs_path = './VGG_BatchNorm/test_accs_path'
    grad_save_path = './VGG_BatchNorm/grad_path'
    figures_path = './VGG_BatchNorm/figures'
    os.makedirs(loss_save_path, exist_ok=True)
    os.makedirs(train_accs_path, exist_ok=True)
    os.makedirs(test_accs_path, exist_ok=True)
    os.makedirs(grad_save_path, exist_ok=True)
    os.makedirs(figures_path, exist_ok=True)

    train_loader = get_cifar_loader(train=True)
    val_loader = get_cifar_loader(train=False)

    # 模型列表，可以扩展
    models_to_compare = {
        'VGG_A': VGG_A,
        'VGG_A_BatchNorm': VGG_A_BatchNorm
    }

    learning_rates = [1e-3, 5e-4, 1e-4]
    epochs = 30
    
    loss_curves = {}

    # 对所有模型做对比
    for model_name, model_class in models_to_compare.items():
        print(f"\n===== Training model: {model_name} =====")
        all_losses = []
        all_train_accs = []
        all_test_accs = []


        for lr in learning_rates:
            print(f"Training with learning rate: {lr}")
            model = model_class()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            loss, grads, train_accs, test_accs = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epochs)

            # all_losses.append([np.mean(epoch_loss) for epoch_loss in loss])
            all_losses.append(loss)
            all_train_accs.append(train_accs)
            all_test_accs.append(test_accs)

            # 保存 loss/acc
            np.savetxt(os.path.join(loss_save_path, f'{model_name}_loss_lr{lr}.txt'), all_losses[-1], fmt='%.6f')
            np.savetxt(os.path.join(train_accs_path, f'{model_name}_train_acc_lr{lr}.txt'), train_accs, fmt='%.6f')
            np.savetxt(os.path.join(test_accs_path, f'{model_name}_test_acc_lr{lr}.txt'), test_accs, fmt='%.6f')

            # 保存梯度范数
            grad_norms = []
            for epoch_grads in grads:
                flat_grads = [g.flatten() for g in epoch_grads if g is not None]
                if flat_grads:
                    all_grads = np.concatenate(flat_grads)
                    grad_norms.append(np.linalg.norm(all_grads))
                else:
                    grad_norms.append(0.0)
            grad_file = os.path.join(grad_save_path, f'{model_name}_grad_norms_lr{lr}.txt')
            np.savetxt(grad_file, grad_norms, fmt='%.6f')

        all_losses = np.array(all_losses)
        all_train_accs = np.array(all_train_accs)
        all_test_accs = np.array(all_test_accs)

        # 保存 loss 数据
        loss_file = os.path.join(loss_save_path, f'{model_name}_loss_all.txt')
        np.savetxt(loss_file, all_losses.reshape(len(learning_rates), -1), fmt='%.6f')

        # 保存 train_accs 数据
        train_accs_file = os.path.join(train_accs_path, f'{model_name}_train_accs_all.txt')
        np.savetxt(train_accs_file, all_train_accs.reshape(len(learning_rates), -1), fmt='%.6f')

        # 保存 test_accs 数据
        test_accs_file = os.path.join(test_accs_path, f'{model_name}_test_accs_all.txt')
        np.savetxt(test_accs_file, all_test_accs.reshape(len(learning_rates), -1), fmt='%.6f')


        # 计算最大/最小 loss 曲线
        max_curve = np.max(all_losses, axis=0).flatten()
        min_curve = np.min(all_losses, axis=0).flatten()

        # 存储到字典中
        loss_curves[model_name] = {'min': min_curve, 'max': max_curve}

        # 绘图函数，支持传入文件名和标题
        def plot_loss_landscape(min_curve, max_curve, model_name):
            steps = np.arange(len(min_curve))
            plt.figure(figsize=(12, 6))
            plt.plot(steps, max_curve, label='Max Loss', color='#e41a1c', linewidth=2)  # 红色
            plt.plot(steps, min_curve, label='Min Loss', color='#377eb8', linewidth=2)  # 蓝色
            plt.fill_between(steps, min_curve, max_curve, color='#4daf4a', alpha=0.3, label='Gap')  # 绿色填充
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.title(f'Loss Landscape: {model_name}', fontsize=14, fontweight='bold')
            plt.legend(fontsize=11)
            plt.grid(True, linestyle='--', alpha=0.5)
            save_path = os.path.join(figures_path, f'{model_name}_loss_landscape.png')
            print(f"Saving loss landscape to {save_path}")
            plt.savefig(save_path, dpi=300)
            plt.close()

        # 绘制当前模型的 loss landscape
        plot_loss_landscape(min_curve, max_curve, model_name)
    
    plot_all_loss_landscapes(loss_curves, os.path.join(figures_path, 'combined_loss_landscape.png'))


    # 所有模型 loss 对比图
    plt.figure(figsize=(12, 6))
    for model_name, model_class in models_to_compare.items():
        loss_path = os.path.join(loss_save_path, f'{model_name}_loss_all.txt')
        all_losses = np.loadtxt(loss_path).reshape(len(learning_rates), epochs)
        avg_loss_curve = all_losses.mean(axis=0)
        plt.plot(np.arange(epochs), avg_loss_curve, label=model_name)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Loss Curve Comparison Between Models')
    plt.legend()
    save_path = os.path.join(figures_path, 'models_loss_comparison.png')
    print(f"Saving comparison plot to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.close()

    for lr in learning_rates:
        plt.figure(figsize=(12, 6))
        for model_name in models_to_compare:
            loss_file = os.path.join(loss_save_path, f'{model_name}_loss_lr{lr}.txt')
            if os.path.exists(loss_file):
                loss_curve = np.loadtxt(loss_file)
                plt.plot(loss_curve, label=model_name)
        plt.title(f'Loss Comparison @ lr={lr}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(figures_path, f'loss_comparison_lr{lr}.png'))
        plt.close()
    
    # train accuracy
    for lr in learning_rates:
        plt.figure(figsize=(12, 6))
        for model_name in models_to_compare:
            train_accs_file = os.path.join(train_accs_path, f'{model_name}_train_acc_lr{lr}.txt')
            if os.path.exists(train_accs_file):
                train_acc_curve = np.loadtxt(train_accs_file)
                plt.plot(train_acc_curve, label=model_name)
        plt.title(f'Train Accuracy Comparison @ lr={lr}')
        plt.xlabel('Epoch')
        plt.ylabel('Train Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(figures_path, f'train_acc_comparison_lr{lr}.png'))
        plt.close()

    # test accuracy
    for lr in learning_rates:
        plt.figure(figsize=(12, 6))
        for model_name in models_to_compare:
            test_accs_file = os.path.join(test_accs_path, f'{model_name}_test_acc_lr{lr}.txt')
            if os.path.exists(test_accs_file):
                test_acc_curve = np.loadtxt(test_accs_file)
                plt.plot(test_acc_curve, label=model_name)
        plt.title(f'Test Accuracy Comparison @ lr={lr}')
        plt.xlabel('Epoch')
        plt.ylabel('Test Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(figures_path, f'test_acc_comparison_lr{lr}.png'))
        plt.close()

    for model_name in models_to_compare:
        plt.figure(figsize=(12, 6))
        for lr in learning_rates:
            loss_file = os.path.join(loss_save_path, f'{model_name}_loss_lr{lr}.txt')
            if os.path.exists(loss_file):
                loss_curve = np.loadtxt(loss_file)
                plt.plot(loss_curve, label=f'lr={lr}')
        plt.title(f'Loss Comparison for {model_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(figures_path, f'{model_name}_loss_lr_comparison.png'))
        plt.close()

    for model_name in models_to_compare:
        plt.figure(figsize=(12, 6))
        for lr in learning_rates:
            train_acc_file = os.path.join(train_accs_path, f'{model_name}_train_acc_lr{lr}.txt')
            if os.path.exists(loss_file):
                train_acc_curve = np.loadtxt(train_acc_file)
                plt.plot(train_acc_curve, label=f'lr={lr}')
        plt.title(f'Train Accuracy Comparison for {model_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Train Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(figures_path, f'{model_name}_train_acc_lr_comparison.png'))
        plt.close()

    for model_name in models_to_compare:
        plt.figure(figsize=(12, 6))
        for lr in learning_rates:
            test_acc_file = os.path.join(test_accs_path, f'{model_name}_test_acc_lr{lr}.txt')
            if os.path.exists(loss_file):
                test_acc_curve = np.loadtxt(test_acc_file)
                plt.plot(test_acc_curve, label=f'lr={lr}')
        plt.title(f'Test Accuracy Comparison for {model_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Test Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(figures_path, f'{model_name}_test_acc_lr_comparison.png'))
        plt.close()
