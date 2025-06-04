import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def load_grad_norms(file_path):
    """
    每一行是一个 epoch 的整体梯度 L2 范数
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        norms = [float(line.strip()) for line in lines]
    return np.array(norms)

def compute_max_grad_diff(norms):
    """
    相邻 epoch 的最大梯度范数差
    """
    diffs = np.abs(norms[1:] - norms[:-1])
    max_diff = np.max(diffs)
    return max_diff, diffs

def plot_grad_norms(norm_dict, title, filename):
    plt.figure(figsize=(10, 6))
    for label, values in norm_dict.items():
        plt.plot(values, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Gradient L2 Norm")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    plt.close()

def plot_grad_diff(diff_dict, title, filename):
    plt.figure(figsize=(10, 6))
    for label, diffs in diff_dict.items():
        plt.plot(diffs, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Gradient Norm Change")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    plt.close()

# 主目录
base_dir = "/Users/suxueqing/Desktop/PJ2/codes/VGG_BatchNorm/grad_path"
file_list = glob(os.path.join(base_dir, "*.txt"))

grad_norms_dict = {}
grad_diff_dict = {}
max_diff_table = {}

for file_path in file_list:
    fname = os.path.basename(file_path)
    label = fname.replace("VGG_", "").replace("_grad_norms_", " LR=").replace(".txt", "")
    
    norms = load_grad_norms(file_path)
    max_diff, diffs = compute_max_grad_diff(norms)

    grad_norms_dict[label] = norms
    grad_diff_dict[label] = diffs
    max_diff_table[label] = max_diff

# 绘图
plot_grad_norms(grad_norms_dict, "Gradient Norms over Epochs", "gradient_norms.png")
plot_grad_diff(grad_diff_dict, "Gradient Norm Differences between Epochs", "gradient_diffs.png")

# 打印最大差值表格
print("\n Maximum Gradient Norm Change Table")
for label, value in max_diff_table.items():
    print(f"{label:<40}: {value:.4f}")
