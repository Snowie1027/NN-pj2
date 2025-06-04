# Project-2 Report: Neural Network and Deep Learning

**Instructor:** Prof. Yanwei Fu  
**Student Name:** 苏雪清  
**Student ID:** 22307130420  
**Date:** June 4, 2025  
**GitHub Link:** [待填写]  
**Dataset & Model Link:** [待填写]  

---

## 摘要

本项目围绕 CIFAR-10 图像分类任务，基于 PyTorch 框架训练并分析卷积神经网络。在第一部分中，构建了包含全连接层、卷积层、池化层、激活函数与 Batch Normalization 等组件的神经网络，并系统地研究了不同超参数设置、损失函数、正则化策略和优化器对模型性能的影响。第二部分重点分析了 Batch Normalization 的作用机制与实际效果，通过实现类 VGG 网络并对比有无 BN 的训练过程，深入探讨其对网络优化与收敛速度的影响。实验结果表明，Batch Normalization 显著提升了训练效率与最终精度，验证了其在现代深度学习模型中的重要价值。

---

## 目录

- [1. 项目结构](#1-项目结构)  
- [2. 数据准备与预处理](#2-数据准备与预处理)  
- [3. 网络结构设计](#3-网络结构设计)  
- [4. 高级组件设计](#4-高级组件设计)  
- [5. 优化策略实验](#5-优化策略实验)  
- [6. 结果与分析](#6-结果与分析)  
- [7. 如何运行](#7-如何运行)  
- [8. 参考文献](#8-参考文献)  

---

## 1. 项目结构

codes/
├── CIFAR10_Training/
│ ├── data_loader.py # 数据加载与预处理代码
│ ├── basic_layers.py # 基础卷积神经网络模块
│ ├── advanced_layers.py # BatchNorm、Dropout及其他高级模块实现
│ ├── model_variants.py # 不同模型结构与优化策略实验代码
│ ├── train.py # 训练脚本
│ └── test.py # 测试脚本
├── VGG_BatchNorm
│ ├── data/
│ │ ├── cifar-10-python.tar.gz
│ │ └── loaders.py
│ ├── models
│ │ └── vgg.py
│ ├── utils
│ │ └──nn.py
│ ├── Gradient_Norm.py
└─└── VGG_Loss_Landscape.py


---

## 2. 数据准备与预处理

- 使用 PyTorch 的 `torchvision.datasets.CIFAR10` 加载 CIFAR-10 数据集。
- 训练集进行数据增强：随机裁剪（`RandomCrop(32, padding=4)`）、随机水平翻转（`RandomHorizontalFlip()`）。
- 归一化处理，通道均值与标准差均为 0.5。
- 训练集和测试集分别用 `shuffle=True` 和 `shuffle=False` 方式加载。
- 类别均衡，训练集中每个类别均为 5000 张图像。

---

## 3. 网络结构设计

- 设计基础卷积神经网络，包含：
  - 多个 3×3 卷积层（`Conv2D`）
  - 批归一化层（`BatchNorm2D`）
  - 激活函数（`ReLU`）
  - 最大池化层（`MaxPool2D`）
  - 全连接层实现分类输出
- 网络结构专为 32×32 彩色图像设计，保持特征图尺寸合理。

---

## 4. 高级组件设计

- **Batch Normalization**  
  减少内部协变量偏移，稳定训练，提升收敛速度和性能。
  
- **Dropout**  
  随机丢弃神经元，防止过拟合，增强泛化能力。
  
- **ResidualBlock**  
  残差连接，缓解梯度消失，提高深层网络训练效果。
  
- **SEBlock**  
  通道注意力机制，提升模型对关键特征的捕捉能力。

---

## 5. 优化策略实验

- 比较不同卷积通道数和全连接层大小对模型性能的影响。
- 评估不同损失函数（`CrossEntropyLoss`、标签平滑）及正则化（L2 正则）对训练效果的影响。
- 激活函数比较（ReLU、LeakyReLU、GELU）分析。
- 实验显示 Batch Normalization 和 Dropout 组合显著提升了模型的准确率和训练稳定性。

---

## 6. 结果与分析

- Batch Normalization 显著加快收敛速度，提升准确率约 5-6%。
- Dropout 在防止过拟合方面有效，进一步提高泛化能力。
- 增加模型容量（卷积通道、全连接单元）提升表达能力，但计算开销增加。
- 标签平滑和 L2 正则化对提升测试准确率有一定辅助作用。
- 激活函数不同对训练动态和最终准确率影响明显。

---

## 7. 如何运行

1. 安装依赖：
    ```bash
    pip install -r requirements.txt
    ```
2. 下载 CIFAR-10 数据集（脚本会自动下载）：
    ```bash
    python codes/CIFAR10_Training/train.py
    ```
3. 训练模型，支持不同超参数配置，具体参数可修改 `train.py` 文件。
4. 测试与评估：
    ```bash
    python codes/CIFAR10_Training/test.py
    ```
5. 训练过程及结果日志保存在 `logs/` 目录，模型权重保存在 `models/` 目录。

---

## 8. 参考文献

1. Krizhevsky, A., Hinton, G. (2009). Learning multiple layers of features from tiny images. Technical report, University of Toronto.  
2. Ioffe, S., Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. *arXiv preprint arXiv:1502.03167*.  
3. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., Salakhutdinov, R. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *Journal of Machine Learning Research*, 15(56), 1929-1958.  
4. He, K., Zhang, X., Ren, S., Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR*.  
5. Hu, J., Shen, L., Sun, G. (2018). Squeeze-and-Excitation Networks. *CVPR*.
