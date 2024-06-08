from torch import nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from model import DeepModel
from dataload import create_datasets, create_test_dataset
from utils import train, evaluate, visualize
import matplotlib.pyplot as plt

# 设置设备
device = "cuda"

# 配置文件
config = {
    "data_dir": "./dataset",
    "bc_points_file": "bc_points.npy",
    "bc_labels_file": "bc_label.npy",
    "ic_points_file": "ic_points.npy",
    "ic_labels_file": "ic_label.npy"
}

# 创建数据集和数据加载器
train_dataset = create_datasets(config["data_dir"])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 实例化模型
model = DeepModel().to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
loss_file = "losses.npz"
losses, l2_error_u, l2_error_v, l2_error_p, l2_error_total = train(model, 
        train_loader, criterion, optimizer, epochs=20, device=device, loss_file=loss_file)
# 测试数据
test_inputs, test_labels = create_test_dataset(config["data_dir"])

# 评估模型
evaluate(model, test_inputs, test_labels, device)

# 可视化结果
# 加载 loss 值
data = np.load("losses.npz")
losses = data['loss']
l2_error_u = data['l2_error_u']
l2_error_v = data['l2_error_v']
l2_error_p = data['l2_error_p']
l2_error_total = data['l2_error_total']

# 绘制误差曲线
plt.figure(dpi=300)

epochs = np.arange(len(l2_error_u))

plt.plot(epochs, l2_error_u, 'b-', label="Normalized L2 Error of U")
plt.plot(epochs, l2_error_v, 'g-', label="Normalized L2 Error of V")
plt.plot(epochs, l2_error_p, 'r-', label="Normalized L2 Error of P")
plt.plot(epochs, l2_error_total, 'k-', label="Normalized L2 Error of Total")

plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Normalized L2 Error')
plt.ylim(0, 1)
plt.title('Normalized L2 Errors over Epochs')
plt.savefig("Normalized_L2_Errors.png")

visualize(model, test_inputs, test_labels, losses, path="./videos", device=device, dpi=300)
