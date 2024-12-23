import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time

# 记录开始时间
start_time = time.time()

# 读取数据
feature1 = pd.read_csv(r"C:\Users\86176\Desktop\GNNs &Transportation\processed_counts\feature1_gmm_fe.csv").drop(columns=['stop'])
feature2 = pd.read_csv(r"C:\Users\86176\Desktop\GNNs &Transportation\processed_counts\feature2_gmm_fe.csv").drop(columns=['stop'])
feature3 = pd.read_csv(r"C:\Users\86176\Desktop\GNNs &Transportation\processed_counts\feature3_gmm_fe.csv").drop(columns=['stop'])
true_values = pd.read_csv(r"C:\Users\86176\Desktop\GNNs &Transportation\processed_counts\2022_02_25_17_00_true_values.csv")['2022-02-25_True_Value']

# 将数据转换为 PyTorch 张量
X1 = torch.tensor(feature1.values, dtype=torch.float32)
X2 = torch.tensor(feature2.values, dtype=torch.float32)
X3 = torch.tensor(feature3.values, dtype=torch.float32)
y_true = torch.tensor(true_values.values, dtype=torch.float32).reshape(-1, 1)

# 定义液体神经网络模型
class LiquidNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LiquidNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        return x

# 动态调整模型输入维度
hidden_dim = 256
output_dim = 1

# 定义损失函数
criterion = nn.MSELoss()

# 修改训练函数，记录每 100 代的损失值
def train_model(input_tensor, y, hidden_dim, output_dim, epochs=10000, lr=0.001):
    input_dim = input_tensor.shape[1]
    model = LiquidNN(input_dim, hidden_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_values = []  # 用于存储每 100 代的损失值
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(input_tensor)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0 or epoch == epochs - 1:
            loss_item = loss.item()
            loss_values.append(loss_item)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss_item}")
    return model, loss_values

# 训练模型并记录损失值
print("Training model1...")
model1, loss_values1 = train_model(X1, y_true, hidden_dim, output_dim)

print("Training model2...")
model2, loss_values2 = train_model(X2, y_true, hidden_dim, output_dim)

print("Training model3...")
model3, loss_values3 = train_model(X3, y_true, hidden_dim, output_dim)

# 获取每个模型的预测结果
relu = nn.ReLU()

model1.eval()
model2.eval()
model3.eval()

pred1 = relu(model1(X1)).detach()
pred2 = relu(model2(X2)).detach()
pred3 = relu(model3(X3)).detach()

# 定义权重优化网络
class WeightNet(nn.Module):
    def __init__(self):
        super(WeightNet, self).__init__()
        # 初始化四个可训练权重
        self.w1 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.w2 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.w3 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.w0 = nn.Parameter(torch.randn(1, requires_grad=True))  # 新增全局权重 w0

    def forward(self, pred1, pred2, pred3):
        # 将权重正值化并归一化
        total_weight = torch.abs(self.w1) + torch.abs(self.w2) + torch.abs(self.w3)
        w1 = torch.abs(self.w1) / total_weight
        w2 = torch.abs(self.w2) / total_weight
        w3 = torch.abs(self.w3) / total_weight

        # 加权预测
        weighted_pred = w1 * pred1 + w2 * pred2 + w3 * pred3
        # 最终输出乘以全局权重 w0
        final_pred = self.w0 * weighted_pred
        return final_pred, (w1, w2, w3, self.w0)

# 初始化权重优化网络
weight_net = WeightNet()
optimizer_weights = optim.Adam(weight_net.parameters(), lr=0.01)

# 训练权重网络
epochs = 5000
for epoch in range(epochs):
    weight_net.train()
    optimizer_weights.zero_grad()
    # 计算加权预测
    final_pred, weights = weight_net(pred1, pred2, pred3)
    loss = criterion(final_pred, y_true)
    loss.backward()
    optimizer_weights.step()
    if (epoch + 1) % 100 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# 获取训练后的权重
trained_w1, trained_w2, trained_w3, trained_w0 = weights
print(f"Trained weights: w1={trained_w1.item():.4f}, w2={trained_w2.item():.4f}, w3={trained_w3.item():.4f}, w0={trained_w0.item():.4f}")

# 使用训练好的权重计算最终预测
final_predictions_weighted = (trained_w0 * (trained_w1 * pred1 + trained_w2 * pred2 + trained_w3 * pred3)).detach().numpy()

# 保存预测值
prediction_df = pd.DataFrame({
    "True Values": y_true.numpy().flatten(),
    "Predictions Model1": pred1.numpy().flatten(),
    "Predictions Model2": pred2.numpy().flatten(),
    "Predictions Model3": pred3.numpy().flatten(),
    "Final Predictions (Weighted)": final_predictions_weighted.flatten()
})
prediction_df.to_csv(r"C:\Users\86176\Desktop\GNNs &Transportation\processed_counts\predictions_weighted.csv", index=False)

print("Weighted predictions saved to 'predictions_weighted.csv'.")

# 评价指标计算
mae = mean_absolute_error(y_true.numpy(), final_predictions_weighted)
rmse = np.sqrt(mean_squared_error(y_true.numpy(), final_predictions_weighted))

def calculate_accuracy(true_values, predictions, tolerance):
    errors = np.abs(true_values - predictions)
    accuracy = np.mean(errors <= (tolerance * true_values) + 0.25)
    return accuracy

accuracy_5 = calculate_accuracy(y_true.numpy(), final_predictions_weighted, 0.05)
accuracy_10 = calculate_accuracy(y_true.numpy(), final_predictions_weighted, 0.10)

print(f"Weighted MAE: {mae}")
print(f"Weighted RMSE: {rmse}")
print(f"Weighted Accuracy within 5% tolerance: {accuracy_5 * 100:.2f}%")
print(f"Weighted Accuracy within 10% tolerance: {accuracy_10 * 100:.2f}%")

# 记录结束时间并计算运行时间
end_time = time.time()
runtime = end_time - start_time
print(f"Total runtime: {runtime:.2f} seconds")
