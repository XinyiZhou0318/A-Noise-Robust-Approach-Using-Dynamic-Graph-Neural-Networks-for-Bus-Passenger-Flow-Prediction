import pandas as pd
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 文件路径
arrival_data_path = r"C:\Users\86176\Desktop\GNNs &Transportation\processed_counts\2022_02_processed_new.csv"
stops_path = r"C:\Users\86176\Desktop\GNNs &Transportation\test\Dynamic GNN\stops.csv"
stop_times_path = r"C:\Users\86176\Desktop\GNNs &Transportation\test\Dynamic GNN\stop_times.csv"
zero_passenger_stops_path = r"C:\Users\86176\Desktop\GNNs &Transportation\test\Dynamic GNN\zero_passenger_stops_02.csv"

# 输出路径
output_path_1 = r"C:\Users\86176\Desktop\GNNs &Transportation\processed_counts\feature1.csv"
output_path_2 = r"C:\Users\86176\Desktop\GNNs &Transportation\processed_counts\feature2.csv"
output_path_3 = r"C:\Users\86176\Desktop\GNNs &Transportation\processed_counts\feature3.csv"

# 加载数据
arrival_df = pd.read_csv(arrival_data_path, encoding='latin1', on_bad_lines='skip')
stops_df = pd.read_csv(stops_path)
stop_times_df = pd.read_csv(stop_times_path)
zero_passenger_stops_df = pd.read_csv(zero_passenger_stops_path)

# 定义需要的时间段
target_date = "2022-02-25"
target_columns_1 = [col for col in arrival_df.columns if target_date in col and "15:00:00" <= col.split(' ')[1] <= "16:50:00"]
input_dates_2 = ["2022-02-22", "2022-02-23", "2022-02-24"]
input_dates_3 = ["2022-02-04", "2022-02-11", "2022-02-18"]

# 提取三组矩阵
matrix_1 = arrival_df[['stop'] + target_columns_1].set_index('stop').fillna(0)
matrix_2 = arrival_df[['stop'] + [col for col in arrival_df.columns if any(date in col for date in input_dates_2) and "17:00:00" in col]].set_index('stop').fillna(0)
matrix_3 = arrival_df[['stop'] + [col for col in arrival_df.columns if any(date in col for date in input_dates_3) and "17:00:00" in col]].set_index('stop').fillna(0)

# 动态图创建函数，处理取消的站点
def create_dynamic_graph(trip_data, stops_data, zero_passenger_data):
    G = nx.DiGraph()
    stop_ids = stops_data['stop_id'].unique()

    for _, row in stops_data.iterrows():
        G.add_node(row['stop_id'], stop_name=row['stop_name'])

    for trip_id in trip_data['trip_id'].unique():
        trip_stops = trip_data[trip_data['trip_id'] == trip_id].sort_values('stop_sequence')['stop_id'].tolist()
        for i in range(len(trip_stops) - 1):
            if trip_stops[i] in stop_ids and trip_stops[i + 1] in stop_ids:
                G.add_edge(trip_stops[i], trip_stops[i + 1])

    # 根据 zero_passenger_stops_df 中的 stop 列和 ons、offs 判断取消的站点
    canceled_stops = zero_passenger_data[(zero_passenger_data['ons'] == 0) & (zero_passenger_data['offs'] == 0)]['stop'].unique()
    for stop_id in canceled_stops:
        if stop_id in G:
            predecessors = list(G.predecessors(stop_id))
            successors = list(G.successors(stop_id))
            for pred in predecessors:
                for succ in successors:
                    G.add_edge(pred, succ)
            G.remove_node(stop_id)

    return G

# 定义 DGCN 模型
class DGCN(torch.nn.Module):
    def __init__(self, input_dim):
        super(DGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, input_dim)  # 隐藏层维度等于输入列数
        self.W0 = torch.nn.Parameter(torch.randn(input_dim, input_dim))
        self.Wt = torch.nn.Parameter(torch.randn(input_dim, input_dim))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h_neigh = self.conv1(x, edge_index)  # GCN 聚合邻居信息
        h_self = torch.matmul(x, self.W0)  # 自身特征
        h_change = torch.matmul(h_neigh - x, self.Wt)  # 邻居变化特征
        h_combined = h_neigh + h_self + h_change  # 总特征
        return h_combined

# 提取特征函数
def extract_graph_features(matrix, G, model):
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    num_nodes = matrix.shape[0]

    # 确保边的索引在节点范围内
    valid_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
    edge_index = edge_index[:, valid_mask]

    # 转换为 PyTorch 张量
    node_features = torch.tensor(matrix.values, dtype=torch.float)
    data = Data(x=node_features, edge_index=edge_index)

    # 使用模型提取特征
    with torch.no_grad():
        features = model(data)
    return features

# 处理每个矩阵并提取特征
def process_matrix(matrix, stops_df, stop_times_df, zero_passenger_stops_df, output_path):
    input_dim = matrix.shape[1]  # 输入维度等于列数
    model = DGCN(input_dim)

    # 创建动态图
    G = create_dynamic_graph(stop_times_df, stops_df, zero_passenger_stops_df)

    # 提取图特征
    features = extract_graph_features(matrix, G, model)

    # 保存为 CSV 文件
    features_df = pd.DataFrame(features.numpy(), index=matrix.index, columns=matrix.columns)
    features_df.to_csv(output_path)
    print(f"特征已保存至 {output_path}")

# 处理每个矩阵并提取特征
process_matrix(matrix_1, stops_df, stop_times_df, zero_passenger_stops_df, output_path_1)
process_matrix(matrix_2, stops_df, stop_times_df, zero_passenger_stops_df, output_path_2)
process_matrix(matrix_3, stops_df, stop_times_df, zero_passenger_stops_df, output_path_3)

# 加载特征矩阵和真实值
matrix_1 = pd.read_csv(r"C:\Users\86176\Desktop\GNNs &Transportation\processed_counts\feature1.csv", index_col=0)
matrix_2 = pd.read_csv(r"C:\Users\86176\Desktop\GNNs &Transportation\processed_counts\feature2.csv", index_col=0)
matrix_3 = pd.read_csv(r"C:\Users\86176\Desktop\GNNs &Transportation\processed_counts\feature3.csv", index_col=0)
true_values_df = pd.read_csv(r"C:\Users\86176\Desktop\GNNs &Transportation\processed_counts\2022_02_25_17_00_true_values.csv")

# 检查真实值的列名
print(true_values_df.columns)

# 更新真实值列名
true_values = torch.tensor(true_values_df['2022-02-25_True_Value'].values, dtype=torch.float32)  # 根据实际列名修改

# 初始化权重
W1 = torch.nn.Parameter(torch.randn(3, 1, requires_grad=True))
W2 = torch.nn.Parameter(torch.randn(1, requires_grad=True))

# 将三个特征矩阵转为tensor
feature_1 = torch.tensor(matrix_1.values, dtype=torch.float32)
feature_2 = torch.tensor(matrix_2.values, dtype=torch.float32)
feature_3 = torch.tensor(matrix_3.values, dtype=torch.float32)

# 整合特征并计算预测值的函数
def predict(h1, h2, h3, W1, W2):
    combined_features = torch.cat([h1.mean(dim=1, keepdim=True), h2.mean(dim=1, keepdim=True), h3.mean(dim=1, keepdim=True)], dim=1)
    hidden = F.relu(torch.matmul(combined_features, W1))
    prediction = W2 * hidden.sum(dim=1)
    return prediction

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam([W1, W2], lr=0.001)

# 训练模型
num_epochs = 600
losses = []
mae_values = []
rmse_values = []
accuracy_5_percent = []
accuracy_10_percent = []

# 计算准确率 (误差在5%以内和10%以内的正确率)
def calculate_accuracy(true_vals, pred_vals, threshold):
    percent_error = np.abs(true_vals - pred_vals)
    correct_predictions = np.sum(percent_error <= threshold * true_vals)
    accuracy = (correct_predictions / len(true_vals)) * 100
    return accuracy, correct_predictions

for epoch in range(num_epochs):
    optimizer.zero_grad()

    # 预测值
    predictions = predict(feature_1, feature_2, feature_3, W1, W2)

    # predictions=abs(predictions)
    # 计算损失
    loss = criterion(predictions, true_values)
    losses.append(loss.item())

    # 计算MAE和RMSE
    mae = mean_absolute_error(true_values.detach().numpy(), predictions.detach().numpy())
    rmse = np.sqrt(mean_squared_error(true_values.detach().numpy(), predictions.detach().numpy()))
    mae_values.append(mae)
    rmse_values.append(rmse)

    # 计算准确率
    accuracy_5, _ = calculate_accuracy(true_values.detach().numpy(), predictions.detach().numpy(), 0.05)
    accuracy_10, _ = calculate_accuracy(true_values.detach().numpy(), predictions.detach().numpy(), 0.10)
    accuracy_5_percent.append(accuracy_5)
    accuracy_10_percent.append(accuracy_10)

    # 反向传播和优化
    loss.backward()
    optimizer.step()

# 绘制损失曲线
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(2, 2, 2)
plt.plot(mae_values)
plt.title('MAE (Mean Absolute Error)')
plt.xlabel('Epoch')
plt.ylabel('MAE')

plt.subplot(2, 2, 3)
plt.plot(rmse_values)
plt.title('RMSE (Root Mean Squared Error)')
plt.xlabel('Epoch')
plt.ylabel('RMSE')

plt.subplot(2, 2, 4)
plt.plot(accuracy_5_percent, label="<=5% Accuracy")
plt.plot(accuracy_10_percent, label="<=10% Accuracy")
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()

# 输出最终预测值
print("最终预测值:\n", predictions)

# 将预测值保存为 CSV 文件
predictions_df = pd.DataFrame(predictions.detach().numpy(), columns=["Predicted_Value"])
predictions_df.to_csv(r"C:\Users\86176\Desktop\GNNs &Transportation\processed_counts\predictions.csv", index=False)
print("预测值已保存到 'predictions.csv'")

# 输出评估指标
print(f"最终 MAE: {mae_values[-1]}")
print(f"最终 RMSE: {rmse_values[-1]}")
print(f"最终 Accuracy (<=5%): {accuracy_5_percent[-1]}%")
print(f"最终 Accuracy (<=10%): {accuracy_10_percent[-1]}%")
