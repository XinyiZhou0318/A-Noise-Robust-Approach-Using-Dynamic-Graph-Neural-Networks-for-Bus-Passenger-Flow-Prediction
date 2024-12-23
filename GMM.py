import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture

# 设置文件路径
input_file_path = "C:\\Users\\86176\\Desktop\\GNNs &Transportation\\processed_counts\\2022_02_processed_new.csv"
output_file_path = "C:\\Users\\86176\\Desktop\\GNNs &Transportation\\processed_counts\\gmm_data.csv"

# 读取数据
data = pd.read_csv(input_file_path)

# 初始化一个存储 GMM 滤波结果的 DataFrame
gmm_filtered_data = data.copy()

# GMM 滤波函数
def gmm_filter(data, n_components=2, threshold=0.05):
    """
    使用高斯混合模型 (GMM) 进行噪声检测与修正。
    :param data: 输入的一维时间序列数据
    :param n_components: GMM 中高斯分布的数量
    :param threshold: 概率密度的阈值，低于此值判定为噪声
    :return: 滤波后的时间序列
    """
    try:
        # 拟合 GMM 模型
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        data = data.reshape(-1, 1)  # 转换为二维数组
        gmm.fit(data)

        # 计算每个点的概率密度
        log_probs = gmm.score_samples(data)
        probs = np.exp(log_probs)  # 转为概率值

        # 识别噪声点
        is_noise = probs < threshold

        # 替换噪声点为对应高斯分布的均值
        corrected_data = data.flatten()
        for i, noise in enumerate(is_noise):
            if noise:
                # 找到属于的高斯分布，并替换为均值
                component = gmm.predict(data[i].reshape(1, -1))
                corrected_data[i] = gmm.means_[component][0]

        return corrected_data
    except Exception as e:
        print(f"GMM 滤波错误: {e}")
        return data

# 对每个站点的时间序列应用 GMM 滤波
for index, row in data.iterrows():
    time_series = row[1:].values.astype(float)  # 提取时间序列
    filtered_series = gmm_filter(time_series)  # 应用 GMM 滤波
    gmm_filtered_data.iloc[index, 1:] = filtered_series  # 更新滤波后的数据

# 保存滤波结果到指定路径
gmm_filtered_data.to_csv(output_file_path, index=False)

print(f"GMM 滤波后的数据已保存到: {output_file_path}")
