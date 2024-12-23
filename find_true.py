import pandas as pd
from datetime import datetime

# 文件路径
file_path = r"C:\Users\86176\Desktop\GNNs &Transportation\processed_counts\2022_01_processed_new.xlsx"

# 读取 Excel 文件
df = pd.read_excel(file_path)

# 检查列名
# print("列名：", df.columns)

# 目标时间
time_column = datetime(2022, 1, 25, 17, 0, 0)  # 使用 datetime 类型

# 检查目标列是否存在
if time_column not in df.columns:
    print(f"列 {time_column} 不存在。请检查可能的列名：")
    matching_columns = [col for col in df.columns if isinstance(col, datetime) and col.date() == time_column.date()]
    print(f"可能的列名：{matching_columns}")
    raise ValueError(f"列 {time_column} 不存在，请检查数据。")

# 提取目标列数据
filtered_df = df[["stop", time_column]]

# 重命名列为 Stop_ID 和 目标时间列格式
filtered_df.columns = ["Stop_ID", f"{time_column.strftime('%Y-%m-%d')}_True_Value"]

# 保存为新的 Excel 文件
output_path = r"C:\Users\86176\Desktop\GNNs &Transportation\processed_counts\2022_01_25_17_00_True_Value.csv"
filtered_df.to_csv(output_path, index=False)

print(f"保存成功，文件路径为：{output_path}")
