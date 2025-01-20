import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
# 读取Excel文件的前三列
data = pd.read_excel('domain10.xlsx', usecols=[2, 3, 4])

# 将数据转换为适合热力图的格式
# 假设数据是数值型的
data = data.iloc[:, :3]

# 创建透视表，以便绘制热力图
heatmap_data = data.pivot(index=data.columns[1], columns=data.columns[0], values=data.columns[2])
# 去除 NaN 值
heatmap_data = heatmap_data.dropna()
# 绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, cmap='coolwarm')
plt.title('edge-TV_distance')
plt.xlabel(data.columns[0])
plt.ylabel(data.columns[1])

# 控制横纵坐标的刻度数量
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # 仅显示整数
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))  # 仅显示整数

# 格式化坐标轴刻度
def format_func(value, tick_number):
    return f'{value:.1f}'  # 保留一位小数

plt.gca().xaxis.set_major_formatter(FuncFormatter(format_func))
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_func))

plt.xlim(heatmap_data.columns.min(), heatmap_data.columns.max())
plt.ylim(heatmap_data.index.max(), heatmap_data.index.min())

plt.show()




