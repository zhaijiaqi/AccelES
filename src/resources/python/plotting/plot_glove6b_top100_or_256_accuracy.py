import seaborn as sns
import matplotlib.pyplot as plt


# 设置颜色代码
color1 = "#038355" # 孔雀绿
color2 = "#ffc34e" # 向日黄

# 输入数据
x = [1,2,3,4,5,6,7]
y1 = [100, 37.5, 62.5, 78.12, 86,90.67, 93]
y2 = [100, 100, 100, 100, 100, 100, 100]

# 设置字体
# font = {'family' : 'Times New Roman'}
# plt.rc('font', **font)
plt.figure(figsize=(10, 8))

# 绘图
sns.set_style("whitegrid") # 设置背景样式
plt.rcParams.update({
    'font.family': 'serif',  # 可根据需要更改字体家族
    'font.serif': ['Times New Roman'],  # 指定特定的字体，如 Times New Roman
    'axes.labelsize': 12,  # 设置轴标签的字体大小
    'xtick.labelsize': 10,  # 设置 x 轴刻度标签的字体大小
    'ytick.labelsize': 10,  # 设置 y 轴刻度标签的字体大小
})
sns.lineplot(x=x, y=y1, color=color1, linewidth=4.0, marker="o", markersize=16, markeredgecolor="white", markeredgewidth=1.5, label='Approximate with Top-100')
sns.lineplot(x=x, y=y2, color=color2, linewidth=4.0, marker="s", markersize=16, markeredgecolor="white", markeredgewidth=1.5, label='Approximate with Top-256')

# 添加标题和标签
plt.title("Approximate Top-100 of Glove", fontweight='bold', fontsize=32, pad=30)
plt.xlabel("Top-K from 1 to 100", fontsize=24, loc='center', labelpad=10)
plt.ylabel("Precision", fontsize=24, loc='center')
 
# 添加图例
plt.legend(loc='lower right', frameon=True, fontsize=18)
 
# 设置刻度字体和范围
# 设置横坐标刻度及标签
fixed_ticks = [1,2,3,4,5,6,7]                       # 横坐标刻度
fixed_labels = ['1','8','16','32','50','75','100']  # 横坐标标签
plt.xticks(fixed_ticks, fixed_labels, fontsize=24)   # 定义横坐标刻度及label
plt.yticks(fontsize=24)
plt.xlim(1, 7)
plt.ylim(0, 100.5)
 
# 设置坐标轴样式
for spine in plt.gca().spines.values():
    spine.set_edgecolor("#CCCCCC")
    spine.set_linewidth(1.5)
 
plt.savefig('Approximate_with_Top-100_or_256.png', dpi=300, bbox_inches='tight')