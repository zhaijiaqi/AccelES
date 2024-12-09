import numpy as np
import matplotlib.pyplot as plt
import re

def read_arrays_from_exp(exp):
    # 使用正则表达式匹配两个数组
    pattern = r'\[\s*([^\]]+)\]'
    matches = re.findall(pattern, exp)
    # 确保我们找到了两个数组
    if len(matches) != 2:
        raise ValueError("文件中没有找到两个数组")
    # 将匹配到的数组内容转换为整数列表
    ref = list(map(int, matches[0].replace('\n', ' ').split()))
    print(ref)
    appro = list(map(int, matches[1].replace('\n', ' ').split()))
    print(appro)
    return ref, appro

def read_experiment_results(file_path):
    with open(file_path, 'r') as file:
        content = file.read().strip()
    # 分割每次实验的结果
    experiments = content.split('\n\n')
    ref = [None]*100
    appro = [None]*100
    for i,exp in enumerate(experiments):
        ref[i], appro[i] = read_arrays_from_exp(exp)
    return ref, appro

def find_indices(ref, appro):
    # 创建一个字典来存储 appro 中元素的位置
    appro_dict = {value: index for index, value in enumerate(appro)}
    # 创建结果列表，找到对应元素的位置，如果找不到则设为1000
    indices = [appro_dict.get(value, 1000) for value in ref]
    return indices

def generate_colors(num_experiments):
    # 使用颜色映射来生成不同的颜色
    cmap = plt.get_cmap('tab10')  # 'tab10' 颜色映射可以生成 10 种不同的颜色
    colors = [cmap(i) for i in np.linspace(0, 1, num_experiments)]
    return colors

# 读取实验结果
# ref, appro = read_experiment_results('/Users/halo/Documents/code/Topk-SpMV/data/matrices_for_testing/glove6B/sparse_result/approximate_results.txt')
# ref, appro = read_experiment_results('/Users/halo/Documents/code/Topk-SpMV/data/matrices_for_testing/self_generated_matrix/result/approximate_results.txt')
ref, appro = read_experiment_results('/data/sub1/jqzhai/program/Topk-SpMV/data/matrices_for_testing/cc_zh/results/approximate_results.txt')
# ref, appro = read_experiment_results('/data/sub1/jqzhai/program/Topk-SpMV/data/matrices_for_testing/glove6B/results/approximate_results.txt')

num_experiments = len(ref)
colors = generate_colors(num_experiments)

# 找到ref数组在appro数组中的位置
positions_list = []
max_position = 0    # 记录最远的位置
max_position_index = 0    # 记录最远位置对应的实验编号
for i in range(num_experiments):
    positions = find_indices(ref[i], appro[i])
    max_position = max(max_position, max(positions))
    max_position_index = i if max(positions) == max_position else max_position_index
    positions_list.append(positions)

# 绘制散点图
# 设置字体
font = {'family' : 'Times New Roman',
        'size'   : 18}
plt.rc('font', **font) 
plt.figure(figsize=(10, 8))
for i, positions in enumerate(positions_list):
    plt.scatter(positions, np.full(len(positions), num_experiments - i), c=colors[6], s=13, alpha=0.9, label=f'Experiment {i+1}', marker='o')
plt.axvline(x=100, color='red', linestyle='--', linewidth=1.5, alpha=1)
plt.axvline(x=max_position, color='black', linestyle='--', alpha=0.5, label = f"max_position={max_position}")
# 标注最远位置
plt.annotate(f"The farthest location={max_position}", xy=(max_position, 100-max_position_index), xytext=(-180, +40), 
             textcoords='offset points', 
             arrowprops=dict(arrowstyle='->', color='black'),
             fontsize=16, color='black', verticalalignment='bottom')    # 1000w 的 xytext=(-120, +50), glove.6B 的 xytext=(+10, +30)
plt.xlabel('Real Top-K position',labelpad=10, loc='right')
plt.ylabel('Exp. No.', loc='top', rotation=0, labelpad=-10)
ax = plt.gca()
ax.yaxis.set_label_coords(-0.02, 1.02) 
yticks = ax.get_yticks()
yticks = [tick for tick in yticks if tick != 0]
ax.set_yticks(yticks)
plt.xlim(0, 256)
plt.ylim(0, 100)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.title('CC_zh Corpus', fontweight='bold', fontsize=32, pad=30)
# plt.title('Self-generated SpM with 10,000,000 rows', fontweight='bold', fontsize=32, pad=30)
plt.grid(False)
# plt.savefig('approximate_scatter_glove.png')
# plt.savefig('approximate_scatter_10m.png')
plt.savefig('approximate_scatter_cczh.png')

