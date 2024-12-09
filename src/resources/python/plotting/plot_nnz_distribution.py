import sys
sys.path.insert(1, '/data/sub1/jqzhai/program/approximate-spmv-topk/src/resources/python/quantify')
import quantify_util as qu
import numpy as np
import matplotlib.pyplot as plt


def parse_sparse_glove6B_to_spm(path):
    with open(path,'r') as f:
        spm = []
        # 逐行读取文件
        for line in f:
            # 分割字符串，以空格为分隔符
            parts = line.split()
            # 获取浮点数部分并转换为浮点数，忽略第一个字符串
            floats = [float(x) for x in parts[1:]]
            # 将浮点数添加到列表中
            spm.append(floats)
    spm = np.array(spm)
    return spm

if __name__ == "__main__":
    # 自生成spm
    # rows = 5000000
    # cols = 512
    # density = 0.04
    # spm = qu.create_spm_gamma_nnz(rows,cols,density

    # 真实数据集
    spm = parse_sparse_glove6B_to_spm("/data/sub1/jqzhai/program/approximate-spmv-topk/data/matrices_for_testing/cc_zh/sparse.cc.zh.300.vec")
    non_zero_counts = np.count_nonzero(spm, axis=1)
    # non_zero_counts[non_zero_counts>50] = 0

    # 绘制直方图
    plt.hist(non_zero_counts, bins=100, color='blue', alpha=0.7)
    plt.title('nnz per row distribution in cc_zh')
    plt.xlabel('nnz per row')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    plt.savefig(f'./sparse.cc_zh_nnz_distribution_{spm.shape[0]}_{spm.shape[1]}.jpeg')

