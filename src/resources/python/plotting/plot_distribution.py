import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
import math


def plot_slice(spm):
    # 将spm切分为split_size块
    slice_size = spm.shape[0] // 32
    redundent_size = spm.shape[0] % 32
    slices = [spm[i:i+slice_size] for i in range(0, 32*slice_size, slice_size)]
    slices[32-1] = np.concatenate((slices[32-1], spm[-redundent_size:]), axis=0)

    for i,slice in enumerate(slices):
        a = i//10
        b = i%10
        nonzero_indices = np.nonzero(slice)
        fp32 = slice[nonzero_indices].tolist()
        plt.hist(fp32, bins=1000, histtype='step', color=f'#{a}{b}BCDE', density=True, edgecolor=f'#{a}{b}BCDE', alpha=0.7)
        plt.xlabel('fp32 value')
        plt.ylabel('Number of fp32 numbers')
        plt.title(f'Value distribution in gamma slice{a}{b}')
        # plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig(f'./distribution_{a}{b}.jpeg')

# 绘制概率密度函数图∏
# plt.plot(x, pdf, label=f'shape={shape}, scale={scale}')
def plot_spm(spm):
    nonzero_indices = np.nonzero(spm)
    fp32 = spm[nonzero_indices].tolist()
    plt.hist(fp32, bins=1000, histtype='step', color=f'blue', density=True, edgecolor=f'blue', alpha=0.7)
    plt.xlabel('fp32 value')
    plt.ylabel('Number of fp32 numbers')
    plt.title(f'Value distribution in normal')
    # plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(f'./distribution_{spm.shape[0]}_{spm.shape[1]}.jpeg')
        
if __name__ == "__main__":
    m = np.random.randn(10000,512)*2-1
    mask = np.random.choice([0, 1], size=m.shape, p=[1-0.04, 0.04])
    spm = m*mask
    plot_spm(spm)



