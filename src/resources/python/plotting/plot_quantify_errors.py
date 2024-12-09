#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Apr 11 09:34:36 2024

@author: halo
"""

import seaborn as sns
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import os
import matplotlib.lines as lines
import pandas as pd
import numpy as np
import scipy.stats as st
from matplotlib.patches import Patch, Rectangle
from plot_exec_time import get_exp_label, get_upper_ci_size
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.lines import Line2D
from plot_utils import *
import re
import datetime

DATE = datetime.date.today().strftime("%Y_%m_%d")

SPLIT = "split32"
# SPLIT = "split0"

QUANTIFY_MODE = "respectively"
# QUANTIFY_MODE = "with_uniform"
# QUANTIFY_MODE = "with_arcsin"

APPROXIMATE_K = 256
# APPROXIMATE_K = 512

QUANTIFY_RESULT_FOLDER = "/data/sub1/jqzhai/program/approximate-spmv-topk/data/results/quantify"

ck_256_512_quantified_dir = os.path.join(QUANTIFY_RESULT_FOLDER, f"ck_256_512_quantified_{QUANTIFY_MODE}_{SPLIT}")

def process_content(content):
    """
    处理读入的文件内容,转化为data

    Args:
        content -> str: 文件内容

    Returns:
        data -> dict: 存放top-256和top-512量化后计算 Top-K=[1,8,16,32,50,75,100]精度的词典
    """
    data = {}
    ck = None
    found_ck_256 = False
    found_ck_512 = False
    lines = content.split('\n')
    # 分别读CK=256和CK=512后的7行，存入data
    for i, line in enumerate(lines):
        if line.startswith("CK="):
            ck = int(line.split("=")[1])
            if ck == 256:
                found_ck_256 = True
                continue
            if ck == 512:
                found_ck_512 = True
                continue
        if found_ck_256 and not found_ck_512:
            if len(data.setdefault(256, {})) < 7:
                matches = re.findall(r'(K=\d+)\s+avg_precison:(\d+.\d+)%\s+recall:(\d+.\d+)%', line)
                for k, avg_precision, recall in matches:
                    k = int(k.split('=')[1])
                    avg_precision = round(float(avg_precision), 2)
                    recall = round(float(recall), 2)
                    data[ck][k] = {'avg_precision': avg_precision, 'recall': recall}
            else:
                continue
        if found_ck_512:
            if len(data.setdefault(512, {})) < 7:
                matches = re.findall(r'(K=\d+)\s+avg_precison:(\d+.\d+)%\s+recall:(\d+.\d+)%', line)
                for k, avg_precision, recall in matches:
                    k = int(k.split('=')[1])
                    avg_precision = round(float(avg_precision), 2)
                    recall = round(float(recall), 2)
                    data[ck][k] = {'avg_precision': avg_precision, 'recall': recall}
            else:
                break
    return data

def read_files_in_directory(directory):
    """
    读取目录中的文件，并将所有文件内容处理成词典形式
    
    Args:
        directory -> str: 目录地址
    
    Returns:
        files_data -> dict: 包含目录中所有文件内容的词典
    """
    files_content = {}
    files_data = {}
    
    for filename in sorted(os.listdir(directory)):
        if "cols512" in filename and "degrees40" in filename:
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                with open(filepath, 'r') as file:
                    files_content[filename] = file.read()
                    files_data[filename] = process_content(files_content[filename])
    return files_data


def plot_err(files_and_data):
    """
    绘制量化、分核近似精度误差图
    
    Args:
        files_and_datas -> dict: 包含所有文件内容的词典
    
    Returns:
        None
    """
    fixed_ticks = [1,2,3,4,5,6,7]                       # 横坐标刻度
    fixed_labels = ['1','8','16','32','50','75','100']  # 横坐标标签
    x=np.array([1,2,3,4,5,6,7])                         # 定义横坐标x
    y=np.array([100,100,100,100,99.98,99.91,99.15])     # 定义y
    plt.figure(figsize=(20,10))                         # 定义图片大小
    plt.rcParams["font.family"] = ["Times"]
    plt.annotate("Top-K (from 1 to 100)", fontsize=24, xy=(0.5, 0.04), xycoords="figure fraction", ha="center") 
    plt.annotate("Uniform", fontsize=24, xy=(0.02, 0.7), xycoords="figure fraction", ha="left")
    plt.annotate("Gamma", fontsize=24, xy=(0.02, 0.3), xycoords="figure fraction", ha="left")
    plt.annotate(f"{get_exp_label(0.5e7,'N=',True)}", fontsize=24, xy=(0.2, 0.9), xycoords="figure fraction", ha="left")  
    plt.annotate(f"{get_exp_label(1e7,'N=',True)}", fontsize=24, xy=(0.5, 0.9), xycoords="figure fraction", ha="left")
    plt.annotate(f"{get_exp_label(1.5e7,'N=',True)}", fontsize=24, xy=(0.75, 0.9), xycoords="figure fraction", ha="left")

    suffixs_row = [5000000, 10000000, 15000000]
    suffixs_distribution = ['uniform','gamma']
    suffixs_B = [4,5,6]
    for i, rows in enumerate(suffixs_row):
        for j, distribution in enumerate(suffixs_distribution):
            plt.subplot(2,3,j*3+i+1) 
            plt.xticks(fixed_ticks, fixed_labels)   # 定义横坐标刻度及label
            # plt.title(f'{rows}\t{distribution}')
            for k, B in enumerate(suffixs_B):
                filename = f'rows{rows}_cols512_degrees40_distibution_{distribution}_B{B}_{SPLIT}'
                if filename in files_and_data and len(files_and_data[filename])!=0:
                    APPROXIMATE_K_dir = files_and_data[filename][APPROXIMATE_K]
                    y_APPROXIMATE_K = []
                    for value in APPROXIMATE_K_dir.values():
                        y_APPROXIMATE_K.append(round(value['avg_precision']/100,2))
                    if B==4:
                        plt.plot(x,y_APPROXIMATE_K,':', color='r', linewidth=3, label='B=4')
                    elif B==5:
                        plt.plot(x,y_APPROXIMATE_K,'--o', color='b', lw=3, label="B=5") 
                    elif B==6:
                        plt.plot(x,y_APPROXIMATE_K, '-.d', color='g', lw=3, label='B=6')
                    if distribution=='gamma':
                        plt.ylim(0.95,1.0) ##设置x轴范围，同理y
                    elif distribution=='uniform':
                        plt.ylim(0.0,1.0)
                    plt.legend(loc='lower left', fontsize=10)
                    plt.ylabel('Precision')
                    plt.grid(True)
    
    save_plot("../../../../data/plots", f"quantify_errors_{SPLIT}_K{APPROXIMATE_K}_{QUANTIFY_MODE}" + ".{}")  


if __name__ == "__main__":
    # read filenames and data -> return dict
    files_and_data = read_files_in_directory(ck_256_512_quantified_dir)
    # for filename, data in files_and_data.items():
    #     print(f"Filename: {filename}\data:\n{data}\n")
    plot_err(files_and_data)