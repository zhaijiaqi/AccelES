"""
    This file is used to search the best ReSparse ratio and test the accuracy after Quantification & Multi-core & ReSparse.

    Usage:
        python search_best_resparse_ratio.py -B 6 --split 32 --input_path /path/to/mtx_txt_file
"""
import numpy as np
import argparse
import concurrent.futures
import time
import os
import sys
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.append("..")
import quantify_util as qu


if __name__ == "__main__":


    # 预定义参数
    K = 100
    APPROXIMATE_K = 512
    NUM_TEST = 100
    SPLIT_NUM = 32
    
    parser = argparse.ArgumentParser(description="Test the accuracy after quantification")
    parser.add_argument("-B", "--bits", type=int, help="The number of bits used for quantization", default=6) 
    parser.add_argument("--split", type=int, help="Split matrix to 32 spieces or not", default=SPLIT_NUM)
    parser.add_argument("--matrix_file", type=str, help="The input txt file for generating the sparse matrix")
    parser.add_argument("--accuracy_threshold", type=float, help="The accuracy threshold for convergence", default=0.999)
    args = parser.parse_args()
    
    B = args.bits
    distribution = "normal"
    symmetric = False if distribution=="gamma" else True    # 根据数据分布格式选择对称量化和非对称量化
    split = args.split
    input_path = args.matrix_file
    accuracy_threshold = args.accuracy_threshold
    input_file_name = os.path.basename(input_path)
    input_dir = os.path.dirname(input_path)
    output_dir = os.path.join(input_dir, f"{SPLIT_NUM}-cores")
    result_log_file_name = os.path.join(output_dir, "results", f"{input_file_name}_search_resparse_{accuracy_threshold}_CK{APPROXIMATE_K}_split{split}.txt")
    if not os.path.exists(os.path.dirname(result_log_file_name)):
        os.makedirs(os.path.dirname(result_log_file_name))
    
    # 解析glove6B文件并生成对应的spm
    spm = qu.parse_txt_to_spm(input_path)
    rows = spm.shape[0]
    cols = spm.shape[1]


    original_stdout = sys.stdout
    # 搜索最佳ReSparse比例
    with open(result_log_file_name, "w") as f:
        # sys.stdout = f
        epoch = 0
        convergence = [0]*100
        attempted_percent = set()
        high = 1.0
        low = 0.0
        while epoch < 100:
            start_time = time.time()
            if epoch == 0:                  # 第一次测试时不 ReSparse
                resparse_top_percent = 1.0
            elif epoch == 1:                # 第二次测试时 ReSparse 设为50%
                resparse_top_percent = (low+high)/2
            # ReSparse
            print("Epoch: ", epoch) 
            qu.print_num_nnz(spm, "Before ReSparse")
            spm_rsp = qu.resparse_spm(spm, resparse_top_percent)
            qu.print_num_nnz(spm_rsp, "After ReSparse")
            # Quantify
            spm_qtf = qu.quantify_matrix(spm_rsp, B, symmetric)
            # 检查是否有全0行
            qu.check_empty_rows(spm_qtf)
            qu.print_num_nnz(spm_qtf, "After quantification")
            attempted_percent.add(resparse_top_percent)

            # 使用线程池并行执行测试循环中的每次迭代
            summery_precision = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                # 提交每次迭代的任务给线程池
                future_to_precision = {executor.submit(qu.run_single_precision_test, cols, distribution, spm, K, spm_qtf, B, symmetric, split, APPROXIMATE_K): i for i in range(NUM_TEST)}
                # 获取结果
                summery_precision = [future.result() for future in concurrent.futures.as_completed(future_to_precision)]

            if epoch == 0:  # 计算不ReSparse的指标
                convergence[epoch] = qu.print_convergence_matrics(summery_precision, APPROXIMATE_K, K, split)
                convergence_threshold = convergence[0] * accuracy_threshold
            else:
                convergence[epoch] = qu.print_convergence_matrics(summery_precision, APPROXIMATE_K, K, split)
                if convergence[epoch] >= convergence_threshold:
                    high = resparse_top_percent
                else:
                    low = resparse_top_percent
                resparse_top_percent = round((low+high)/2, 3)
                if (resparse_top_percent in attempted_percent):   # 收敛  and (convergence[epoch] >= convergence_threshold)
                    print(f"\nConvergence reached {(high)*100}%, accuracy_threshold={accuracy_threshold}, stop testing\n")
                    # 将Shuffled矩阵保存为mtx文件   (因为ReSparse & Quantify后矩阵的顺序可能发生变化，两个矩阵要保持一致)
                    coo_spm = qu.matrix_to_coo(spm)
                    output_spm_file_name = os.path.join(output_dir, f"{input_file_name}_shuffled_{high}.mtx")
                    qu.write_coo_to_mtx(coo_spm, output_spm_file_name)
                    # 将ReSparse & Quantified矩阵保存为mtx文件
                    coo_spm_qtf = qu.matrix_to_coo(spm_qtf)
                    output_spmqtf_file_name = os.path.join(output_dir, f"{input_file_name}_quantified_B6_ReSparse{high}.mtx")
                    qu.write_coo_to_mtx(coo_spm_qtf, output_spmqtf_file_name)
                    break
                # elif (resparse_top_percent in attempted_percent) and (convergence[epoch] < convergence_threshold):
                #     high = max(1.0, resparse_top_percent+0.1)
            print(f"Convergence: {convergence}")
            print()
            epoch = epoch + 1
            end_time = time.time()
            print("Elapsed time:", end_time - start_time, "seconds")

        sys.stdout = original_stdout
    
    print(f"Shuffled matrix saved to {output_spm_file_name}.")
    print(f"ReSparse & Quantify done, saved to {output_spmqtf_file_name}.")