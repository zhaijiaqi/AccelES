import numpy as np
import argparse
import concurrent.futures
import time
import os
import sys
sys.path.insert(1, '../')
import quantify_util as qu


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Test the accuracy after quantification")
    parser.add_argument("-B", "--bits", type=int, help="The number of bits used for quantization", default=6)
    parser.add_argument("--split", type=int, help="Split matrix to 32 spieces or not", default=24)
    parser.add_argument("--distribution", type=str, help="The distribution of the matrix, normal or gamma", default="normal")
    parser.add_argument("--spm_path", type=str, help="The input txt file for the sparse matrix")
    parser.add_argument("--q_spm_path", type=str, help="The input txt file for the quantified sparse matrix")
    args = parser.parse_args()
    
    B = args.bits
    split = args.split
    distribution = args.distribution
    spm_path = args.spm_path
    q_spm_path = args.q_spm_path
    
    # 解析glove6B文件并生成对应的spm
    spm = qu.read_coo_matrix(spm_path)
    rows = spm.shape[0]
    cols = spm.shape[1]
    q_spm = qu.read_coo_matrix(q_spm_path)
    # 预定义参数
    K = 100
    APPROXIMATE_K = 384
    NUM_TEST = 100
    rst_output_file_name = os.path.basename(spm_path+"_NDCG.txt")
    
    symmetric = False if distribution=="gamma" else True    # 根据数据分布格式选择对称量化和非对称量化
    rst_output_file = os.path.join(os.path.dirname(spm_path), "results", rst_output_file_name)
    if not os.path.exists(os.path.dirname(rst_output_file)):
        os.makedirs(os.path.dirname(rst_output_file))
    original_stdout = sys.stdout


    with open(rst_output_file,"w") as f:
        sys.stdout = f
        print("NDCG of", os.path.basename(spm_path),"\n")
        print(f"rows:{rows}\tcols:{cols}\tK:{K}\tdistribution:{distribution}\tB:{B}\tsplit:{split}\tsymmetric:{symmetric}\n")
        qu.print_num_nnz(spm, "Before quantification")
        qu.print_num_nnz(q_spm, "After quantification")

        # 使用线程池并行执行测试循环中的每次迭代
        summery_precision = []
        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            # 提交每次迭代的任务给线程池
            ndcg_precision = {executor.submit(qu.run_single_ndcg_test, cols, distribution, spm, K, q_spm, B, symmetric, split, APPROXIMATE_K): i for i in range(NUM_TEST)}
            # 获取结果
            summery_precisions = [future.result() for future in concurrent.futures.as_completed(ndcg_precision)]
        end = time.time()
        qu.print_avg_precision(summery_precisions, APPROXIMATE_K)
        print(f"exec_time:\t{(end-start)}s")
        sys.stdout = original_stdout

print(f"file haved been writeen to: {rst_output_file}")