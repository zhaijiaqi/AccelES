import numpy as np
import argparse
import concurrent.futures
import time
import os
import sys
sys.path.insert(1, '/data/sub1/jqzhai/program/Topk-SpMV/src/resources/python/quantify')
import quantify_util as qu

"""
    直接生成sparse matrix、vector进行spmv运算
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the accuracy after quantification")
    parser.add_argument("-B", "--bits", type=int, help="The number of bits used for quantization", default=6) 
    parser.add_argument("-r", "--rows", type=int, help="Number of rows in the matrix", default=40000)   
    parser.add_argument("-c", "--cols", type=int, help="Number of columns in the matrix", default=512)  
    parser.add_argument("--density", type=float, help="Average number of non-zero entries per row", default=0.0)   
    parser.add_argument("--distribution", type=str, help="Distribution of the value in sparse matrix", default="normal")
    parser.add_argument("--split", type=int, help="Split matrix to 32 spieces or not", default=0)
    parser.add_argument("--resparse", type=float, help="ReSparse top percent", default=1)
    args = parser.parse_args()
    B = args.bits
    rows = args.rows
    cols = args.cols
    density = args.density
    distribution = args.distribution
    split = args.split
    resparse_top_percent = args.resparse

    K = 100
    APPROXIMATE_K = 384
    NUM_TEST = 100
    DEFAULT_OUTPUT_DIR = "/data/sub1/jqzhai/program/Topk-SpMV/data/matrices_for_testing/self_generated_matrix/result"
    symmetric = False if distribution=="gamma" else True    # 根据数据分布格式选择对称量化和非对称量化
    
    rst_output_file_name = f"ReSparse{resparse_top_percent}_bits{B}_rows{rows}_cols{cols}_density{density*100}_distribution{distribution}_split{split}_APPROXIMATE_K_{APPROXIMATE_K}"
    rst_output_file = os.path.join(DEFAULT_OUTPUT_DIR,rst_output_file_name)
    original_stdout = sys.stdout
    with open(rst_output_file,"w") as f:
        sys.stdout = f
        print(f"ROWS={rows}\tCOLS={cols}\tdensity={density*100}%\tdistribution={distribution}\tsplit={split}\tAPPROXIMATE_K={APPROXIMATE_K}\tB={B}\n")

        spm = qu.create_spm(rows,cols,density,distribution)
        # ReSparse
        qu.print_num_nnz(spm, "Before ReSparse")
        spm_rsp = qu.resparse_spm(spm, resparse_top_percent)
        qu.print_num_nnz(spm_rsp, "After ReSparse")
        # Quantify
        spm_qtf = qu.quantify_matrix(spm_rsp, B, symmetric)
        # 检查是否有全0行
        qu.check_empty_rows(spm_qtf)
        qu.print_num_nnz(spm_qtf, "After quantification")

        # 使用线程池并行执行测试循环中的每次迭代
        summery_precision = []
        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # 提交每次迭代的任务给线程池
            future_to_precision = {executor.submit(qu.run_single_precision_test, cols, distribution, spm, K, spm_qtf, B, symmetric, split, APPROXIMATE_K): i for i in range(NUM_TEST)}
            # 获取结果
            summery_precision = [future.result() for future in concurrent.futures.as_completed(future_to_precision)]
        end = time.time()
        
        print(f"exec_time:\t{(end-start)}s")
        qu.print_avg_precision(summery_precision, APPROXIMATE_K)
        sys.stdout = original_stdout