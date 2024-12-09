import numpy as np
import argparse
import concurrent.futures
import time
import os
import sys
sys.path.insert(1, '/data/sub1/jqzhai/program/approximate-spmv-topk/src/resources/python/quantify')
import quantify_util as qu

"""
    直接生成sparse matrix、vector进行spmv运算
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the accuracy after quantification")
    parser.add_argument("-B", "--bits", type=int, help="The number of bits used for quantization", default=4) 
    parser.add_argument("-r", "--rows", type=int, help="Number of rows in the matrix", default=5000000)   
    parser.add_argument("-c", "--cols", type=int, help="Number of columns in the matrix", default=512)  
    parser.add_argument("--density", type=float, help="Average number of non-zero entries per row", default=0.04) 
    parser.add_argument("--split", type=int, help="Split matrix to 32 spieces or not", default=32)
    args = parser.parse_args()
    B = args.bits
    rows = args.rows
    cols = args.cols
    density = args.density
    split = args.split

    K = 100
    APPROXIMATE_K = 256
    NUM_TEST = 100
    DEFAULT_OUTPUT_DIR = "/data/sub1/jqzhai/program/approximate-spmv-topk/data/matrices_for_testing/self_generated_matrix/nnz_per_row_distribution_result"
    
    rst_output_file_name = f"rows{rows}_density{density}_gamma-nnz-per-row_B{B}_split{split}"
    rst_output_file = os.path.join(DEFAULT_OUTPUT_DIR,rst_output_file_name)
    original_stdout = sys.stdout
    with open(rst_output_file,"w") as f:
        sys.stdout = f
        print(f"ROWS={rows}\tCOLS={cols}\tdensity={density*100}%\tsplit={split}\tAPPROXIMATE_K={APPROXIMATE_K}\tB={B}\n")

        # 生成稀疏矩阵spm，
        spm = qu.create_spm_gamma_nnz(rows,cols,density)
        spm_qtf = qu.quantify_matrix(spm, B, True)

        # 使用线程池并行执行测试循环中的每次迭代
        summery_precision = []
        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # 提交每次迭代的任务给线程池
            future_to_precision = {executor.submit(qu.run_single_precision_test, cols, "normal", spm, K, spm_qtf, B, True, split, APPROXIMATE_K): i for i in range(NUM_TEST)}
            # 获取结果
            summery_precision = [future.result() for future in concurrent.futures.as_completed(future_to_precision)]
        end = time.time()
        
        print(f"exec_time:\t{(end-start)}s")
        qu.print_avg_precision(summery_precision, APPROXIMATE_K)
        sys.stdout = original_stdout