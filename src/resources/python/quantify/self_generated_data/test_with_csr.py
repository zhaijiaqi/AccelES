from scipy.sparse import rand
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, save_npz, load_npz
from random import sample

from sparse_dot_topn import sp_matmul_topn
from csr_util import quantify_csr, quantify_mtx, read_mtx, top_K_indices, split_csr_matrix, create_random_vec
import argparse
import os
import sys

################################################################
# Script used for quantize accurancy testing                   #
################################################################

DEFAULT_OUTPUT_DIR = "/data/sub1/jqzhai/program/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/results"
FULL_PRECISION_MATRIX_INPUT_DIR = "/data/sub1/jqzhai/program/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-"
CK = 512
SPLIT_NUM = 32
# CK = (int(K/8)+1)*8*4       # 32个小核要统计的总c*k
TEST_NUM = 100              # 测试次数
GAMMA_K = 3.0
K=[1, 8, 16, 32, 50, 75, 100]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the accuracy after quantification")
    parser.add_argument("-B", "--bits", type=int, help="The number of bits used for quantization", default=6) 
    parser.add_argument("-r", "--rows", type=int, help="Number of rows in the matrix", default=5000000)   
    parser.add_argument("-c", "--cols", type=str, help="Number of columns in the matrix", default=1024)  
    parser.add_argument("--degrees", type=int, help="Average number of non-zero entries per row", default=20)   
    parser.add_argument("--distribution", type=str, help="Distribution that determines the number of non-zero entries per row", default="gamma")
    parser.add_argument("--split", type=int, help="Split matrix to 32 spieces or not", default=32)
    args = parser.parse_args()
    B = args.bits
    rows = args.rows
    cols = args.cols
    degrees = args.degrees
    distribution = args.distribution
    split = args.split
    

    matrice_file_name = f"matrix_{rows}_{cols}_{degrees}_{distribution}.mtx"
    matrice_file = os.path.join(FULL_PRECISION_MATRIX_INPUT_DIR, matrice_file_name)
    rst_output_file_name = f"ReSparse_rows{rows}_cols{cols}_degrees{degrees}_distibution_{distribution}_B{B}_split{split}"
    rst_output_file = os.path.join(DEFAULT_OUTPUT_DIR,rst_output_file_name)

    # write result to file  
    original_stdout = sys.stdout
    with open(rst_output_file,"w") as f:
        sys.stdout = f
        # Read .mtx  &  turn into csr  &  quantize
        full_coo = read_mtx(matrice_file)
        full_csr = csr_matrix(full_coo)
        quantified_csr = quantify_mtx(matrice_file, B, distribution)
        if split > 0:
            split_quan_matrics = split_csr_matrix(quantified_csr, split) # split quantified matricss 
        # quanized_csr = load_npz(QUANTIZATION_INPUT_PATH)
        rows = quantified_csr.shape[0]
        cols = quantified_csr.shape[1]
        nnz = quantified_csr.nnz
        density = round(nnz/(rows*cols)*100,2)
        
        # test TEST_NUM random vectors & Count the samples with accuracy less than 100%
        print(f"ROWS={rows}\tCOLS={cols}\tdegrees={degrees}\tdensity={density}%\tdistribution={distribution}\tsplit={split}\nTEST_NUM={TEST_NUM}\tK={K}\tB={B}\n")
        # matrics
        quantified_precision_256=[0]*len(K)
        quantified_precision_512=[0]*len(K)
        total_precision_256=[0]*len(K)
        total_precision_512=[0]*len(K)
        error_num_256=[0]*len(K)
        error_num_512=[0]*len(K)
        for i in range(TEST_NUM):
            vec = create_random_vec(cols)
            quantified_vec = quantify_csr(vec, B) # 测试发现vec采用的量化方式与matrix相同时，精度最高
            # Test the calculation of full precision and quantized precision respectively
            # full-precision results
            full_res = sp_matmul_topn(full_csr, vec.transpose(), 1, float('-inf'), n_threads=32)
            gold_ref = top_K_indices(full_res.data, 512)    # ref 直接求 Top-512
            # quanized_precison results
            quantified_top_256 = []
            quantified_top_512 = []
            if split==0:
                quantified_res = sp_matmul_topn(quantified_csr, quantified_vec.transpose(), 1, float('-inf'), n_threads=32)
                quantified_top_256 = top_K_indices(quantified_res.data, CK)
                quantified_top_512 = top_K_indices(quantified_res.data, CK*2)
            # split to 32 spieces
            else:
                for i, split_matrix in enumerate(split_quan_matrics):
                    split_res = sp_matmul_topn(split_matrix,quantified_vec.transpose(), 1, -1e9, n_threads=32)
                    split_top_256 = top_K_indices(split_res.data, int(CK/32))+i*(rows//split)
                    split_top_512 = top_K_indices(split_res.data, int(CK*2/32))+i*(rows//split)
                    quantified_top_256.extend(split_top_256)
                    quantified_top_512.extend(split_top_512)
            # compare results
            for j,k in enumerate(K):
                quantified_precision_256[j] = len(np.intersect1d(quantified_top_256, gold_ref[-k:]))/k
                quantified_precision_512[j] = len(np.intersect1d(quantified_top_512, gold_ref[-k:]))/k
                if quantified_precision_256[j]<1.0:    # count error num
                    # print(f'{i+1}:Precision:{quantified_precision[i]*100}%')
                    error_num_256[j]+=1
                if quantified_precision_512[j]<1.0:
                    error_num_512[j]+=1
                total_precision_256[j] += quantified_precision_256[j]
                total_precision_512[j] += quantified_precision_512[j]
        print(f"CK={CK}")
        for i,k in enumerate(K):
            print(f"K={k}\tavg_precison:{total_precision_256[i]/TEST_NUM*100}%\trecall:{(1-error_num_256[i]/TEST_NUM)*100}%")
        print(f"\nCK={CK*2}")
        for i,k in enumerate(K):
            print(f"K={k}\tavg_precison:{total_precision_512[i]/TEST_NUM*100}%\trecall:{(1-error_num_512[i]/TEST_NUM)*100}%")
    sys.stdout = original_stdout