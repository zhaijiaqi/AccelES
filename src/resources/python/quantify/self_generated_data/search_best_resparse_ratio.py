from scipy.sparse import coo_matrix, csr_matrix
import csr_util
import argparse
import os
import sys

################################################################
# Script used for search for the best resparse ratio for SpM   #
################################################################

CK = 512
SPLIT_NUM = 32             # 24-cores
TEST_NUM = 100              # 测试次数
K=[1, 8, 16, 32, 50, 75, 100]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the accuracy after quantification")
    parser.add_argument("-B", "--bits", type=int, help="The number of bits used for quantization", default=6) 
    parser.add_argument("--matrix_file", type=str, help="The input full precision matrix file", default=None)
    parser.add_argument("--qtf_matrix_file", type=str, help="The input quantized & resparsed matrix file", default=None)
    parser.add_argument("--distribution", type=str, help="Distribution that determines the number of non-zero entries per row", default="normal")
    parser.add_argument("--split", type=int, help="Split matrix to 32 spieces or not", default=SPLIT_NUM)
    parser.add_argument("--accuracy_threshold", type=float, help="The accuracy threshold for the test", default=0.999)
    args = parser.parse_args()
    B = args.bits
    matrix_file = args.matrix_file
    qtf_matrix_file = args.qtf_matrix_file
    distribution = args.distribution
    split = args.split
    accuracy_threshold = args.accuracy_threshold

    # 解析文件路径
    input_dir = os.path.dirname(matrix_file)
    result_dir = os.path.join(input_dir, "results")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_file_name = os.path.basename(matrix_file)+f"_search_resparse_{accuracy_threshold}_CK{CK}_split{split}.txt"

    # 近似计算并求精度
    original_stdout = sys.stdout
    with open(os.path.join(result_dir, result_file_name), "w") as f:
        sys.stdout = f
        # Read .mtx  &  turn into csr  &  quantize
        full_coo = csr_util.read_mtx(matrix_file)
        full_csr = csr_util.shuffle_rows_csr(csr_matrix(full_coo))

        # Search parameters
        epoch = 0
        convergence = [0]*100
        attempted_percent = set()
        high = 1.0
        low = 0.0

        while epoch < 100:
            print("Epoch: ", epoch) 
            if epoch == 0:                  # 第一次测试时不 ReSparse
                resparse_top_percent = 1.0
            elif epoch == 1:                # 第二次测试时 ReSparse 设为50%
                resparse_top_percent = (low+high)/2
            # ReSparse & Quantify
            if qtf_matrix_file is None:
                csr_util.print_num_nnz_csr(full_csr, "Before Resparse")
                spm_rsp = csr_util.check_empty_rows_csr(csr_util.resparse_csr(full_csr, resparse_top_percent))
                csr_util.print_num_nnz_csr(spm_rsp, "After Resparse")
                qtf_spm_csr = csr_util.quantify_csr(spm_rsp, B)
                csr_util.print_num_nnz_csr(qtf_spm_csr, "After Quantification")
            else:
                qtf_spm_coo = csr_util.read_mtx(qtf_matrix_file)
                qtf_spm_csr = csr_matrix(qtf_spm_coo)
            rows = qtf_spm_csr.shape[0]
            cols = qtf_spm_csr.shape[1]
            attempted_percent.add(resparse_top_percent)

            if epoch == 0:  # 计算不ReSparse的指标
                convergence[epoch] = csr_util.cpt_convergence_matrics(full_csr, qtf_spm_csr, B, CK, TEST_NUM, split)
                convergence_threshold = convergence[0] * accuracy_threshold
            else:
                convergence[epoch] = csr_util.cpt_convergence_matrics(full_csr, qtf_spm_csr, B, CK, TEST_NUM, split)
                if convergence[epoch] >= convergence_threshold:
                    high = resparse_top_percent
                else:
                    low = resparse_top_percent
                resparse_top_percent = round((low+high)/2, 3)
                if (resparse_top_percent in attempted_percent):   # 收敛  去掉了 and (convergence[epoch] >= convergence_threshold) 来缩短搜索过程
                    print(f"\nConvergence reached {(high)*100}%, accuracy_threshold={accuracy_threshold}, stop testing\n")
                    # 保存shuflled 矩阵
                    output_spm_file_name = os.path.join(os.path.dirname(input_dir), f"{matrix_file}_shuffled_{high}.mtx")
                    full_coo_shuffle = coo_matrix(full_csr)
                    csr_util.write_coo_mtx(output_spm_file_name, full_coo_shuffle)
                    # 保存ReSparse & Quantified矩阵
                    coo_spm_qtf = coo_matrix(qtf_spm_csr)
                    output_spmqtf_file_name = os.path.join(os.path.dirname(input_dir), f"{matrix_file}_quantified_B6_ReSparse{high}.mtx")
                    csr_util.write_coo_mtx(output_spmqtf_file_name, coo_spm_qtf)
                    break
                # elif (resparse_top_percent in attempted_percent) and (convergence[epoch] < convergence_threshold):
                #     high = max(1.0, resparse_top_percent+0.1)
            print(f"Convergence: {convergence}")
            print()
            epoch = epoch + 1
        sys.stdout = original_stdout
    print(f"ReSparse & Quantify done, saved to {output_spmqtf_file_name}.")