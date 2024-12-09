import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, save_npz
from sparse_dot_topn import sp_matmul_topn
import random

def quantify_csr(csr, B):
    min_value = min(csr.data)
    max_value = max(csr.data)
    csr.data = csr.data/max(abs(min_value),abs(max_value))  # 将稀疏矩阵值缩放到(-1,1)
    M = 2**B - 1  # B位整数的最大值
    # 遍历矩阵中的每个元素，并应用量化公式
    quantized_data = np.clip(np.int32(np.round(M/2*csr.data)), int(-M/2)-1, int(M/2))
    # 剔除值为0的元素
    non_zero_mask = quantized_data != 0
    non_zero_data = quantized_data[non_zero_mask]
    non_zero_indices = csr.indices[non_zero_mask]
    # 更新indptr以反映剔除零元素后的新索引
    non_zero_indptr = np.zeros(csr.indptr.shape, dtype=csr.indptr.dtype)
    non_zero_indptr[0] = 0
    for i in range(1, len(csr.indptr)):
        non_zero_indptr[i] = non_zero_indptr[i-1] + np.sum(non_zero_mask[csr.indptr[i-1]:csr.indptr[i]])
    csr = csr_matrix((non_zero_data, non_zero_indices, non_zero_indptr), shape=csr.shape)
    return csr

def quantify_vec(vec, B):
    min_value = min(vec.data)
    max_value = max(vec.data)
    vec.data = vec.data/max(abs(min_value),abs(max_value))  # 将稀疏矩阵值缩放到(-1,1)
    M = 2**B - 1  # B位整数的最大值
    # 遍历矩阵中的每个元素，并应用量化公式
    quantized_data = np.clip(np.int32(np.round(M/2*vec.data)), int(-M/2)-1, int(M/2))
    vec = csr_matrix((quantized_data, vec.indices, vec.indptr), shape=vec.shape)
    return vec


def quantify_mtx(input_file, B, distribution):
    # Turn the arrays into a CSR;
    matrix = read_mtx(input_file)
    csr = csr_matrix(matrix)
    print(f"min_value={min(csr.data)}, max_value={max(csr.data)}")
    quantified_csr = quantify_csr(csr, B)
    print(f"Quantified:\tmin_value={min(quantified_csr.data)}, max_value={max(quantified_csr.data)}")
    # save_npz(output_file,quantized_csr)
    return quantified_csr


def read_mtx(input_file):
    x = None
    y = None
    val = None
    with open(input_file) as f:
        lines = f.readlines()
        size = int(lines[2].split(" ")[2])
        rows = int(lines[2].split(" ")[0])
        cols = int(lines[2].split(" ")[1])
        x = np.zeros(size, dtype=int)
        y = np.zeros(size, dtype=int)
        val = np.zeros(size)

        for i, l in enumerate(lines[3:]):
            x_c, y_c, v_c = l.split(" ")
            x_c = int(x_c) 
            y_c = int(y_c)
            v_c = float(v_c)
            x[i] = x_c - 1
            y[i] = y_c - 1
            val[i] = v_c
    return ((val,(x,y)))


def write_coo_mtx(output_file, coo):
    with open(output_file, 'w') as f:
        # 写入文件头
        f.write('%%MatrixMarket matrix coordinate real general\n')
        f.write('%\n')
        f.write(f'{coo.shape[0]} {coo.shape[1]} {coo.nnz}\n')
        # 写入非零元素
        for i in range(coo.nnz):
            f.write(f'{coo.row[i] + 1} {coo.col[i] + 1} {coo.data[i]}\n')
            

def shuffle_rows_coo(spm_coo):
    unique_rows = np.unique(spm_coo.row)
    shuffled_rows = np.random.permutation(unique_rows)
    new_row_indices = np.zeros_like(spm_coo.row)
    new_col_indices = np.zeros_like(spm_coo.col)
    new_data = np.zeros_like(spm_coo.data)
    new_index = 0 
    for new_row, old_row in enumerate(shuffled_rows):
        row_mask = spm_coo.row == old_row
        original_cols = spm_coo.col[row_mask]
        original_data = spm_coo.data[row_mask]
        new_row_indices[new_index:new_index + len(original_data)] = new_row
        new_col_indices[new_index:new_index + len(original_data)] = original_cols
        new_data[new_index:new_index + len(original_data)] = original_data
        new_index += len(original_data)
    
    shuffled_coo = coo_matrix((new_data, (new_row_indices, new_col_indices)), shape=spm_coo.shape)
    return shuffled_coo

def shuffle_rows_csr(spm_csr):
    num_rows = spm_csr.shape[0]
    shuffled_rows = np.random.permutation(num_rows)
    new_data = []
    new_indices = []
    new_indptr = [0]
    for new_row in shuffled_rows:
        start = spm_csr.indptr[new_row]
        end = spm_csr.indptr[new_row + 1]
        new_data.extend(spm_csr.data[start:end])
        new_indices.extend(spm_csr.indices[start:end])
        new_indptr.append(new_indptr[-1] + (end - start))
    shuffled_csr = csr_matrix((np.array(new_data), np.array(new_indices), np.array(new_indptr)), shape=spm_csr.shape)
    return shuffled_csr


def top_K_indices(array, k):
    sorted_indices = np.argsort(array)  # 对数组进行排序并返回索引
    top_K = sorted_indices[-k:]  # 获取排序后的最后K个索引
    return top_K


def split_csr_matrix(matrix, num_splits=2):
    nrows = matrix.shape[0]
    avg_rows_per_split = nrows // num_splits
    splits = []
    indptr = matrix.indptr
    indices = matrix.indices
    data = matrix.data
    # 分成32片，每次迭代生成一个split_matrix
    for i in range(num_splits):
        start = i * avg_rows_per_split
        end = (i + 1) * avg_rows_per_split
        if i == num_splits - 1:
            end = nrows
        row_data = data[indptr[start]:indptr[end]]
        row_indices = indices[indptr[start]:indptr[end]]
        row_ptr = [x-indptr[start] for x in indptr[start:end+1]]
        row_data = np.array(row_data)
        row_indices = np.array(row_indices)
        row_ptr = np.array(row_ptr)
        splits.append(csr_matrix((row_data, row_indices, row_ptr), shape=(end-start,matrix.shape[1])))  # 可能有问题
    return splits


def create_random_vec(cols):
    vec_np = np.random.uniform(-1,1,cols)
    vec = csr_matrix(vec_np)
    return vec


def resparse_csr(spm_csr, accuracy_threshold=0.8):
    print("Resparse CSR with top_percent={:.2%}".format(accuracy_threshold))
    # 获取矩阵的形状
    num_rows, num_cols = spm_csr.shape
    # 平均值
    avg_abs_val = np.mean(np.abs(spm_csr.data))
    # 用于存储结果的数组
    data = []
    indices = []
    indptr = [0]
    # 遍历每一行
    for row in range(num_rows):
        # 获取当前行的非零元素
        start = spm_csr.indptr[row]
        end = spm_csr.indptr[row + 1]
        row_data = spm_csr.data[start:end]
        non_zero_indices = spm_csr.indices[start:end]
        # 跳过空行
        if len(row_data) == 0:
            indptr.append(indptr[-1])
            continue
        # 该行所有元素都很重要（都超过平均重要度）
        if np.min(np.abs(row_data)) > avg_abs_val:
            data.extend(row_data)
            indices.extend(non_zero_indices)
        else:
            # 对非零元素进行排序
            sorted_indices = np.argsort(np.abs(row_data))[::-1]
            sorted_values = row_data[sorted_indices]
            sorted_cols = non_zero_indices[sorted_indices]
            num_to_keep = int(np.ceil(len(sorted_values) * accuracy_threshold))    # 计算需要保留的元素数量
            # 保留重要度大的的数据
            data.extend(sorted_values[:num_to_keep])
            indices.extend(sorted_cols[:num_to_keep])
        indptr.append(indptr[-1] + len(data) - indptr[-1])
    return csr_matrix((data, indices, indptr), shape=(num_rows, num_cols))


def check_empty_rows_csr(spm_qtf_csr):
    num_rows, num_cols = spm_qtf_csr.shape
    data = spm_qtf_csr.data.copy()
    indices = spm_qtf_csr.indices.copy()
    indptr = spm_qtf_csr.indptr.copy()
    
    for row in range(num_rows):
        start = indptr[row]
        end = indptr[row + 1]
        if start == end:
            # Row is empty
            col_to_fill = random.randint(0, num_cols - 1)
            data = np.insert(data, start, 1)
            indices = np.insert(indices, start, col_to_fill)
            for i in range(row + 1, num_rows + 1):
                indptr[i] += 1
    return csr_matrix((data, indices, indptr), shape=(num_rows, num_cols))


def print_num_nnz_csr(spm_csr, notes):
    nnz = spm_csr.nnz  # 直接获取非零元素的数量
    rows = spm_csr.shape[0]
    cols = spm_csr.shape[1]
    density = round(nnz/(rows*cols)*100,2)
    print(f"{notes}: Number of non-zero entries: {nnz}, Density: {density}%")
    return nnz


def cpt_convergence_matrics(full_csr, qtf_spm_csr, B, CK, test_num=100, split=32):
    K=[1, 8, 16, 32, 50, 75, 100]
    weight_coefficient = [1.0,1/8,1/16,1/32,1/50,1/75,1/100]
    avg_precision=[0]*len(K)
    rows = full_csr.shape[0]
    cols = full_csr.shape[1]
    nnz = qtf_spm_csr.nnz
    density = round(nnz/(rows*cols)*100,2)
    print(f"ROWS={rows}\tCOLS={cols}\tdensity={density}%\tsplit={split}\tTEST_NUM={test_num}\tB={B}")
    # matrics
    quantified_precision=[0]*len(K)
    total_precision=[0]*len(K)
    error_num=[0]*len(K)
    for i in range(test_num):
        vec = create_random_vec(cols)
        quantified_vec = quantify_vec(vec, B) # 测试发现vec采用的量化方式与matrix相同时，精度最高
        # full-precision results
        full_res = sp_matmul_topn(full_csr, vec.transpose(), 1, float('-inf'), n_threads=32)
        gold_ref = top_K_indices(full_res.data, CK)    # ref 直接求 Top-256
        # quanized_precison results
        quantified_top_256 = []
        if split==0:
            quantified_res = sp_matmul_topn(qtf_spm_csr, quantified_vec.transpose(), 1, -32768, n_threads=32)
            quantified_top_256 = top_K_indices(quantified_res.data, CK)
        # split to spieces
        else:
            split_quan_matrics = split_csr_matrix(qtf_spm_csr, split)
            for i, split_matrix in enumerate(split_quan_matrics):
                split_res = sp_matmul_topn(split_matrix,quantified_vec.transpose(), 1, -32768, n_threads=32)
                split_top_256 = top_K_indices(split_res.data, int(CK/32))+i*(rows//split)
                quantified_top_256.extend(split_top_256)
        # compare results
        for j,k in enumerate(K):
            quantified_precision[j] = len(np.intersect1d(quantified_top_256, gold_ref[-k:]))/k
            if quantified_precision[j]<1.0:    # count error num
                # print(f'{i+1}:Precision:{quantified_precision[i]*100}%')
                error_num[j]+=1
            total_precision[j] += quantified_precision[j]
    print(f"CK={CK}")
    for i,k in enumerate(K):
        avg_precision[i] = total_precision[i]/test_num
        print(f"K={k}\tavg_precison:{total_precision[i]/test_num*100}%\trecall:{(1-error_num[i]/test_num)*100}%")
    convergence_matrics = sum([avg_precision[i]*weight_coefficient[i] for i in range(len(K))])
    print(f"Convergence_matrics:{convergence_matrics}")
    return convergence_matrics