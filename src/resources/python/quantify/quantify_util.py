import numpy as np
import random

# 矩阵向量点乘
def matrix_vector_multiply(matrix, vector):
    return np.dot(matrix, vector)

def quantify_matrix(matrix, B, symmetric):
    """
    量化矩阵

    Args:
        matrix:     待量化的矩阵
        B:          量化位数 bits
        symmetric:  是否采用均匀量化 [True/False]

    Returns:
        返回量化后的矩阵/向量
    """
    
    M = 2**B-1
    if symmetric:   # 对称量化
        M = int(M/2)
    max_value = np.max(matrix)
    min_value = np.min(matrix)
    matrix /= max(abs(max_value),abs(min_value))
    matrix = np.clip(np.int32(np.round((M*matrix))), int(-M)-1, int(M))   # 均匀量化  （结论：线性变换的精度最高）
    return matrix


def  quantify_vector(vector, B, symmetric):
    """
    量化向量（保证和矩阵采用相同的量化放缩比例）

    Args:
        matrix:     待量化的矩阵
        B:          量化位数 bits
        symmetric:  是否采用均匀量化 [True/False]

    Returns:
        返回量化后的矩阵/向量
    """
    min_value = np.min(vector)
    max_value = np.max(vector)
    M = 2**B-1
    if symmetric:   # 对称量化
        M = int(M/2)
    vector /= max(abs(max_value),abs(min_value))
    return np.clip(np.int32(np.round(M*vector)), int(-M)-1, int(M))    # 均匀量化  （结论：线性变换的精度最高）


def top_K_indices(array, k):
    """
    返回top-k行索引
    
    Args:
        array: 一个数组
        k: 返回的索引个数
    
    Returns:
        返回top-k的索引
    """
    
    sorted_indices = np.argsort(array)  # 对数组进行排序并返回索引
    top_K = np.flip(sorted_indices[-k:])  # 获取排序后的最后K个索引
    # for k in top_K:
    #     print(array[k])
    return top_K

def top_K_results(array, k):
    """
    返回top-k行索引
    
    Args:
        array: 一个数组
        k: 返回的索引个数
    
    Returns:
        返回top-k的results
    """
    
    sorted_results = np.sort(array)  # 对数组进行排序并返回索引
    top_K = np.flip(sorted_results[-k:])  # 获取排序后的最后K个索引
    # for k in top_K:
    #     print(array[k])
    return top_K

def create_spm(rows, cols, spm_density, distribution):
    """
    生成稀疏矩阵spm和稠密向量v
    
    Args:
        rows: 稀疏矩阵的行数
        cols: 稀疏矩阵和稠密向量的列数
        spm_density: 稀疏矩阵的稠密度
        distribution: spm和v的元素数值服从的概率分布
    
    Returns:
        稀疏矩阵spm, 稠密向量v
    """
    
    if distribution == "uniform":
        m = np.random.rand(rows,cols)*2-1   # 取值范围（-1,1）
    elif distribution == "normal":
        m = np.random.randn(rows,cols)
    elif distribution == "gamma":
        m = np.random.gamma(3,2,(rows,cols))
    # 稠密矩阵随机稀疏化
    mask = np.random.choice([0, 1], size=m.shape, p=[1-spm_density, spm_density])
    spm = m*mask
    return spm

def extract_rows_by_indices(matrix, row_indices):
    """
    根据行坐标从矩阵中提取相应的行，然后将它们合并成一个新的矩阵

    Args:
        matrix: 原始矩阵    -> np.list
        row_indices: 行坐标list -> list

    Returns:
        new_matrix: 仅包含row_indices指定行的新矩阵
    """
    # 确保approximate_top_k_idx是一个列表
    if not isinstance(row_indices, list):
        raise ValueError("approximate_top_k_idx 必须是一个列表")
    # 提取指定行的数据
    extracted_rows = matrix[row_indices, :]
    return extracted_rows

def create_spm_gamma_nnz(rows, cols, density):
    """
    生成一个nnz值服从正态分布、每行nnz元素个数服从gamma分布的矩阵
    
    Args:
        row: spm行数
        cols: spm列数
        spm_density: spm非零元素占比
    Returns:
        返回spm, spm满足: 值服从正态分布、每行nnz元素个数服从gamma分布
    """
    # 从Gamma分布中生成每行包含的非零元素个数
    shape_param = 3
    scale_param = density*1000 / shape_param
    nonzero_per_row = np.random.gamma(shape_param, scale_param, rows).astype(int)
    # 创建稀疏矩阵
    spm = np.zeros((rows, cols))
    # 遍历每一行，随机选择列索引并设置非零值
    for i in range(rows):
        num_nonzero = nonzero_per_row[i]
        nonzero_indices = np.random.choice(cols, num_nonzero, replace=True)
        spm[i, nonzero_indices] = np.random.randn(num_nonzero)  # 将非零值设置为标准正态分布的随机值
    return spm
    
def create_v(cols,distribution):
    # if distribution == "uniform":
    #     v = np.random.rand(cols)*2-1        # 取值范围（-1,1）
    # elif distribution == "normal":
    #     v = np.random.randn(cols)
    # elif distribution == "gamma":
    #     v = np.random.gamma(3,2,cols)
    v = np.random.rand(cols)*2-1 
    return v

def create_spv(cols, density):
    # 生成一个随机向量，范围在 [-1, 1)
    spv = np.random.rand(cols) * 2 - 1
    # 根据密度决定哪些元素为零
    mask = np.random.rand(cols) < density
    spv[~mask] = 0
    return spv

def split_approximate(spm, v, split_size, APPROXIMATE_K):
    """
    返回分核近似后的approximate_top_k_row_index
    
    Args:
        spm:待切分的spm
        split:分成split个核
    
    Returns:
        返回分核近似后的approximate_top_k_row_index
    """
    # 将spm切分为split_size块
    slice_size = spm.shape[0] // split_size
    redundent_size = spm.shape[0] % split_size
    slices = [spm[i:i+slice_size] for i in range(0, split_size*slice_size, slice_size)]
    if redundent_size > 0:
        slices[split_size-1] = np.concatenate((slices[split_size-1], spm[-redundent_size:]), axis=0)
    # 每块分别进行spmv运算，统计top_sk， sk * slice = APPROXIMATE_K
    approximate_top_k = []
    for i,slice in enumerate(slices):
        result = matrix_vector_multiply(slice, v)
        top_sk = top_K_indices(result, int(APPROXIMATE_K/split_size))+i*slice_size
        approximate_top_k.extend(top_sk)
    return approximate_top_k

def compt_precision(ref_top_k_idx, approximate_top_k):
    """
    计算真实ref_top_k落入近似approximate_top_k的比率
    
    Args:
        ref_top_k_idx:全精度top100
        approximate_top_k:量化后topk
    
    Returns:
        返回精度k=[1,8,16,32,50,75,100]时的近似精度
    """
    
    # print(ref_top_k_idx[::-1])
    # print(approximate_top_k[::-1])
    # 统计精度
    # print(f"CK={APPROXIMATE_K}")
    top_k_precision = []
    top_ks = [1,8,16,32,50,75,100]
    for i,K in enumerate(top_ks):
        top_k_precision.append(len(list(set(approximate_top_k) & set(ref_top_k_idx[0:K])))/K)
        # print(f"K={K}\tprecision:{top_k_precision*100}%")
    return top_k_precision

def print_avg_precision(summery_precision, APPROXIMATE_K):
    """
    打印多次测试的平均精度
    
    Args:
        summery_precision: 多次测试的精度
        APPROXIMATE_K: 近似K
    
    Returns:
        打印结果
    """
    
    print(f"TEST_NUM={len(summery_precision)}")
    print(f"CK={APPROXIMATE_K}")
    col_sums = [sum(row[i] for row in summery_precision) for i in range(len(summery_precision[0]))]
    avg_precision = [col_sum / len(summery_precision) for col_sum in col_sums]
    recall_counts = [col.count(1.0) for col in zip(*summery_precision)]
    top_ks = [1,8,16,32,50,75,100]
    for i,K in enumerate(top_ks):
        print(f"K={K}\tavg_precision:{round(avg_precision[i]*100,2)}%\trecall={recall_counts[i]/len(summery_precision)*100}%")
    return
        
# 定义一个函数来执行单次迭代的测试
def run_single_precision_test(cols, distribution, spm, K, spm_qtf, B, symmetric, split, APPROXIMATE_K):
    """
    执行单次迭代的 precision 测试

    Args:
    
    Returns:

    单次测试的top-[1,8,16,32,50,75,100] precision
    """
    
    v = create_v(cols, distribution)        # dense v
    # v = spm[random.randint(0, len(spm)-1)]    # 随机取spm一行作为v SpV
    ref_result = matrix_vector_multiply(spm, v)
    ref_top_k_idx = top_K_indices(ref_result, K)
    v_qtf = quantify_vector(v, B, symmetric)
    if split == 0 or split == 1:
        approximate_result = matrix_vector_multiply(spm_qtf, v_qtf)
        approximate_top_k_idx = top_K_indices(approximate_result, APPROXIMATE_K)
    else:
        approximate_top_k_idx = split_approximate(spm=spm_qtf, v=v_qtf, split_size=split, APPROXIMATE_K=APPROXIMATE_K)
        # 输出真实结果落入每个分核中的个数
        # split_size = spm.shape[0] // split
        # interval_ranges = [(i * split_size, (i + 1) * split_size) for i in range(split)]
        # count_per_interval = [np.sum((ref_top_k_idx >= interval[0]) & (ref_top_k_idx < interval[1])) for interval in interval_ranges]
        # print(count_per_interval)
    top_k_precision = compt_precision(ref_top_k_idx, approximate_top_k_idx)
    # print(ref_top_k_idx,"\n",approximate_top_k)
    # print("\n")
    # if top_k_precision[6] < 0.95:
    #     print(f"{np.count_nonzero(v)}\t{np.max(v)}\t{np.min(v)}")
    #     print(top_k_precision)
    return top_k_precision

# 定义一个函数来执行单次迭代的测试
def run_single_spmspv_test(cols, density, spm, K, spm_qtf, B, symmetric, split, APPROXIMATE_K):
    """
    执行单次迭代的 precision 测试

    Args:
    
    Returns:

    单次测试的top-[1,8,16,32,50,75,100] precision
    """
    
    v = create_spv(cols, density)        # dense v
    # v = spm[random.randint(0, len(spm)-1)]    # 随机取spm一行作为v SpV
    ref_result = matrix_vector_multiply(spm, v)
    ref_top_k_idx = top_K_indices(ref_result, K)
    v_qtf = quantify_vector(v, B, symmetric)
    if split == 0 or split == 1:
        approximate_result = matrix_vector_multiply(spm_qtf, v_qtf)
        approximate_top_k_idx = top_K_indices(approximate_result, APPROXIMATE_K)
    else:
        approximate_top_k_idx = split_approximate(spm=spm_qtf, v=v_qtf, split_size=split, APPROXIMATE_K=APPROXIMATE_K)
        # 输出真实结果落入每个分核中的个数
        # split_size = spm.shape[0] // split
        # interval_ranges = [(i * split_size, (i + 1) * split_size) for i in range(split)]
        # count_per_interval = [np.sum((ref_top_k_idx >= interval[0]) & (ref_top_k_idx < interval[1])) for interval in interval_ranges]
        # print(count_per_interval)
    top_k_precision = compt_precision(ref_top_k_idx, approximate_top_k_idx)
    # print(ref_top_k_idx,"\n",approximate_top_k)
    # print("\n")
    # if top_k_precision[6] < 0.95:
    #     print(f"{np.count_nonzero(v)}\t{np.max(v)}\t{np.min(v)}")
    #     print(top_k_precision)
    return top_k_precision

def ndcg(sw_res_idx, sw_res_val, hw_res_idx, hw_res_val): 
    """
    测试 NDCG 指标

    Args: 软件仿真结果:sw_res_idx,sw_res_val, 硬件结果:hw_res_idx,hw_res_val
    
    Returns: ndcg, dcg, idcg
    """
    sw_res = {k: v for (k, v) in zip(sw_res_idx, sw_res_val)}
    dcg = 0
    idcg = 0
    for i, (idx, res) in enumerate(zip(hw_res_idx, hw_res_val)):
        relevance = sw_res[idx] if idx in sw_res else 0
        dcg += relevance / np.log2(i + 1 + 1)
    for i, (idx, res) in enumerate(zip(sw_res_idx, sw_res_val)):
        relevance = res
        idcg += relevance / np.log2(i + 1 + 1)
    return dcg / idcg, dcg, idcg

def run_single_ndcg_test(cols, distribution, spm, K, spm_qtf, B, symmetric, split, APPROXIMATE_K):
    """
    执行单次迭代的 ndcg 测试

    Args:
    
    Returns:

    单次测试的top-[1,8,16,32,50,75,100] precision
    """
    
    v = create_v(cols, distribution)        # dense v
    # v = spm[random.randint(0, len(spm)-1)]    # 随机取spm一行作为v SpV
    ref_result = matrix_vector_multiply(spm, v)
    v_qtf = quantify_vector(v, B, symmetric)
    # 获得 近似结果的idx
    if split == 0 or split == 1:
        approximate_result = matrix_vector_multiply(spm_qtf, v_qtf)
        approximate_top_k_idx = top_K_indices(approximate_result, APPROXIMATE_K)
    else:
        approximate_top_k_idx = split_approximate(spm=spm_qtf, v=v_qtf, split_size=split, APPROXIMATE_K=APPROXIMATE_K)

    # 根据 idx 重计算 res
    new_spm = extract_rows_by_indices(spm, approximate_top_k_idx)
    new_res = matrix_vector_multiply(new_spm, v)

    KS = [1,8,16,32,50,75,100]
    ndcg_precisions = []
    for k in KS:
        ref_top_k_idx = top_K_indices(ref_result, k)
        ref_top_k_res = top_K_results(ref_result, k)
        hw_top_k_res = top_K_results(new_res, k)
        _hw_top_k_idx = top_K_indices(new_res, k)
        hw_top_k_idx = []
        for _idx in _hw_top_k_idx:
            hw_top_k_idx.append(approximate_top_k_idx[_idx])
        # 计算 edcg 指标
        ep, dcg, idcg  = ndcg(ref_top_k_idx, ref_top_k_res, hw_top_k_idx, hw_top_k_res)
        ndcg_precisions.append(ep)
    # 返回 edcg 指标精度
    return ndcg_precisions


def resparse_spm(spm, accuracy_threshold=0.8):
    print("Resparse SpM with top_percent={:.2%}".format(accuracy_threshold))
    # 获取矩阵的形状
    num_rows, num_cols = spm.shape
    # 平均值
    avg_abs_val = np.mean(np.abs(spm))  
    # 用于存储结果的数组
    result = np.zeros_like(spm)
    # 遍历每一行
    for row in range(num_rows):
        # 获取当前行的非零元素
        row_data = spm[row, :]
        non_zero_indices = np.nonzero(row_data)[0]
        non_zero_values = row_data[non_zero_indices]
        # 跳过空行
        if len(non_zero_values) == 0:
            continue
        # 该行所有元素都很重要（都超过平均重要度）
        if np.min(np.abs(non_zero_values)) > avg_abs_val:
            result[row] = row_data
        else:
            # 对非零元素进行排序
            sorted_indices = np.argsort(np.abs(non_zero_values))[::-1]
            sorted_values = non_zero_values[sorted_indices]
            sorted_cols = non_zero_indices[sorted_indices]
            num_to_keep = int(np.ceil(len(sorted_values) * accuracy_threshold))    # 计算需要保留的元素数量
            # 保留重要度大的的数据
            result[row, sorted_cols[:num_to_keep]] = sorted_values[:num_to_keep]
    return result

def print_num_nnz(spm, notes):
    num_nnz = 0
    for row in spm:
        num_nnz += np.count_nonzero(row)
    print(f"{notes}: Number of non-zero entries: {num_nnz}, Density: {num_nnz/spm.size:.2%}")
    return num_nnz


def print_convergence_matrics(summery_precision, APPROXIMATE_K, K, split):
    """
    打印多次测试的平均精度
    
    Args:
        summery_precision: 多次测试的精度
        APPROXIMATE_K: 近似K
    
    Returns:
        打印结果
    """
    
    print(f"TEST_NUM={len(summery_precision)}")
    print(f"SPLIT={split}")
    print(f"CK={APPROXIMATE_K}")
    print(f"K={K}")
    col_sums = [sum(row[i] for row in summery_precision) for i in range(len(summery_precision[0]))]
    avg_precision = [col_sum / len(summery_precision) for col_sum in col_sums]
    recall_counts = [col.count(1.0) for col in zip(*summery_precision)]
    top_ks = [1,8,16,32,50,75,100]
    weight_coefficient = [1.0,1/8,1/16,1/32,1/50,1/75,1/100]
    for i,K in enumerate(top_ks):
        print(f"K={K}\tavg_precision:{round(avg_precision[i]*100,2)}%\trecall={recall_counts[i]/len(summery_precision)*100}%")
    convergence_matrics = sum([avg_precision[i]*weight_coefficient[i] for i in range(len(top_ks))])
    return convergence_matrics

# 检查spm_qtf是否存在空行并随机填充1
def check_empty_rows(spm_qtf):
    num_rows, num_cols = spm_qtf.shape
    for row in range(num_rows):
        if np.count_nonzero(spm_qtf[row]) == 0:
            # print(f"Warning: Row {row} is empty")
            spm_qtf[row, random.randint(0, num_cols-1)] = 1

class COOEntry:
    def __init__(self, row, col, value):
        self.row = row
        self.col = col
        self.value = value

def read_matrix_from_txt(filename):
    matrix = []
    with open(filename, 'r') as file:
        for line in file:
            row_values = [float(val) for val in line.split()[1:]]
            matrix.append(row_values)
    return matrix

def matrix_to_coo(matrix):
    coo_entries = []
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if value != 0:
                coo_entries.append(COOEntry(i, j, value))
    return coo_entries

def write_coo_to_mtx(coo_entries, filename):
    with open(filename, 'w') as file:
        row = max(entry.row for entry in coo_entries) + 1
        col = max(entry.col for entry in coo_entries) + 1
        num_nonzero = len(coo_entries)
        header = f"%%MatrixMarket matrix coordinate real general\n%\n{row} {col} {num_nonzero}\n"
        file.write(header)
        for entry in coo_entries:
            file.write(f"{entry.row+1} {entry.col+1} {entry.value}\n")

def parse_txt_to_spm(path):
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
    np.random.shuffle(spm)  # 按行随机重排
    return spm


def read_coo_matrix(file_path):
    # 读取文件内容
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # 跳过文件头部的注释行
    data_lines = lines[2:]
    
    # 初始化COO格式的数据
    rows = []
    cols = []
    values = []
    
    # 解析每一行的数据
    for line in data_lines:
        row, col, value = map(float, line.split())
        rows.append(int(row) - 1)  # 转换为0-based索引
        cols.append(int(col) - 1)  # 转换为0-based索引
        values.append(value)
    
    # 获取矩阵的维度
    num_rows = int(lines[2].split()[0])
    num_cols = int(lines[2].split()[1])
    
    # 创建一个全零矩阵
    matrix = np.zeros((num_rows, num_cols))
    
    # 将COO格式的数据填充到矩阵中
    for i in range(len(rows)):
        matrix[rows[i], cols[i]] = values[i]
    
    return matrix