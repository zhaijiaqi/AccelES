#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cstdlib>
#include <pthread.h>
#include <algorithm>
#include <chrono>
#include <random>
#include <queue>
#include <functional>
#include "mkl.h"


typedef struct {
    float value;
    int index;
} Element;

typedef struct {
    int thread_id;
    sparse_matrix_t A;
    float *x;
    float *y;
    int row;
    int col;
    int k;
    int num_threads;
    Element* top_k;
} SpMVThreadData;

struct COOMatrix {
    std::vector<int> row;
    std::vector<int> col;
    std::vector<float> val;
    int rows;
    int cols;
    int size;
};


COOMatrix read_coo_matrix(const std::string& input_path, bool zero_index = true) {
    COOMatrix coo_matrix;
    std::ifstream file(input_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << input_path << std::endl;
        exit(1);
    }

    std::string line;
    std::getline(file, line);
    std::getline(file, line);
    std::getline(file, line);
    std::istringstream iss(line);
    iss >> coo_matrix.rows >> coo_matrix.cols >> coo_matrix.size;

    coo_matrix.row.resize(coo_matrix.size);
    coo_matrix.col.resize(coo_matrix.size);
    coo_matrix.val.resize(coo_matrix.size);

    int index_offset = zero_index ? 0 : 1;

    for (int i = 0; i < coo_matrix.size; ++i) {
        std::getline(file, line);
        std::istringstream iss(line);
        int x_c, y_c;
        float v_c;
        iss >> x_c >> y_c >> v_c;
        coo_matrix.row[i] = x_c - index_offset;
        coo_matrix.col[i] = y_c - index_offset;
        coo_matrix.val[i] = v_c;
    }

    file.close();
    return coo_matrix;
}

sparse_matrix_t coo_to_csr(const COOMatrix& coo_matrix) {
    sparse_matrix_t A;
    sparse_status_t status;

    // Create temporary non-const arrays
    std::vector<int> row_indices(coo_matrix.row.begin(), coo_matrix.row.end());
    std::vector<int> col_indices(coo_matrix.col.begin(), coo_matrix.col.end());
    std::vector<float> values(coo_matrix.val.begin(), coo_matrix.val.end());

    // Create a COO format sparse matrix
    sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;
    status = mkl_sparse_s_create_coo(&A, indexing, coo_matrix.rows, coo_matrix.cols, coo_matrix.size,
                                     row_indices.data(), col_indices.data(), values.data());
    if (status != SPARSE_STATUS_SUCCESS) {
        std::cerr << "Failed to create COO matrix" << std::endl;
        exit(1);
    }

    // Convert COO to CSR
    sparse_matrix_t csr_matrix = NULL;
    status = mkl_sparse_convert_csr(A, SPARSE_OPERATION_NON_TRANSPOSE, &csr_matrix);
    if (status != SPARSE_STATUS_SUCCESS) {
        std::cerr << "Failed to convert COO to CSR" << std::endl;
        exit(1);
    }

    // Destroy the temporary COO matrix
    mkl_sparse_destroy(A);

    return csr_matrix;
}

float *generate_random_vector(int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    float *x = (float *)malloc(cols * sizeof(float));
    for (int i = 0; i < cols; ++i) {
        x[i] = dis(gen) * 2 - 1;
    }

    return x;
}

bool compare_elements(const Element &a, const Element &b) {
    return a.value > b.value;
}

void top_k_heap(float *y, int n, int k, Element *top_k) {
    // define min_heap
    std::priority_queue<Element, std::vector<Element>, std::function<bool(Element, Element)>> min_heap(
        [](const Element &a, const Element &b) {
            return a.value > b.value;
        }
    );

    for (int i = 0; i < n; i++) {
        if (min_heap.size() < k) {
            min_heap.push({y[i], i});
        } else if (y[i] > min_heap.top().value) {
            min_heap.pop();
            min_heap.push({y[i], i});
        }
    }

    for (int i = k - 1; i >= 0; i--) {
        top_k[i] = min_heap.top();
        min_heap.pop();
    }
}

void *spmv_thread(void *arg) {
    SpMVThreadData *data = (SpMVThreadData *)arg;
    sparse_matrix_t A = data->A;
    float *x = data->x;
    float *y = data->y;
    int row = data->row;
    int col = data->col;
    int k = data->k;
    Element* top_k = data->top_k;
    int thread_id = data->thread_id;
    int num_threads = data->num_threads;

    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    // cpt
    mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, descr, x, 0.0, y);
    // sort
    top_k_heap(y, row, k / num_threads, top_k);
    pthread_exit(NULL);
}

void print_sparse_matrix(sparse_matrix_t A) {
    sparse_status_t status;
    sparse_index_base_t indexing;
    MKL_INT *row_start, *row_end;
    float *values;
    MKL_INT *col_index;
    MKL_INT rows, cols, nnz;

    status = mkl_sparse_s_export_csr(A, &indexing, &rows, &cols, &row_start, &row_end, &col_index, &values);
    if (status != SPARSE_STATUS_SUCCESS) {
        std::cerr << "Failed to export CSR matrix" << std::endl;
        return;
    }

    nnz = row_start[rows];

    std::cout << "Number of rows: " << rows << std::endl;
    std::cout << "Number of columns: " << cols << std::endl;
    std::cout << "Number of non-zero elements: " << nnz << std::endl;\
    std::cout << std::endl;
}

void parallel_spmv(COOMatrix cooA, int row, int col, int k, int num_threads, Element* top_k, int num_test) {
    pthread_t threads[num_threads];
    SpMVThreadData thread_data[num_threads];
    float* y = (float*)malloc(row * sizeof(float));

    for (int i = 0; i < num_threads; i++) {
        thread_data[i].y = y;
        thread_data[i].A = coo_to_csr(cooA);
        thread_data[i].row = row;
        thread_data[i].col = col;
        thread_data[i].k = k;
        thread_data[i].top_k = top_k;
        thread_data[i].num_threads = num_threads;
        // std::cout << "i = " << i << std::endl;
        // print_sparse_matrix(thread_data[i].A);
    }

    std::chrono::duration<double> cpt_time = (std::chrono::duration<double>)0;
    std::chrono::duration<double> sum_cpt_time = (std::chrono::duration<double>)0;
    for (int i = 0;i < num_test;i++) {
        float* x = generate_random_vector(col);
        for (int i = 0; i < num_threads; i++) {
            thread_data[i].x = x;
        }
        // cpt
        auto start_cpt = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_threads; i++) {
            thread_data[i].thread_id = i; 
            pthread_attr_t attr;
            pthread_attr_init(&attr);
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);        
            CPU_SET(i, &cpuset);      
            pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset);
            pthread_create(&threads[i], &attr, spmv_thread, (void*)&thread_data[i]);
            pthread_attr_destroy(&attr);
        }
        auto end_cpt = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> cpt_time = end_cpt - start_cpt;
        sum_cpt_time += cpt_time;
        // std::cout << "compute time: " << cpt_time.count() * 1000 << " ms" << std::endl;
        free(x);
    }
    std::cout << "avg compute time: " << sum_cpt_time.count() / num_test * 1000 << " ms" << std::endl;

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
        mkl_sparse_destroy(thread_data[i].A);
    }
    free(y);
}



void top_k_spmv(sparse_matrix_t A, int row, int col, int k, int num_threads, Element* top_k, int num_test) {
    float* y = (float*)malloc(row * sizeof(float));
    
    std::chrono::duration<double> cpt_time = (std::chrono::duration<double>)0;
    std::chrono::duration<double> sort_time = (std::chrono::duration<double>)0;
    for (int i = 0;i < num_test;i++) {

        float* x = generate_random_vector(col);

        struct matrix_descr descr;
        descr.type = SPARSE_MATRIX_TYPE_GENERAL;
        auto start_cpt = std::chrono::high_resolution_clock::now();
        mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, descr, x, 0.0, y);
        auto end_cpt = std::chrono::high_resolution_clock::now();
        cpt_time += (end_cpt - start_cpt);

        auto start_sort = std::chrono::high_resolution_clock::now();
        top_k_heap(y, row, k, top_k);
        auto end_sort = std::chrono::high_resolution_clock::now();
        sort_time += (end_sort - start_sort);
        free(x);
    }
    std::cout << "compute time: " << cpt_time.count() / num_test * 1000 << " ms" << std::endl;
    std::cout << "sort time: " << sort_time.count() / num_test * 1000 << " ms" << std::endl;
    std::cout << "exec time: " << (cpt_time.count() + sort_time.count()) / num_test * 1000 << "ms" << std::endl;
    free(y);
}

int main() {
    std::vector<std::string> input_paths = {

        // 1 core
        "/home/jqzhai/Topk-SpMV/data/matrices_for_testing/baidu_encyclopedia/sparse.sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5_shuffled.mtx",
        "/home/jqzhai/Topk-SpMV/data/matrices_for_testing/cc_zh/sparse.cc.zh.300.vec_shuffled.mtx",
        "/home/jqzhai/Topk-SpMV/data/matrices_for_testing/crawl_300d_2M/sparse.crawl-300d-2M.vec_shuffled.mtx",
        "/home/jqzhai/Topk-SpMV/data/matrices_for_testing/glove6B/sparse.glove.6B.50d.txt_shuffled.mtx",
        "/home/jqzhai/Topk-SpMV/data/matrices_for_testing/wiki_news_300d_1M/sparse-wiki-news-300d-1M.vec_shuffled.mtx",
        "/home/jqzhai/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/matrix_5000000_1024_20_uniform.mtx",
        "/home/jqzhai/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/matrix_10000000_768_40_gamma.mtx_shuffled.mtx",
        "/home/jqzhai/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/matrix_15000000_512_40_uniform.mtx"

        // multicore
        // "/home/jqzhai/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/cpu_multicore_mtx/baidu_multicore.mtx",
        // "/home/jqzhai/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/cpu_multicore_mtx/cc_zh_multicore.mtx",
        // "/home/jqzhai/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/cpu_multicore_mtx/crawl_multicore.mtx",
        // "/home/jqzhai/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/cpu_multicore_mtx/glove_multicore.mtx",
        // "/home/jqzhai/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/cpu_multicore_mtx/wiki_news_multicore.mtx",
        // "/home/jqzhai/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/cpu_multicore_mtx/5M_multicore.mtx",
        // "/home/jqzhai/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/cpu_multicore_mtx/10M_multicore.mtx",
        // "/home/jqzhai/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/cpu_multicore_mtx/15M_multicore.mtx"

        // multicore resparse
        // "/home/jqzhai/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/cpu_multicore_mtx/resparsed_baidu_multicore.mtx",
        // "/home/jqzhai/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/cpu_multicore_mtx/resparsed_cczh_multicore.mtx",
        // "/home/jqzhai/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/cpu_multicore_mtx/resparsed_crawl_multicore.mtx",
        // "/home/jqzhai/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/cpu_multicore_mtx/resparsed_glove_multicore.mtx",
        // "/home/jqzhai/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/cpu_multicore_mtx/resparsed_wikinews_multicore.mtx",
        // "/home/jqzhai/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/cpu_multicore_mtx/resparsed_5M_multicore.mtx",
        // "/home/jqzhai/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/cpu_multicore_mtx/resparsed_10M_multicore.mtx",
        // "/home/jqzhai/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/cpu_multicore_mtx/resparsed_15M_multicore.mtx"
    };
    bool zero_index = false;
    int k = 100;               // Top-K
    int num_threads = 1;
    int num_test=100;

    for (const auto& input_path : input_paths) {
        COOMatrix coo_matrix = read_coo_matrix(input_path, zero_index);
        sparse_matrix_t A = coo_to_csr(coo_matrix);

        std::cout << "Read csr done for " << input_path << "!" << std::endl;
        int row = coo_matrix.rows;
        int col = coo_matrix.cols;
        Element* top_k = (Element*)malloc(k * sizeof(Element));
        if(num_threads == 1){
            top_k_spmv(A, row, col, k, num_threads, top_k, num_test);
        } else{
            parallel_spmv(coo_matrix, row, col, k, num_threads, top_k, num_test);
        }
        free(top_k);
        mkl_sparse_destroy(A);
    }
    return 0;
}