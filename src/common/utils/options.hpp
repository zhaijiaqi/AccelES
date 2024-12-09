#pragma once

#include <getopt.h>
#include <string>
#include <cstdlib>

//////////////////////////////
//////////////////////////////

#define DEBUG false
#define RESET false
// Paths are absolute paths to the host executable!

// #define DEFAULT_MTX_FILE "/home/jqzhai/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/matrix_10000_512_40_uniform.mtx"
// #define DEFAULT_Q_MTX_FILE "/home/jqzhai/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/matrix_10000_512_40_uniform_B6.mtx"
// #define DEFAULT_MTX_FILE "/home/jqzhai/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/matrix_5000000_512_40_uniform.mtx_shuffled.mtx"
// #define DEFAULT_Q_MTX_FILE "/home/jqzhai/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/matrix_5000000_512_40_uniform.mtx_quantified_B6_ReSparse0.844.mtx"
// #define DEFAULT_MTX_FILE "/home/jqzhai/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/matrix_5000000_1024_20_uniform.mtx_shuffled.mtx"
// #define DEFAULT_Q_MTX_FILE "/home/jqzhai/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/matrix_5000000_1024_20_uniform.mtx_quantified_B6_ReSparse0.846.mtx"
// #define DEFAULT_Q_MTX_FILE "/home/jqzhai/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/matrix_10000000_768_40_uniform.mtx_quantified_B6_ReSparse0.822.mtx"
// #define DEFAULT_Q_MTX_FILE "/home/jqzhai/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/matrix_15000000_512_40_uniform.mtx_quantified_B6_ReSparse0.83.mtx"



// #define DEFAULT_MTX_FILE "/home/jqzhai/Topk-SpMV/data/matrices_for_testing/baidu_encyclopedia/sparse.sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5_shuffled.mtx"
// #define DEFAULT_Q_MTX_FILE "/home/jqzhai/Topk-SpMV/data/matrices_for_testing/baidu_encyclopedia/sparse.sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5_quantified_B6_ReSparse0.617.mtx"
// #define DEFAULT_Q_MTX_FILE "/home/jqzhai/Topk-SpMV/data/matrices_for_testing/baidu_encyclopedia/sparse.sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5_quantified_B6_ReSparse0.759.mtx"

// #define DEFAULT_MTX_FILE "/home/jqzhai/Topk-SpMV/data/matrices_for_testing/cc_zh/sparse.cc.zh.300.vec_shuffled.mtx"
// #define DEFAULT_Q_MTX_FILE "/home/jqzhai/Topk-SpMV/data/matrices_for_testing/cc_zh/sparse.cc.zh.300.vec_quantified_B6_ReSparse0.718.mtx"

// #define DEFAULT_MTX_FILE "/home/jqzhai/Topk-SpMV/data/matrices_for_testing/crawl_300d_2M/sparse.crawl-300d-2M.vec_shuffled.mtx"
// #define DEFAULT_Q_MTX_FILE "/home/jqzhai/Topk-SpMV/data/matrices_for_testing/crawl_300d_2M/sparse.crawl-300d-2M.vec_quantified_B6_ReSparse0.793.mtx"

#define DEFAULT_MTX_FILE "/home/jqzhai/Topk-SpMV/data/matrices_for_testing/glove6B/sparse.glove.6B.50d.txt_shuffled.mtx"
#define DEFAULT_Q_MTX_FILE "/home/jqzhai/Topk-SpMV/data/matrices_for_testing/glove6B/sparse.glove.6B.50d.txt_quantified_B6_ReSparse0.776.mtx"

// #define DEFAULT_MTX_FILE "/home/jqzhai/Topk-SpMV/data/matrices_for_testing/wiki_news_300d_1M/sparse-wiki-news-300d-1M.vec_shuffled.mtx"
// #define DEFAULT_Q_MTX_FILE "/home/jqzhai/Topk-SpMV/data/matrices_for_testing/wiki_news_300d_1M/sparse-wiki-news-300d-1M.vec_quantified_B6_ReSparse0.771.mtx"


#define DEFAULT_BLOCK_SIZE_1D 32
#define DEFAULT_BLOCK_SIZE_2D 8
#define DEFAULT_NUM_BLOCKS 64
#define DEFAULT_GPU_IMPL 0
#define DEFAULT_USE_HALF_PRECISION_GPU false

#define DEFAULT_NUM_TESTS 30

#define DEFAULT_TOP_K 100

#define XCLBIN "/data/sub1/jqzhai/program/Topk-SpMV/build/sw_emu/xilinx_u280_gen3x16_xdma_1_202211_1/spmv_bscsr_top_k_main.xclbin"



enum GPU_IMPL { CSR = 0, CSR_LIGHTSPMV = 1, COO = 2 };

struct Options {
    std::string matrix_path = DEFAULT_MTX_FILE;
    std::string q_matrix_path = DEFAULT_Q_MTX_FILE;
    bool use_sample_matrix = false;
    bool reset = RESET;
    uint num_tests = DEFAULT_NUM_TESTS;
    int debug = DEBUG;
    bool ignore_matrix_values = false;
    int top_k_value = DEFAULT_TOP_K;
    std::string xclbin_path = XCLBIN;
    GPU_IMPL gpu_impl = GPU_IMPL(DEFAULT_GPU_IMPL);
    bool use_half_precision_gpu = DEFAULT_USE_HALF_PRECISION_GPU;
    int block_size_1d = DEFAULT_BLOCK_SIZE_1D;
    int block_size_2d = DEFAULT_BLOCK_SIZE_2D;
    int num_blocks = DEFAULT_NUM_BLOCKS;

    Options(int argc, char *argv[]) {
        // m: path to the directory that stores the input matrix, stored as MTX
        // s: use a small example matrix instead of the input files
        // d: if present, print all debug information, else a single summary line at the end
        // t: if present, repeat the computation the specified number of times
    	// x: xclbin path
    	// v: if present, ignore values in the matrix and set all of them to 1; used to load graph topologies
        int opt;
        static struct option long_options[] = {{"debug", no_argument, 0, 'd'},
                                               {"use_sample_matrix", no_argument, 0, 's'},
											   {"no_reset", no_argument, 0, 'r'},
                                               {"matrix_path", required_argument, 0, 'm'},
                                               {"q_matrix_path", required_argument, 0, 'q'},
                                               {"num_tests", required_argument, 0, 't'},
                                               {"xclbin", required_argument, 0, 'x'},
											   {"ignore_matrix_values", no_argument, 0, 'v'},
											   {"k", required_argument, 0, 'k'},
                                               {"block_size_1d", required_argument, 0, 'b'},
                                               {"block_size_2d", required_argument, 0, 'c'},
                                               {"num_blocks", required_argument, 0, 'g'},
                                               {"gpu_impl", required_argument, 0, 'i'},
                                               {"half_precision_gpu", no_argument, 0, 'a'},
                                               {0, 0, 0, 0}};
        int option_index = 0;

        while ((opt = getopt_long(argc, argv, "dm:q:st:x:vk:rb:c:g:i:a", long_options, &option_index)) != EOF) {
            switch (opt) {
                case 'd':
                    debug = true;
                    break;
                case 'r':
                	reset = true;
					break;
                case 'm':
                    matrix_path = optarg;
                    break;
                case 'q':
                    q_matrix_path = optarg;
                    break;
                case 's':
                    use_sample_matrix = true;
                    break;
                case 't':
                    num_tests = atoi(optarg);
                    break;
                case 'x':
                    xclbin_path = optarg;
                    break;
                case 'v':
                	ignore_matrix_values = true;
					break;
                case 'k':
					top_k_value = atoi(optarg);
					break;
                case 'b':
                    block_size_1d = atoi(optarg);
                    break;
                case 'c':
                    block_size_2d = atoi(optarg);
                    break;
                case 'g':
                    num_blocks = atoi(optarg);
                    break;
                case 'i':
                    gpu_impl = GPU_IMPL(atoi(optarg));
                    break;
                case 'a':
                    use_half_precision_gpu = true;
                    break;
                default:
                    break;
            }
        }
    }
};
