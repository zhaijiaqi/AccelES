# CPU benchmark

---

### 1.Dependency
Install `Intel MKL Libaray`: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html

将链接文件加到`LD_LIBRARY_PATH`中
```
    export LD_LIBRARY_PATH=/path/to/mkl-2023.1.0-h213fc3f_46344/lib:$LD_LIBRARY_PATH
```

### 2.Compile
```
    g++ -o test_mkl cpu_mkl_baseline.cpp -I/opt/intel/oneapi/mkl/2024.2/include/ -L /opt/intel/oneapi/mkl/2024.2/lib/intel64/ -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
```

### 3.Run
```
    ./test_mkl
```

**Note**

To run the `multicore` or `multicore resparse` versions, download the dataset to the corresponding path, modify `num_threads=32`, uncomment the corresponding path code and then recompile and run.


### 4.Results
#### 1 core 
```
Read csr done for /data/sub1/jqzhai/program/Topk-SpMV/data/matrices_for_testing/baidu_encyclopedia/sparse.sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5_shuffled.mtx!
compute time: 61.7609 ms
sort time: 13.6322 ms
exec time: 75.3931ms
Read csr done for /data/sub1/jqzhai/program/Topk-SpMV/data/matrices_for_testing/cc_zh/sparse.cc.zh.300.vec_shuffled.mtx!
compute time: 38.3772 ms
sort time: 40.5081 ms
exec time: 78.8853ms
Read csr done for /data/sub1/jqzhai/program/Topk-SpMV/data/matrices_for_testing/crawl_300d_2M/sparse.crawl-300d-2M.vec_quantified_B6_ReSparse0.638.mtx!
compute time: 301.947 ms
sort time: 40.253 ms
exec time: 342.2ms
Read csr done for /data/sub1/jqzhai/program/Topk-SpMV/data/matrices_for_testing/glove6B/sparse.glove.6B.50d.txt_shuffled.mtx!
compute time: 22.3741 ms
sort time: 8.97355 ms
exec time: 31.3477ms
Read csr done for /data/sub1/jqzhai/program/Topk-SpMV/data/matrices_for_testing/wiki_news_300d_1M/sparse-wiki-news-300d-1M.vec_shuffled.mtx!
compute time: 24.5341 ms
sort time: 20.8676 ms
exec time: 45.4017ms
Read csr done for /data/sub1/jqzhai/program/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/matrix_5000000_1024_20_uniform.mtx!
compute time: 96.9073 ms
sort time: 99.9001 ms
exec time: 196.807ms
Read csr done for /data/sub1/jqzhai/program/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/matrix_10000000_768_40_gamma.mtx_shuffled.mtx!
compute time: 345.544 ms
sort time: 198.904 ms
exec time: 544.448ms
Read csr done for /data/sub1/jqzhai/program/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/matrix_15000000_512_40_uniform.mtx!
compute time: 500.595 ms
sort time: 297.411 ms
exec time: 798.006ms
```

#### 32 core
```
Read csr done for /data/sub1/jqzhai/program/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/cpu_multicore_mtx/baidu_multicore.mtx!
avg compute time: 8.92997 ms
Read csr done for /data/sub1/jqzhai/program/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/cpu_multicore_mtx/cc_zh_multicore.mtx!
avg compute time: 6.21003 ms
Read csr done for /data/sub1/jqzhai/program/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/cpu_multicore_mtx/crawl_multicore.mtx!
avg compute time: 45.2263 ms
Read csr done for /data/sub1/jqzhai/program/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/cpu_multicore_mtx/glove_multicore.mtx!
avg compute time: 3.10515 ms
Read csr done for /data/sub1/jqzhai/program/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/cpu_multicore_mtx/wiki_news_multicore.mtx!
avg compute time: 3.75915 ms
Read csr done for /data/sub1/jqzhai/program/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/cpu_multicore_mtx/5M_multicore.mtx!
avg compute time: 10.5544 ms
Read csr done for /data/sub1/jqzhai/program/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/cpu_multicore_mtx/10M_multicore.mtx!
avg compute time: 37.1931 ms
Read csr done for /data/sub1/jqzhai/program/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/cpu_multicore_mtx/15M_multicore.mtx!
avg compute time: 58.1556 ms
```

#### 32 core ReSparse
```
Read csr done for /data/sub1/jqzhai/program/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/cpu_multicore_mtx/resparsed_baidu_multicore.mtx!
avg compute time: 5.56983 ms
Read csr done for /data/sub1/jqzhai/program/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/cpu_multicore_mtx/resparsed_cczh_multicore.mtx!
avg compute time: 6.03359 ms
Read csr done for /data/sub1/jqzhai/program/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/cpu_multicore_mtx/resparsed_crawl_multicore.mtx!
avg compute time: 42.9944 ms
Read csr done for /data/sub1/jqzhai/program/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/cpu_multicore_mtx/resparsed_glove_multicore.mtx!
avg compute time: 2.34097 ms
Read csr done for /data/sub1/jqzhai/program/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/cpu_multicore_mtx/resparsed_wikinews_multicore.mtx!
avg compute time: 3.72593 ms
Read csr done for /data/sub1/jqzhai/program/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/cpu_multicore_mtx/resparsed_5M_multicore.mtx!
avg compute time: 12.7406 ms
Read csr done for /data/sub1/jqzhai/program/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/cpu_multicore_mtx/resparsed_10M_multicore.mtx!
avg compute time: 27.8192 ms
Read csr done for /data/sub1/jqzhai/program/Topk-SpMV/data/matrices_for_testing/self_generated_csr/normal_distribution_+-/cpu_multicore_mtx/resparsed_15M_multicore.mtx!
avg compute time: 41.8475 ms
```
