# AccelES: Accelerating Top-K SpMV for Embedding Similarity via Low-bit Pruning

---

Top-K sparse matrix-vector multiplication (SpMV) is critical for embedding similarity calculations in high-dimensional data analysis but faces challenges from the mismatch between large-scale sparse matrices and traditional cache architectures in real-time applications. To address this, we propose **AccelES**, a Top-K SpMV accelerator that improves bandwidth utilization and reduces redundant computations through low-bit quantization, novel sparse matrix storage formats (**Ultra-CSR** and **Random-CSR**), and a non-zero granularity pruning algorithm (**ReSparse**). AccelES efficiently identifies Top-K rows using low-bit SpMV and performs precise computations for these rows, minimizing computational and memory overhead while optimizing data transmission.  

## Dependencies, Compilation and Running

---

### 1.External Dependencies
#### Hardware Dependencies
Our experiments use a server with hardware dependencies shown below.
```
[CPU]  : Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz
[DRAM] : 256GB DDR4 238GB/s
[GPU]  : H100-PCIE-80GB HBM2e
[FPGA] : Xilinx Alveo U280  platform: xilinx_u280_gen3x16_xdma_1_202211_1
```
#### Software Dependencies
Before running AccelES codes, it's essential that you have already install software dependencies shown below.
```
ubuntu 18.04
g++ (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0
GNU Make 4.1
Vitis_HLS 2023.2
```
#### Data Dependencies

We use 11 sarse matrices, including 5 classic embedding corpora and 6 large sparse matrices with tens of millions of rows.

The download links for the five public word vector datasets are as follows:

| Datasets     | Links                                                                                             |
|--------------|---------------------------------------------------------------------------------------------------|
|Baidu         | [download](https://pan.baidu.com/s/1Rn7LtTH0n7SHyHPfjRHbkg)                                       |
|cc_zh         | [download](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.zh.300.vec.gz)                |
|Crawl         | [download](https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip)         |
|GloVe         | [download](https://nlp.stanford.edu/data/glove.6B.zip)                                            |
|Wiki News     | [download](https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip)     |

The original datasets are dense. To sparsify them, we used [sparse-coding](https://github.com/mfaruqui/sparse-coding) to process all datasets. After sparsification, the column dimensions of the datasets are 900, 900, 900, 500, and 900, respectively.

The six large sparse matrices were generated using the script `./src/resources/python/create_matrices.py`. To customize the matrices, simply adjust the values of `NUM_ROWS`, `MAX_COLS`, `DISTRIBUTION`, and `AVERAGE_DEGREE` as needed. 

To quickly validate the performance of the algorithm and accelerator, we provide preprocessed sparse matrix samples.

| Datasets                     | Links                                                                                             |
|----------------------        |---------------------------------------------------------------------------------------------------|
|Spasified_GloVe (txt)         | [download](https://pan.baidu.com/s/1b0WXdXkQdOSvx4IJf-IZ2A) (pwd:1208)                                       |
|Spasified_GloVe (mtx)         | [download](https://pan.baidu.com/s/1Y57m7dzI3XeQ0HBNOMUp4A) (pwd:1208)                                       |
|Quantified_GloVe (mtx)        | [download](https://pan.baidu.com/s/18PjJ_CtCyWukyI54lSnPFQ) (pwd:1208)                                       |
|glove_multicore (mtx)         | [download](https://pan.baidu.com/s/19JjCaVY9TX6HpKqhfksIyw) (pwd:1208)                                       |
|resparse_glove_multicore (mtx) | [download](https://pan.baidu.com/s/1xVv3ol0IPvYpksHeb06KKg) (pwd:1208)                                       |

The generated files are saved in the specified directory. For example, the GloVe-related data is stored in `/path/to/AccelES/data/matrices_for_testing/glove6B`.

### 2.Preprocessing (Quantify & ReSparse)

#### Software Dependencies
Before running Preprocessing codes, it's essential that you have already install software dependencies shown below.

```
conda create --name acceles python=3.8
conda activate acceles
pip install -r requirements.txt
```
#### Run (Quantify & ReSparse) 

to obtain the precision results based on a CPU with 32 cores, along with quantization and ReSparse, and generate the corresponding Sparse Matrices `.mtx` files.
```
cd /path/to/AccelES
cd ./src/resources/python/quantify/datasets
./search.sh
```

**Note:**
Different datasets have an optimal `accuracy_threshold` value. We recommend using `0.999` as the standard.


After running, the following content will be output in the corresponding files. A total of 10 epochs are executed, and the execution time for each epoch depends on the size of the dataset. For the GloVe dataset, the average execution time per epoch is approximately 19 seconds.

```
Processing /home/jqzhai/Topk-SpMV/data/data_for_testing/glove6B/sparse.glove.6B.50d.txt...
Epoch:  0
Before ReSparse: Number of non-zero entries: 27659149, Density: 13.83%
Resparse SpM with top_percent=100.00%
After ReSparse: Number of non-zero entries: 27659149, Density: 13.83%
After quantification: Number of non-zero entries: 25968341, Density: 12.98%
TEST_NUM=100
SPLIT=32
CK=512
K=100
K=1     avg_precision:100.0%    recall=100.0%
K=8     avg_precision:100.0%    recall=100.0%
K=16    avg_precision:100.0%    recall=100.0%
K=32    avg_precision:100.0%    recall=100.0%
K=50    avg_precision:100.0%    recall=100.0%
K=75    avg_precision:100.0%    recall=100.0%
K=100   avg_precision:100.0%    recall=100.0%
Convergence: [1.2620833333333334, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

Elapsed time: 27.543482780456543 seconds
Epoch:  1
Before ReSparse: Number of non-zero entries: 27659149, Density: 13.83%
Resparse SpM with top_percent=50.00%
After ReSparse: Number of non-zero entries: 14052734, Density: 7.03%
After quantification: Number of non-zero entries: 14052738, Density: 7.03%
TEST_NUM=100
SPLIT=32
CK=512
K=100
K=1     avg_precision:100.0%    recall=100.0%
K=8     avg_precision:100.0%    recall=100.0%
K=16    avg_precision:99.62%    recall=94.0%
K=32    avg_precision:99.25%    recall=78.0%
K=50    avg_precision:98.3%     recall=41.0%
K=75    avg_precision:97.03%    recall=14.000000000000002%
K=100   avg_precision:95.04%    recall=0.0%
Convergence: [1.2620833333333334, 1.260382138888889, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

Elapsed time: 28.357963800430298 seconds
Epoch:  2
Before ReSparse: Number of non-zero entries: 27659149, Density: 13.83%
Resparse SpM with top_percent=75.00%
After ReSparse: Number of non-zero entries: 20954935, Density: 10.48%
After quantification: Number of non-zero entries: 20954905, Density: 10.48%
TEST_NUM=100
SPLIT=32
CK=512
K=100
K=1     avg_precision:100.0%    recall=100.0%
K=8     avg_precision:100.0%    recall=100.0%
K=16    avg_precision:100.0%    recall=100.0%
K=32    avg_precision:100.0%    recall=100.0%
K=50    avg_precision:100.0%    recall=100.0%
K=75    avg_precision:99.97%    recall=98.0%
K=100   avg_precision:99.94%    recall=95.0%
Convergence: [1.2620833333333334, 1.260382138888889, 1.262073777777778, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

...
```

### 3.Run CPU Benchmark

See `./src/cpu/README.md`

### 4.Run GPU Benchmark

See `./src/gpu/README.md`

### 5.Run FPGA sw_emu

Before executing the following process, please ensure that the environment is properly configured according to the [Vitis Tutorial](https://github.com/zhaijiaqi/Vitis-Tutorials/tree/2023.2/Getting_Started/Vitis).

1. Compile

    Building for software emulation is quick and should not take more than a few minutes. This step is for verifying the behavior of the system.
    ```
    ./build_sw_emu.sh
    ```
2. Run
    ```
    ./run_sw_emu.sh
    ```
3. Result

    You should see the following messages, indicating that the run completed successfully:

    ```
    ...
    Sort time: 0.124 ms
    precision=100%
    fpga exec time=2729.42 ms
    ----------------
    Mean FPGA execution time=0±0 ms
    Mean read-back time=0±0 ms
    Mean precision=100%±0%
    ----------------
    device process sw_emu_device done
    ...
    ```

### 6.Run FPGA hw

1. Compile

    Building for hardware targets can take a couple of hours.
    ```
    ./build_hw.sh
    ```
2. Run

    After the build completes you can run the application on a system with the AMD Alveo™ U280 Data Center accelerator card using the following command:
    ```
    ./run_hw.sh
    ```
3. Results
    ```
    ...
    Sort time: 0.017 ms
    precision=100%
    fpga exec time=0.25767 ms
    ----------------
    Mean FPGA execution time=0.2607±0.01172 ms
    Mean read-back time=0.06764±0.01172 ms
    Mean precision=100%±0%
    ----------------
    ...
    ```