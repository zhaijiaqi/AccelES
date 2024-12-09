### GPU benchmark

under folder `src/gpu`

```bash
make all
```

exec file will be generated under `bin` , execute and bench

```bash
./bin/approximate-spmv-gpu-csr-topk # add -d use debug mode
```

dataset path is defined in file `src/common/utils/options.hpp` 

```cpp
#define DEFAULT_MTX_FILE "your_mtx_file"
```