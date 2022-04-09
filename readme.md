This repo contains three benchmarks used to examine UvmDiscard/UvmDiscardLazy with Nvidia's UVM driver.
These benchmarks have been changed to use unified virtual memory with the CUDA API cudaMallocManaged.
They are also optimized with prefetch APIs. We also tried to overlap the computation with memory transfers.

The modified benchmarks are listed in three folders:
1. fir (https://github.com/NUCAR-DEV/Hetero-Mark/tree/Develop/src/fir)
2. radix-sort (https://github.com/utcs-scea/altis/tree/master/src/cuda/level1/sort)
3. hash-join (https://github.com/psiul/ICDE2019-GPU-Join)
4. darknet (https://github.com/AlexeyAB/darknet)

A microbenchmark in src-micro controls the available GPU memory to examine the performance under different GPU memory oversubscription ratios.
Folder uvmdiscard contains the user-space library to invoke the proposed memory advises in Nvidia's UVM driver. The customized driver code is proprietary but may be available in the future when it is open-sourced.

To collect the experimental results, first run ```make``` in folders: fir, radix-sort, hash-join, src-micro.
Then, use the script in the root folder to collect all benchmark results:
```
./run.sh
```

The collected data are saved in data-pcie-3 and data-pcie-4 folders.
Use the following commands to obtain the data presented in the paper:
```
cp -r data-pcie-3/4 data
python3 parse.py
```

For deep learning experiments please refer to darknet/readme.md.
The training datasets are imagenet-2012 and shakespare corpus.
