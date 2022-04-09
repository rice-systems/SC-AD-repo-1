all: cpu gpu throughput discard #uvmPrefetch penalty allocate lru sanity thrashing discardlazy

cpu:
	nvcc -O2 prefetch_cpu_to_gpu.c uvm_libs.c -o cpu_to_gpu.x -lcuda -lcudart
gpu:
	nvcc -O2 prefetch_gpu_to_cpu.c uvm_libs.c -o gpu_to_cpu.x -lcuda -lcudart
throughput:
	nvcc -O2 throughput.c uvm_libs.c -o throughput.x -lcuda -lcudart
discard:
	nvcc -O2 discard_throughput.c uvm_libs.c -o discard.x -lcuda -lcudart

# uvmPrefetch:
# 	nvcc -O2 concurrency_uvm.c uvm_libs.c -o uvmprefetch.x -lcuda -lcudart
# penalty:
# 	nvcc -O2 false_discard.c uvm_libs.c -o penalty.x -lcuda -lcudart
# allocate:
# 	nvcc -O2 allocate.cu -o allocate.x -lcuda -lcudart
# lru:
# 	nvcc -O2 uvm_libs.c lru.c -o lru.x -lcuda -lcudart
# sanity:
# 	nvcc -O2 uvm_libs.cpp sanity.cpp -o sanity.x -lcuda -lcudart
# thrashing:
# 	nvcc -O2 uvm_libs.cpp discard_thrashing.cpp -o thrashing.x -lcuda -lcudart
# discardlazy:
# 	nvcc -O2 uvm_libs.cpp discardlazy.cpp -o discardlazy.x -lcuda -lcudart