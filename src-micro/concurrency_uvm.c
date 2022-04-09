#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <limits.h>
#include <assert.h>
#include <time.h>

#include <cuda_runtime.h>
#include "nvml.h"
#include "uvm_libs.h"

#define GB (1024UL*1024*1024*sizeof(uint8_t))
#define MB (1024UL*1024*sizeof(uint8_t))

// You can always find this in nvidia-uvm/nvCpuUuid.c
NvUuid cpuId =
{
    {
       0xa6, 0x5e, 0x0f, 0x4e, 0xd7, 0xd4, 0x7b, 0xa2,
       0x50, 0x47, 0x41, 0x2c, 0x14, 0x2a, 0x77, 0x73
    }
};

NvUuid gpuId = 
{
	{  0x47, 0xef, 0xca, 0xec, 0x57, 0xab, 0x20, 0xb0, 
	   0x66, 0x21, 0x48, 0xfb, 0x6f, 0x4c, 0xd1, 0x59
	}
};

int main() {
	void *buf1, *buf2;
	size_t bufferSize = GB * 3;
	double wall, cpu;
	cudaStream_t s0, s1;

	cudaStreamCreate(&s0);
	cudaStreamCreate(&s1);
	cudaMallocManaged(&buf1, bufferSize, cudaMemAttachGlobal);
	cudaMallocManaged(&buf2, bufferSize, cudaMemAttachGlobal);
	// UvmMigrateAsync(buf1, bufferSize, &cpuId, s0);
	// UvmMigrateAsync(buf2, bufferSize, &cpuId, s1);
	cudaDeviceSynchronize();

	wall = getTime();
	cpu = clock();
	UvmMigrateAsync(buf1, bufferSize, &gpuId, s0);
	UvmMigrateAsync(buf2, bufferSize, &gpuId, s1);
	cudaDeviceSynchronize();
	wall = getTime() - wall;
	cpu = clock() - cpu;

	wall = wall / 1e9;
	cpu = (double) cpu / CLOCKS_PER_SEC;
	printf("latency : %.2f, cpu : %.2f%%, throughput : %.2f GB/s\n",
			wall, cpu / wall * 100, (double) bufferSize / GB * 2 / wall);
	return 0;
}