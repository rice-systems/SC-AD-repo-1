#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <limits.h>
#include <time.h>

#include <cuda_runtime.h>
// #include "nvml.h"
#include "uvm_libs.h"

#define GB (1024UL*1024*1024*sizeof(uint8_t))
#define MB (1024UL*1024*sizeof(uint8_t))

int main() {
	void *buf1, *buf2;
	size_t bufferSize = GB * 10;
	double wall, cpu;
	cudaStream_t s0, s1;

	cudaStreamCreate(&s0);
	// cudaStreamCreate(&s1);
	cudaMalloc(&buf1, bufferSize);
	buf2 = malloc(sizeof(char) * bufferSize);
	memset(buf2, 0, sizeof(char) * bufferSize);

	for (int i = 0; i < 3; i ++)
	{
		wall = getTime();
		cudaMemcpy(buf1, buf2, bufferSize, cudaMemcpyDefault);
		cudaDeviceSynchronize();
		wall = getTime() - wall;

		wall = wall / 1e9;
		printf("cpu->gpu : %.2f GB/s\n",
				(double) bufferSize / GB / wall);
	}

	for (int i = 0; i < 3; i ++)
	{
		wall = getTime();
		cudaMemcpy(buf2, buf1, bufferSize, cudaMemcpyDefault);
		cudaDeviceSynchronize();
		wall = getTime() - wall;

		wall = wall / 1e9;
		printf("gpu->cpu : %.2f GB/s\n",
				(double) bufferSize / GB / wall);
	}
	return 0;
}
