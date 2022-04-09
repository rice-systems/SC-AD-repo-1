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
#define KB (1024UL*sizeof(uint8_t))

int lazy = 0;
double discard, prefetch;
double measure(void *buf1, size_t bufferSize, cudaStream_t s0)
{
	double wall;
	wall = getTime();
	UvmDiscard(buf1, bufferSize, lazy);
	wall = getTime() - wall;
	discard = wall / 1e9;
	
	wall = getTime();
	cudaMemPrefetchAsync(buf1, bufferSize, 0, s0);
	cudaDeviceSynchronize();
	wall = getTime() - wall;
	prefetch = wall / 1e9;

	// printf("throughput : %.2f GB/s\n", (double) bufferSize * 2 / GB / (cpu_to_gpu + gpu_to_cpu));
	// return bufferSize / discard / GB;
	return 0;
}

int main(int argc, char* argv[]) {
	void *buf1;
	size_t bufferSizes[24] = {
		KB*1, KB*2, 4*KB, 8*KB, 16*KB, 32*KB, 64*KB, 128*KB, 256*KB, 512*KB, MB*1, 
		MB*2, MB*4, MB*8, MB*16, MB*32, MB*64, MB*128, MB*256, MB*512,
		GB*1, GB*2, GB*4, GB*8}, bufferSize = GB*10;
	double ans, ans1, ans2, prefetches[24], discards[24];
	cudaStream_t s0;

	if (argc == 2)
		lazy = atoi(argv[1]);
	
	cudaStreamCreate(&s0);
	cudaMallocManaged(&buf1, bufferSize, cudaMemAttachGlobal);
	cudaMemPrefetchAsync(buf1, bufferSize, cudaCpuDeviceId, s0);	
	cudaMemPrefetchAsync(buf1, bufferSize, 0, s0);	
	cudaDeviceSynchronize();

	printf("warmup \n");
	ans = measure(buf1, bufferSize, s0);
	printf("warmup throughput 1 %.2f\n", (double) bufferSize / discard / GB);
	ans = measure(buf1, bufferSize, s0);
	printf("warmup throughput 2 %.2f\n", (double) bufferSize / discard / GB);
	for (int i = 0; i < 11; i ++)
	{
		ans1 = 0; ans2 = 0;
		measure(buf1, bufferSizes[i], s0);
		ans1 += (double) bufferSizes[i] / GB / discard; 
		ans2 += (double) bufferSizes[i] / GB / prefetch;
		measure(buf1, bufferSizes[i], s0);
		ans1 += (double) bufferSizes[i] / GB / discard; 
		ans2 += (double) bufferSizes[i] / GB / prefetch;
		measure(buf1, bufferSizes[i], s0);
		ans1 += (double) bufferSizes[i] / GB / discard; 
		ans2 += (double) bufferSizes[i] / GB / prefetch;
		discards[i] = ans1 / 3;
		prefetches[i] = ans2 / 3;
	}
	for (int i = 11; i < 24; i ++)
	{
		ans1 = 0; ans2 = 0;
		measure((char *) buf1 + 16 * 1024 * 1024, bufferSizes[i], s0);
		ans1 += (double) bufferSizes[i] / GB / discard; 
		ans2 += (double) bufferSizes[i] / GB / prefetch;
		measure((char *) buf1 + 16 * 1024 * 1024, bufferSizes[i], s0);
		ans1 += (double) bufferSizes[i] / GB / discard; 
		ans2 += (double) bufferSizes[i] / GB / prefetch;
		measure((char *) buf1 + 16 * 1024 * 1024, bufferSizes[i], s0);
		ans1 += (double) bufferSizes[i] / GB / discard; 
		ans2 += (double) bufferSizes[i] / GB / prefetch;
		discards[i] = ans1 / 3;
		prefetches[i] = ans2 / 3;
	}

	printf("discard:\n");
	for (int i = 0; i < 24; i ++)
		printf("%.2f\n", discards[i]);
	printf("\nprefetch:\n");
	for (int i = 0; i < 24; i ++)
		printf("%.2f\n", prefetches[i]);
	return 0;
}
