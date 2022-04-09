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

double measure(void *buf1, size_t bufferSize, cudaStream_t s0)
{
	double wall, cpu_to_gpu, gpu_to_cpu;
	wall = getTime();
	cudaMemPrefetchAsync(buf1, bufferSize, 0, s0);
	cudaStreamSynchronize(s0);
	wall = getTime() - wall;
	cpu_to_gpu = wall / 1e9;
	
	wall = getTime();
	cudaMemPrefetchAsync(buf1, bufferSize, cudaCpuDeviceId, s0);
	cudaStreamSynchronize(s0);
	wall = getTime() - wall;
	gpu_to_cpu = wall / 1e9;

	// printf("throughput : %.2f GB/s\n", (double) bufferSize * 2 / GB / (cpu_to_gpu + gpu_to_cpu));
	return bufferSize / gpu_to_cpu / GB;
}

int main() {
	void *buf1, *buf2;
	size_t bufferSizes[24] = {
		KB*1, KB*2, 4*KB, 8*KB, 16*KB, 32*KB, 64*KB, 128*KB, 256*KB, 512*KB, MB*1, 
		MB*2, MB*4, MB*8, MB*16, MB*32, MB*64, MB*128, MB*256, MB*512,
		GB*1, GB*2, GB*4, GB*8}, bufferSize = GB*10;
	double ans, wall;
	cudaStream_t s0;

	cudaStreamCreate(&s0);
	cudaMallocManaged(&buf1, bufferSize, cudaMemAttachGlobal);
	// buf2 = malloc(bufferSize);
	cudaMallocHost(&buf2, bufferSize);
	memset(buf2, 0, bufferSize);
	cudaMemPrefetchAsync(buf1, bufferSize, 0, s0);
	cudaMemPrefetchAsync(buf1, bufferSize, cudaCpuDeviceId, s0);	
	cudaDeviceSynchronize();

	printf("warmup \n");
	ans = measure(buf1, bufferSize, s0);
	printf("warmup throughput 1 %.2f\n", ans);
	ans = measure(buf1, bufferSize, s0);
	printf("warmup throughput 2 %.2f\n", ans);
	for (int i = 0; i < 11; i ++)
	{
		ans = 0;
		ans += measure((char *) buf1 + GB * 2, bufferSizes[i], s0);
		ans += measure((char *) buf1 + GB * 2, bufferSizes[i], s0);
		ans += measure((char *) buf1 + GB * 2, bufferSizes[i], s0);
		printf("%.2f\n", ans / 3);
	}
	for (int i = 11; i < 24; i ++)
	{
		ans = 0;
		ans += measure((char *) buf1, bufferSizes[i], s0);
		ans += measure((char *) buf1, bufferSizes[i], s0);
		ans += measure((char *) buf1, bufferSizes[i], s0);
		printf("%.2f\n", ans / 3);
	}

	cudaMemPrefetchAsync(buf1, bufferSize, 0, s0);	
	cudaDeviceSynchronize();
	for (int i = 0; i < 24; i ++)
	{
		ans = 0;
		wall = getTime();
		cudaMemcpy(buf2, buf1, bufferSizes[i], cudaMemcpyDefault);
		wall = getTime() - wall;

		wall = wall / 1e9;
		printf("cudaMemcpy GPU->CPU : %.2f GB/s\n",
				(double) bufferSizes[i] / GB / wall);
	}

	cudaFree(buf1);
	cudaFreeHost(buf2);
	return 0;
}
