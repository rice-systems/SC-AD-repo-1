#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <limits.h>
#include <time.h>

#include <cuda_runtime.h>
#include "nvml.h"
#include "uvm_libs.h"

#define GB (1024UL*1024*1024*sizeof(uint8_t))
#define MB (1024UL*1024*sizeof(uint8_t))

int main() {
	void *buf1;
	size_t bufferSize = GB * 4;
	size_t bufSize[13] = {0};
	double wall;
	cudaStream_t s0, s1;

	for (int i = 0; i < 13; i ++)
		bufSize[i] = (1ULL << i) * MB;
	cudaStreamCreate(&s0);
	cudaStreamCreate(&s1);
	cudaMallocManaged(&buf1, bufferSize, cudaMemAttachGlobal);
	cudaMemPrefetchAsync(buf1, bufferSize, cudaCpuDeviceId, s0);
	cudaMemPrefetchAsync(buf1, bufferSize, 0, s0);
	cudaDeviceSynchronize();

	wall = getTime();
	UvmDiscard(buf1, bufferSize, 1);
	wall = getTime() - wall;

	wall = wall / 1e9;
	printf("Discard: latency : %.2f, throughput : %.2f GB/s\n",
			wall, (double) bufferSize / GB / wall);

	wall = getTime();
	cudaMemPrefetchAsync(buf1, bufferSize, 0, s0);
	cudaDeviceSynchronize();
	wall = getTime() - wall;

	wall = wall / 1e9;
	printf("Prefetch: latency : %.2f, throughput : %.2f GB/s\n",
			wall, (double) bufferSize / GB / wall);



	for (int i = 0; i < 13; i ++) {
		bufferSize = bufSize[i];
		wall = getTime();
		UvmDiscard(buf1, bufferSize, 1);
		cudaMemPrefetchAsync(buf1, bufferSize, 0, s0, 1);
		cudaDeviceSynchronize();
		wall = getTime() - wall;

		wall = wall / 1e9;
		printf("BufSize: %.2f MB, latency : %.2f, throughput : %.2f GB/s\n",
				(double) bufSize[i] / MB, wall, (double) bufferSize / GB / wall);
	}
	return 0;
}