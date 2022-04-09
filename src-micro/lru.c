#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <sys/resource.h>
#include <limits.h>
#include <unistd.h>
#include <cuda.h>
#include "uvm_libs.h"
#include "cuda_runtime.h"

#define GB 1024UL*1024*1024*sizeof(uint8_t)
#define MB 1024UL*1024*sizeof(uint8_t)

int main(int argc, char* argv[]) {
	void *buffer;
	size_t bufferSize = 2 * GB;
	cudaStream_t s[30];
	for (int i = 0; i < 24; i ++)
		cudaStreamCreate(&s[i]);
	if (argc == 2)
		bufferSize = MB * atoi(argv[1]);

	printf("allocating %lu MB size of UVM memory\n", bufferSize);
	double nanoseconds;
	for (int thread = 1; thread <= 12; thread ++) {
		cudaError_t error = cudaMallocManaged(&buffer, bufferSize, cudaMemAttachGlobal);
		cudaDeviceSynchronize();
		nanoseconds = getTime();
		size_t seg = roundup2(bufferSize / thread, 2*MB);
		for (int i = 0; i < thread - 1; i ++) {
			cudaMemPrefetchAsync(buffer, seg, 0, s[i]);
			buffer = buffer + seg;
		}
		cudaMemPrefetchAsync(buffer, bufferSize - (thread - 1) * seg, 0, s[thread - 1]);
		cudaDeviceSynchronize();
		nanoseconds = getTime() - nanoseconds;
		printf("%d-threaded zeroing takes %.2f ms\n", thread, nanoseconds / 1e6);
		cudaFree(buffer);
	}

	cudaDeviceSynchronize();
	cudaError_t error = cudaMallocManaged(&buffer, bufferSize, cudaMemAttachGlobal);
	cudaMemPrefetchAsync(buffer, bufferSize, 0, s[0]);
	cudaDeviceSynchronize();
	nanoseconds = getTime();
	UvmDiscard(buffer, bufferSize);
	cudaDeviceSynchronize();
	nanoseconds = getTime() - nanoseconds;
	printf("discarding 2GB buffer takes %.2f ms\n", nanoseconds / 1e6);

	nanoseconds = getTime();
	cudaMemPrefetchAsync(buffer, bufferSize, 0, s[0]);
	cudaDeviceSynchronize();
	nanoseconds = getTime() - nanoseconds;
	printf("remapping 2GB buffer takes %.2f ms\n", nanoseconds / 1e6);

	for (int thread = 1; thread <= 12; thread ++) {
		UvmDiscard(buffer, bufferSize);
		cudaDeviceSynchronize();
		nanoseconds = getTime();
		size_t seg = roundup2(bufferSize / thread, 2*MB);
		for (int i = 0; i < thread - 1; i ++) {
			cudaMemPrefetchAsync(buffer, seg, 0, s[i]);
			buffer = buffer + seg;
		}
		cudaMemPrefetchAsync(buffer, bufferSize - (thread - 1) * seg, 0, s[thread - 1]);
		nanoseconds = getTime() - nanoseconds;
		printf("%d-threaded remapping takes %.2f ms\n", thread, nanoseconds / 1e6);
	}
	return 0;
}