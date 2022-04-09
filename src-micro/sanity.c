#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <sys/resource.h>
#include <limits.h>

#include <cuda.h>
#include <inc/uvm_discard.h>

using namespace std;

#define GB 1024UL*1024*1024*sizeof(uint8_t)
#define MB 1024UL*1024*sizeof(uint8_t)

uint64_t sum(uint8_t *buf, size_t bufSize)
{
	uint64_t sum;
	for (uint64_t i = 0; i < bufSize; i ++)
		sum += buf[i];
	return sum;
}

int main() {
	uint8_t *buffer;
	size_t bufferSize = MB * 8;

	cudaError_t error = cudaMallocManaged(&buffer, bufferSize);
	printf("cuda malloc managed exited with 0x%x\n", error);
	memset(buffer, 1, bufferSize);
	printf("sum after memset 1 on CPU: %lu\n", sum(buffer, bufferSize));

	cudaMemPrefetchAsync(buffer, bufferSize, 0, NULL);
	cudaMemset(buffer, 0, bufferSize);
	printf("zero buffer on GPU\n");

	cudaMemPrefetchAsync(buffer, bufferSize, cudaCpuDeviceId, NULL);
	cudaDeviceSynchronize();
	printf("sum after fetching back to CPU: %lu\n", sum(buffer, bufferSize));

	memset(buffer, 0, bufferSize);
	printf("sum after memset 0 on CPU: %lu\n", sum(buffer, bufferSize));
	cudaMemPrefetchAsync(buffer, bufferSize, 0, NULL);
	cudaMemset(buffer, 1, bufferSize);

	cudaDeviceSynchronize();
	printf("discard nv status code %u\n", UvmDiscard(buffer, bufferSize, 0));
	cudaMemPrefetchAsync(buffer, bufferSize, cudaCpuDeviceId, NULL);
	cudaDeviceSynchronize();
	printf("sum after migrating to GPU, memset 1, discarding and migrating back to CPU: %lu\n", sum(buffer, bufferSize));
	printf("if you do not see 0s, sanity check of UvmDiscard failed\n");

	error = cudaFree(buffer);
	printf("cuda Free exited with 0x%x\n", error);
	return 0;
}