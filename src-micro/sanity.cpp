#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <sys/resource.h>
#include <limits.h>

#include <cuda.h>
#include "uvm_libs.h"

using namespace std;

#define GB 1024UL*1024*1024*sizeof(uint8_t)
#define MB 1024UL*1024*sizeof(uint8_t)

int sum(uint8_t *buf, size_t bufSize)
{
	int sum = 0;
	for (int i = 0; i < bufSize; i ++) {
		sum += buf[i];
	}
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

	uint8_t *bigBuffer, *smallBuffer;
	size_t bigSize = GB * 14, smallSize = GB * 2;
	double time;
	for (int lazy = 0; lazy < 2; lazy ++)
	{
		cudaMallocManaged(&bigBuffer, bigSize);
		cudaMallocManaged(&smallBuffer, smallSize);

		cudaDeviceSynchronize();

		time = getTime();
		cudaMemPrefetchAsync(smallBuffer, smallSize, 0, NULL);
		cudaDeviceSynchronize();
		time = getTime() - time;
		printf("zero 2GB on GPU: %.2f ms\n", time / 1e6);

		time = getTime();
		cudaMemPrefetchAsync(smallBuffer, smallSize, cudaCpuDeviceId, NULL);
		cudaDeviceSynchronize();
		time = getTime() - time;
		printf("move 2GB to CPU: %.2f ms\n", time / 1e6);

		time = getTime();
		cudaMemPrefetchAsync(smallBuffer, smallSize, 0, NULL);
		cudaDeviceSynchronize();
		time = getTime() - time;
		printf("move 2GB to GPU: %.2f ms\n", time / 1e6);

		time = getTime();
		cudaMemPrefetchAsync(smallBuffer, smallSize, cudaCpuDeviceId, NULL);
		cudaDeviceSynchronize();
		time = getTime() - time;
		printf("move 2GB to CPU: %.2f ms\n", time / 1e6);

		cudaMemPrefetchAsync(bigBuffer, bigSize, 0, NULL);
		cudaDeviceSynchronize();

		time = getTime();
		UvmDiscard(bigBuffer + bigSize / 2, smallSize, lazy);
		cudaDeviceSynchronize();
		time = getTime() - time;
		printf("discard 2GB: lazy %d, %.2f ms\n", lazy, time / 1e6);

		time = getTime();
		cudaMemPrefetchAsync(smallBuffer, smallSize, 0, NULL);
		cudaDeviceSynchronize();
		time = getTime() - time;
		printf("prefetch 2GB to GPU: %.2f ms\n", time / 1e6);

		time = getTime();
		cudaMemPrefetchAsync(smallBuffer, smallSize, 0, NULL);
		cudaDeviceSynchronize();
		time = getTime() - time;
		printf("reprefetch 2GB on GPU: %.2f ms\n", time / 1e6);

		time = getTime();
		UvmDiscard(smallBuffer, smallSize, lazy);
		cudaDeviceSynchronize();
		time = getTime() - time;
		printf("discard 2GB itself: lazy %d, %.2f ms\n", lazy, time / 1e6);

		time = getTime();
		cudaMemPrefetchAsync(smallBuffer, smallSize, 0, NULL);
		cudaDeviceSynchronize();
		time = getTime() - time;
		printf("reprefetch 2GB on GPU: %.2f ms\n", time / 1e6);

		cudaFree(bigBuffer);
		cudaFree(smallBuffer);
	}

	cudaMallocManaged(&bigBuffer, bigSize);
	cudaMallocManaged(&smallBuffer, smallSize);
	cudaDeviceSynchronize();
	time = getTime();
	cudaMemPrefetchAsync(smallBuffer, smallSize, 0, NULL);
	cudaDeviceSynchronize();
	time = getTime() - time;
	printf("zero 2GB: %.2f ms\n", time / 1e6);
	time = getTime();
	cudaMemPrefetchAsync(smallBuffer, smallSize, cudaCpuDeviceId, NULL);
	cudaDeviceSynchronize();
	time = getTime() - time;
	printf("move 2GB to CPU: %.2f ms\n", time / 1e6);
	cudaMemPrefetchAsync(bigBuffer, bigSize, 0, NULL);
	cudaDeviceSynchronize();
	time = getTime();
	cudaMemPrefetchAsync(smallBuffer, smallSize, 0, NULL);
	cudaDeviceSynchronize();
	time = getTime() - time;
	printf("oversubscribe 2GB on GPU: %.2f ms\n", time / 1e6);

	return 0;
}
