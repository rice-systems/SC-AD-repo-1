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
cudaStream_t s;

#define NA 20
unsigned long long sizeA[NA];
double totalTime = 0;

void forward(uint8_t *A[], int n)
{
	double time;
	// for (int i = 0; i < n; i ++)
	// 	printf("size is %llu\n", sizeA[i]);
	for (int i = 0; i < n; i ++) {
		time = getTime();
		cudaMemPrefetchAsync(A[i], sizeA[i], 0, s);
		cudaDeviceSynchronize();
		time = getTime() - time;
		totalTime += time;
		printf("prefetching %d, size %luGB: %.2f ms\n", i, sizeA[i] / GB, time / 1e6);
	}
}

void backward(uint8_t *A[], int n, int lazy)
{
	double time;
	for (int i = n - 1; i >= 0; i --) {
		time = getTime();
		cudaMemPrefetchAsync(A[i], sizeA[i], 0, s);
		cudaDeviceSynchronize();
		time = getTime() - time;
		totalTime += time;
		printf("prefetching %d: %.2f ms\n", i, time / 1e6);

		time = getTime();
		UvmDiscardAsync(A[i], sizeA[i], lazy, s);
		cudaDeviceSynchronize();
		time = getTime() - time;
		printf("discarding %d: %.2f ms\n", i, time / 1e6);
	}
}

int main() {

	uint8_t *bigBuffer, *A[NA];
	size_t bigSize = GB * 14, smallSize = GB * 2;
	double time;
	cudaStreamCreate(&s);
	for (int i = 0; i < NA; i ++) {
		sizeA[i] = smallSize; // - i * (smallSize / NA);
		printf("size %d: %lu\n", i, sizeA[NA]);
	}

	for (int lazy = 0; lazy < 2; lazy ++)
	{
		cudaMallocManaged(&bigBuffer, bigSize);
		for (int i = 0; i < NA; i ++)
			cudaMallocManaged(&A[i], sizeA[i]);

		totalTime = 0;
		forward(A, NA);
		backward(A, NA, lazy);
		printf("TOTAL memprefetch time is %.2f\n", totalTime / 1e6);
		cudaFree(bigBuffer);
		for (int i = 0; i < NA; i ++)
			cudaFree(A[i]);
		printf("FINISHED ------------\n\n");
	}
	return 0;
}
