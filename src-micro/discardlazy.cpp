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

void getstats()
{
	size_t cpu_to_gpu, gpu_to_cpu, gpu_faults;
	UvmGetStats(&cpu_to_gpu, &gpu_to_cpu, &gpu_faults);
}

int main() {

	uint8_t *bigBuffer, *A;
	size_t bigSize = GB * 14, smallSize = GB * 2;
	double time;
	cudaStream_t s;
	cudaStreamCreate(&s);

	for (int lazy = 0; lazy < 2; lazy ++)
	{
		printf("Discard lazy? %d\n", lazy);
		cudaMallocManaged(&bigBuffer, bigSize);
		cudaMallocManaged(&A, smallSize);

		cudaMemPrefetchAsync(bigBuffer, bigSize, 0, s);
		cudaDeviceSynchronize();
		getstats();

		// UvmDiscard(bigBuffer + bigSize / 2, smallSize, lazy);
		getstats();

		cudaMemPrefetchAsync(bigBuffer + bigSize / 2, smallSize, 0, s);
		cudaDeviceSynchronize();
		getstats();

		// UvmDiscard(bigBuffer + bigSize / 2, smallSize, lazy);
		getstats();

		cudaMemPrefetchAsync(A, smallSize, 0, s);
		cudaDeviceSynchronize();
		getstats();

		cudaFree(bigBuffer);
		cudaFree(A);
		printf("FINISHED ------------\n\n");
	}
	return 0;
}
