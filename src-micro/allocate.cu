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
#include "cuda_runtime.h"

#define GB 1024UL*1024*1024*sizeof(uint8_t)
#define MB 1024UL*1024*sizeof(uint8_t)

int main(int argc, char* argv[]) {
	uint8_t *buffer;
	size_t bufferSize = 0;

	if (argc == 2)
		bufferSize = MB * atoi(argv[1]);

	printf("allocating %d MB size of GPU memory\n", atoi(argv[1]));
	if (bufferSize > 0)
		cudaError_t error = cudaMalloc(&buffer, bufferSize);
	while(true)
	{
		sleep(60);
	}
	return 0;
}