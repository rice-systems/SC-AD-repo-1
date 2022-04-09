#include "../uvmdiscard/uvmdiscard.h"
#include <cuda_runtime.h>

int main()
{
	float *x;
	cudaMallocManaged(&x, 1024, cudaMemAttachGlobal);
	UvmProbe();
	cudaFree(x);
	return 0;
}