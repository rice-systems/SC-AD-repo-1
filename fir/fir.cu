#include "fir.h"
#include "../uvmdiscard/uvmdiscard.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <cuda_runtime.h>

int lazy = 0;
int discard = 0;
size_t bufSize;
uint32_t num_tap_ = 16;
uint64_t num_total_data_ = 0;
uint64_t num_data_per_block_ = 0;
uint64_t num_block_ = 0;

float *input_buffer_ = NULL;
float *output_buffer_ = NULL;
float *coeff_ = NULL;
float *cpu_output = NULL;
float *coeff_buffer_ = NULL;
float *history_buffer_ = NULL;
float *history_ = NULL;

cudaStream_t s0, s1;
cudaEvent_t e0, e1;
size_t fetch_cur = 0, fetch_next = 0, window, windowSize;

__global__ void fir_cuda(float *input, float *output, float *coeff,
												 float *history, uint32_t num_tap, uint32_t num_data) {
	uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid > num_data) return;

	float sum = 0;
	uint32_t i = 0;
	for (i = 0; i < num_tap; i++) {
		if (tid >= i) {
			sum = sum + coeff[i] * input[tid - i];
		} else {
			sum = sum + coeff[i] * history[num_tap - (i - tid)];
		}
	}
	output[tid] = sum;
}

void init_cuda_buffers() {
	cudaStreamCreate(&s0);
	cudaStreamCreate(&s1);
	cudaEventCreate(&e0);
	cudaEventCreate(&e1);

	history_ = reinterpret_cast<float *>(malloc(num_tap_ * sizeof(float)));
	cudaMallocManaged(&input_buffer_, sizeof(float) * num_data_per_block_ * num_block_);
	// directly populate it on CPU
	cudaMemPrefetchAsync(input_buffer_, sizeof(float) * num_data_per_block_ * num_block_, cudaCpuDeviceId, s0); // s1
	// printf("Input buffer starts at %p\n", input_buffer_);
	cudaMallocManaged(&output_buffer_, sizeof(float) * num_data_per_block_ * num_block_);
	// directly populate it on CPU
	// cudaMemPrefetchAsync(output_buffer_, sizeof(float) * num_data_per_block_ * num_block_, cudaCpuDeviceId, s0); // s2
	cudaMalloc(&coeff_buffer_, sizeof(float) * num_tap_);
	cudaMalloc(&history_buffer_, sizeof(float) * num_tap_);
	printf("Memory consumption %lu MB\n", 
		sizeof(float) * num_data_per_block_ * num_block_ * 2 / 1024 / 1024);
}

void generate_input() {
	num_total_data_ = num_data_per_block_ * num_block_;
	coeff_ = new float[num_tap_];

	unsigned int seed = 7;

	// Initialize input data
	for (uint64_t i = 0; i < num_total_data_; i++) {
		// input_[i] = i;
		input_buffer_[i] =
			static_cast<float>(rand_r(&seed)) / static_cast<float>(RAND_MAX);
	}

	// Initialize coefficient
	for (unsigned int i = 0; i < num_tap_; i++) {
		// coeff_[i] = i;
		coeff_[i] =
			static_cast<float>(rand_r(&seed)) / static_cast<float>(RAND_MAX);
	}
}

void push_cuda_buffers() {
	for (unsigned i = 0; i < num_tap_; i++) {
		history_[i] = 0.0;
	}

	cudaMemcpy(history_buffer_, history_, num_tap_ * sizeof(float),
						 cudaMemcpyHostToDevice);
	cudaMemcpy(coeff_buffer_, coeff_, num_tap_ * sizeof(float),
						 cudaMemcpyHostToDevice);
}

void init() {
	init_cuda_buffers();
	generate_input();
	push_cuda_buffers();
}

void run() {
	uint64_t count = 0;
	size_t windowSize;
	double time;
	dim3 grid_size(num_data_per_block_ / 64);
	dim3 block_size(64);

	// use a window size of xxKB
	window = bufSize / num_data_per_block_ / sizeof(float);
	if (num_block_ < window)
		fetch_next = num_block_;
	else
		fetch_next = window;
	fetch_cur = 0;

	// start timer
	cudaDeviceSynchronize();
	UvmProbe();
	time = getTime();

	// s0 = s1;

	cudaMemPrefetchAsync(input_buffer_, sizeof(float) * num_data_per_block_ * fetch_next, 0, s0);
	cudaMemPrefetchAsync(output_buffer_, sizeof(float) * num_data_per_block_ * fetch_next, 0, s0);
	cudaEventRecord(e0, s0);
	cudaEventSynchronize(e0);

	while (count < num_block_) {
		if (count == fetch_next)
		{
			// We have launched a batch of GPU kernels for fetch_cur -> fetch_next
			if (fetch_cur >= window) {
				if (discard){
					UvmDiscardAsync(input_buffer_ + ((fetch_cur - window) * num_data_per_block_), 
						sizeof(float) * num_data_per_block_ * window,
						lazy,
						s1);
				}
			}
			// fetch_cur -> fetch_next has been fetched,
			// fetch next window
			fetch_cur = fetch_next;
			fetch_next += window;
			if (fetch_next > num_block_)
				fetch_next = num_block_;
			windowSize = fetch_next - fetch_cur;

			// record computing current batch
			cudaEventRecord(e1, s1);

			// Issue prefetching command for the next batch
			// This overlaps with the previously launched kernels
			cudaMemPrefetchAsync(input_buffer_ + (fetch_cur * num_data_per_block_), sizeof(float) * num_data_per_block_ * windowSize, 0, s0);
			cudaMemPrefetchAsync(output_buffer_ + (fetch_cur * num_data_per_block_), sizeof(float) * num_data_per_block_ * windowSize, 0, s0);

			// record prefetching next batch
			cudaEventRecord(e0, s0);

			// wait for prefetching next batch
			cudaEventSynchronize(e0);

			// wait for computing current batch
			cudaEventSynchronize(e1);
		}

		fir_cuda<<<grid_size, block_size, 0, s1>>>(input_buffer_ + (count * num_data_per_block_), 
												   output_buffer_ + (count * num_data_per_block_),
												   coeff_buffer_, history_buffer_,
												   num_tap_, num_data_per_block_);

		// Using managed buffer makes programming easier
		history_buffer_ = input_buffer_ + count * num_data_per_block_ + num_data_per_block_ - num_tap_;
		count++;
	}

	// fetch everything back to CPU
	// cudaMemPrefetchAsync(output_buffer_, sizeof(float) * num_data_per_block_ * num_block_, 
	// 										 cudaCpuDeviceId, s0);
	cudaDeviceSynchronize();

	// end timer
	time = getTime() - time;
	printf("Runtime: %.2f ms\n", time / 1e6);
	UvmProbe();
	
    // while(1);
}


static void parse_input(int argc, const char **argv) {
    int window_;
    if (argc == 6)
    {
        num_data_per_block_ = atoi(argv[1]);
        num_block_ = atoi(argv[2]);
        window_ = atoi(argv[3]);
        discard = atoi(argv[4]);
        lazy = atoi(argv[5]);
    }
    else
    {
        num_data_per_block_ = 1024;
        num_block_ = 1024;
    }

    bufSize = window_ * 1024; 
    if (num_data_per_block_ * sizeof(float) > bufSize)
        bufSize = num_data_per_block_ * sizeof(float);
    printf("Configuration: block size %lu MB, block # %lu, discard %d, lazy %d, windowSize %lu MB\n",
        num_data_per_block_ * sizeof(float) / 1024 / 1024, num_block_, discard, lazy, bufSize / 1024 / 1024);
}

int main(int argc, const char **argv) {
    parse_input(argc, argv);
    init();
    run();
    return 0;
}
