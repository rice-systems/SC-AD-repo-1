#ifdef FIR_H
#define FIR_H

#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <stdlib.h>

int lazy;
int discard;
size_t bufSize;
uint32_t num_tap_;
uint64_t num_total_data_;
uint64_t num_data_per_block_;
uint64_t num_block_;
float *input_buffer_;
float *output_buffer_;
float *coeff_;
float *cpu_output;
float *coeff_buffer_;
float *history_buffer_;
float *history_;

void init();
void run();
void init_cuda_buffers();
void generate_input();
void push_cuda_buffers();

#endif