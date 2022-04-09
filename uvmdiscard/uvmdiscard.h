#ifndef UVM_DISCARD
#define UVM_DISCARD

#include <stdint.h>
#include <string.h>
#include <sys/ioctl.h>
#include "cuda_runtime.h"

#define DISCARD                     75
#define GETSTATS                    76
#define DISCARDLAZY                 100
#define REVERT                      101
#define UVM_ENABLE_READ_DUPLICATION 44
#define UVM_INITIALIZE              0x30000001

typedef unsigned long long NvU64;
typedef unsigned int       NvU32;
typedef NvU32           NV_STATUS;
typedef NvU64           NvLength;
#define NV_ALIGN_BYTES(size) __attribute__ ((aligned (size)))

typedef struct
{
    NvU64     flags     NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                    // OUT
} UVM_INITIALIZE_PARAMS;

typedef struct
{
    NvU64 base          NV_ALIGN_BYTES(8);
    NvU64 length        NV_ALIGN_BYTES(8);
    NV_STATUS rmStatus;
} UVM_DISCARD_PARAMS;

typedef struct
{
    NvU64          *cpu_to_gpu;
    NvU64          *gpu_to_cpu;
    NvU64          *gpu_faults;
    NV_STATUS       rmStatus;                          // OUT
} UvmGetMemTransferStats_PARAMS;

typedef struct
{
    NvU64     requestedBase NV_ALIGN_BYTES(8); // IN
    NvU64     length        NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                        // OUT
} UVM_ENABLE_READ_DUPLICATION_PARAMS;

extern int debug_discard;
double getTime();
int initUvmFd();
int endDiscard();
int callUvmLinuxIoctl(unsigned long request, void *params);
NV_STATUS UvmDiscard(void *base, NvLength length, int lazy);
void UvmDiscardAsync(void *base, NvLength length, int lazy, cudaStream_t x);
void UvmRevert(void *base, NvLength length);
void UvmRevertAsync(void *base, NvLength length, cudaStream_t x);
void UvmGetStats(size_t *cpu_to_gpu, size_t *gpu_to_cpu, size_t *gpu_faults);
void UvmProbe();
#endif