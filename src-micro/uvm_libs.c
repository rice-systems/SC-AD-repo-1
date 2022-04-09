#include <errno.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <limits.h>
#include <stdlib.h>
#include <fcntl.h>
#include <dirent.h>
#include <time.h>
#include "uvm_libs.h"

static double discardNsec = 0;
int debug_discard = 0;

static double delta, totalMem = 0;
static int uvmFd = -1;
#define PSF_DIR "/proc/self/fd"
#define NVIDIA_UVM_PATH "/dev/nvidia-uvm"

double getTime() 
{
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    double timeInNsec = (double) time.tv_sec * 1e9 +
                       (double) time.tv_nsec;
    return timeInNsec;
}

int initUvmFd()
{
    DIR *d;
    struct dirent *dir;

    char psf_path[300];
    char *psf_realpath;

    d = opendir(PSF_DIR);
    if (d)
    {
        while ((dir = readdir(d)) != NULL)
        {
            if (dir->d_type == DT_LNK)
            {
                sprintf(psf_path, "%s/%s", PSF_DIR, dir->d_name);
                psf_realpath = realpath(psf_path, NULL);
                if (strcmp(psf_realpath, NVIDIA_UVM_PATH) == 0)
                {
                    uvmFd = atoi(dir->d_name);
                    // printf("found uvmFd: %s, %s, %d\n", psf_path, psf_realpath, uvmFd);
                }
                free(psf_realpath);
                if (uvmFd >= 0)
                    break;
            }
        }
        closedir(d);
    }

    if (uvmFd < 0)
    {
        printf("We failed to find the nvidia-uvm fd\n");
        return -1;
    }
    // printf("We have stolen the nvidia-uvm fd %d\n", uvmFd);
    return uvmFd;
}

int endDiscard()
{
    totalMem = totalMem / 1024 / 1024 / 1024;
    discardNsec /= 1e9;
    printf("Discard: Mem %.2fGB, latency %.2f, trpt %.2fGB/s\n",
        totalMem, discardNsec, totalMem / discardNsec);
    return 0;
}

int callUvmLinuxIoctl(unsigned long request, void *params)
{
    int ret;

    delta = getTime();
    if (uvmFd == -1)
        initUvmFd();
    while (1) {
        ret = ioctl(uvmFd, request, params);
        if (ret < 0 && (errno == EINTR || errno == EAGAIN)) {
            // Retry on EINTR/EAGAIN
            // (void)cuosAtomicFetchAndIncrementRelaxed64(&uvmLinuxIoctlRetryCountErrno);
            continue;
        }
        break;
    }
    delta = getTime() - delta;
    discardNsec += delta;

    return ret;
}

struct bufData {
  void* start;
  uint64_t length;
  int lazy;
};
typedef struct bufData bufData;

void CUDART_CB discardCB(cudaStream_t stream, cudaError_t status, void *data)
{
    UvmDiscard(((bufData*) data)->start, ((bufData*) data)->length, ((bufData*) data)->lazy);
    free((bufData*) data);
}

void UvmDiscardAsync(void *base, NvLength length, int lazy, cudaStream_t x)
{
    bufData *data = (bufData *) malloc(sizeof(bufData));
    data->start = base;
    data->length = length;
    data->lazy = lazy;
    cudaStreamAddCallback(x, discardCB, data, 0);
}

void CUDART_CB revertCB(cudaStream_t stream, cudaError_t status, void *data)
{
    UvmRevert(((bufData*) data)->start, ((bufData*) data)->length);
    free((bufData*) data);
}

void UvmRevertAsync(void *base, NvLength length, cudaStream_t x)
{
    bufData *data = (bufData *) malloc(sizeof(bufData));
    data->start = base;
    data->length = length;
    cudaStreamAddCallback(x, revertCB, data, 0);
}

void UvmGetStats(size_t *cpu_to_gpu, size_t *gpu_to_cpu, size_t *gpu_faults)
{
    UvmGetMemTransferStats_PARAMS params;
    memset(&params, 0 , sizeof(params));
    params.cpu_to_gpu = (NvU64 *) cpu_to_gpu;
    params.gpu_to_cpu = (NvU64 *) gpu_to_cpu;
    params.gpu_faults = (NvU64 *) gpu_faults;
    callUvmLinuxIoctl(GETSTATS, &params);
}

void UvmRevert(void *base, NvLength length)
{
    UVM_DISCARD_PARAMS params;

    memset(&params, 0, sizeof(params));
    params.base          = (uintptr_t)base;
    params.length        = length;

    // roundup base
    uintptr_t tail = params.base & 0x7ff;
    if (tail != 0) {
        if (params.length <= (4096 - tail))
            return ;
        params.base += 4096 - tail;
        params.length -= 4096 - tail;
    }
    // rounddown lenth
    if ((params.length & 0x7ff) != 0)
        params.length = params.length >> 12 << 12;

    if (params.length == 0)
        return ;

    totalMem += params.length;
    callUvmLinuxIoctl(REVERT, &params);
}

void UvmProbe()
{
    size_t *c, *g, *f;
    UvmGetStats(c,g,f);
}

NV_STATUS UvmDiscard(void *base, NvLength length, int lazy)
{
    UVM_DISCARD_PARAMS params;

    memset(&params, 0, sizeof(params));

    params.base          = (uintptr_t)base;
    params.length        = length;

    // roundup base
    uintptr_t tail = params.base & 0x7ff;
    if (tail != 0) {
        if (params.length <= (4096 - tail))
            return -1;
        params.base += 4096 - tail;
        params.length -= 4096 - tail;
    }
    // rounddown lenth
    if ((params.length & 0x7ff) != 0)
        params.length = params.length >> 12 << 12;

    if (params.length == 0)
        return -1;

    if (debug_discard)
        printf("calling discard with base %p, length %llu\n", base, length);

    totalMem += params.length;
    if (lazy)
        callUvmLinuxIoctl(DISCARDLAZY, &params);
    else
        callUvmLinuxIoctl(DISCARD, &params);

    return params.rmStatus;
}