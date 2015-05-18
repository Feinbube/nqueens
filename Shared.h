#define CUDA

#ifdef CUDA
#include <cuda.h>
#include "CUDAInfo.h"
#endif

#ifdef WIN32
#define snprintf sprintf_s
#endif

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef unsigned long long ULL;

#define MAX_BOARDSIZE 32
#define MIN_BOARDSIZE 2

#define THREADS_PER_BLOCK 16
#define DEPTH 4