#include <cuda.h>

void CudaInfo();
void CudaErrorInfo(const char* msg);
void CudaErrorInfo(const char* msg, cudaError_t error);

int SharedMemoryPerBlock(int deviceId);