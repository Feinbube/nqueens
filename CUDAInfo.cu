/*
Compute Capability	1,0	1,1	1,2	1,3
				
Threads / Warp	32	32	32	32
Warps / Multiprocessor	24	24	32	32
Threads / Multiprocessor	768	768	1024	1024
Thread Blocks / Multiprocessor	8	8	8	8
Shared Memory / Multiprocessor (bytes)	16384	16384	16384	16384
Register File Size	8192	8192	16384	16384
Register Allocation Unit Size	256	256	512	512
				
				
Warp allocation granularity (for registers)	2	2	2	2
*/

#include "CUDAInfo.h"
#include <stdio.h>
#include <stdlib.h>

cudaDeviceProp deviceProperties [10];
int deviceCount = -1;

void init()
{
	if(deviceCount >= 0)
		return;

	deviceCount = 0;
	cudaError_t err = cudaGetDeviceCount(&deviceCount);
	CudaErrorInfo("cudaGetDeviceCount", err);
	
	for( int i=0; i<deviceCount; i++)
		cudaGetDeviceProperties(&deviceProperties[i], i) ;
}

int SharedMemoryPerBlock(int deviceId)
{
	init();

	return deviceProperties[deviceId].sharedMemPerBlock;	
}

int sharedMemoryPerMultiprocessor(int major, int minor)
{
	return 16384;
}

int threadBlocksPerMultiprocessor(int major, int minor)
{
	return 8;
}

int threadsPerWarp(int major, int minor)
{
	return 32;
}

void CudaInfo()
{
	printf("\nCUDA Information\n");
	printf("---------------------------------------\n");

	init();
	
	for( int i=0; i<deviceCount; i++)
	{
		cudaDeviceProp properties = deviceProperties[i];

		printf("Device(%d):       %s\n", i, properties.name);
		printf("  Compute Cap:   %d.%d\n", properties.major, properties.minor);
		printf("\n");
		printf("Total Memory:\n");
		printf("  Global:        %lu MB\n", properties.totalGlobalMem/1024/1024);
		printf("  Constant:      %lu KB\n", properties.totalConstMem/1024);
		printf("\n");
		printf("Multiprocessors: %d @ %d Mhz\n", properties.multiProcessorCount, properties.clockRate/1000);
		printf("  Blocks:        %d\n", threadBlocksPerMultiprocessor(properties.major, properties.minor));
		printf("  Warps:         %d\n", properties.warpSize);
		printf("  MaxThreads:    %d\n", properties.warpSize * threadsPerWarp(properties.major, properties.minor));
		printf("  Shared Memory: %d KB\n", sharedMemoryPerMultiprocessor(properties.major, properties.minor)/1024);
		printf("\n");
		printf("Per Block:\n");
		printf("  MaxThreads:    %d\n", properties.maxThreadsPerBlock);
		printf("  Registers:     %d\n", properties.regsPerBlock);
		printf("  Shared Memory: %lu KB\n", properties.sharedMemPerBlock/1024);

		printf("---------------------------------------\n");
	}
	printf("\n");
}

void CudaErrorInfo(const char* msg)
{
	cudaError_t err = cudaGetLastError();
	if(cudaSuccess != err)
	{
		fprintf(stderr, "CUDA error [%s]: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void CudaErrorInfo(const char* msg, cudaError_t err)
{
	if(cudaSuccess != err)
	{
		fprintf(stderr, "CUDA error [%s]: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}
