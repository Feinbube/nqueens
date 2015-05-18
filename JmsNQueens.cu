#include "JmsNQueens.h"

unsigned int guessAllocationSize(int size, int depth)
{
	int result = size;

	for(int i=0; i<depth-1; i++)
		result *= (size-i);

	return result;
}

unsigned int getAllocationSize(int threadCount, int depth)
{
	return (threadCount +1) * depth;
}

int initDataUsingNqueen(int board_size, int depth, int * data, unsigned int dataLength)
{
	int threadCount = 0;

    int aQueenBitRes[MAX_BOARDSIZE]; /* results */
    int aQueenBitCol[MAX_BOARDSIZE]; /* marks colummns which already have queens */
    int aQueenBitPosDiag[MAX_BOARDSIZE]; /* marks "positive diagonals" which already have queens */
    int aQueenBitNegDiag[MAX_BOARDSIZE]; /* marks "negative diagonals" which already have queens */
    int aStack[MAX_BOARDSIZE + 2]; /* we use a stack instead of recursion */
    register int* pnStack;

    register int numrows = 0; /* numrows redundant - could use stack */
    register unsigned int lsb; /* least significant bit */
    register unsigned int bitfield; /* bits which are set mark possible positions for a queen */
    int i;
    int odd = board_size & 1; /* 0 if board_size even, 1 if odd */
    int board_minus = depth; /* board size - 1 */
    int mask = (1 << board_size) - 1; /* if board size is N, mask consists of N 1's */

    /* Initialize stack */
    aStack[0] = -1; /* set sentinel -- signifies end of stack */

    /* NOTE: (board_size & 1) is true iff board_size is odd */
    /* We need to loop through 2x if board_size is odd */
    for (i = 0; i < (1 + odd); ++i)
    {
        /* We don't have to optimize this part; it ain't the
           critical loop */
        bitfield = 0;
        if (0 == i)
        {
            /* Handle half of the board, except the middle
               column. So if the board is 5 x 5, the first
               row will be: 00011, since we're not worrying
               about placing a queen in the center column (yet).
            */
            int half = board_size>>1; /* divide by two */
            /* fill in rightmost 1's in bitfield for half of board_size
               If board_size is 7, half of that is 3 (we're discarding the remainder)
               and bitfield will be set to 111 in binary. */
            bitfield = (1 << half) - 1;
            pnStack = aStack + 1; /* stack pointer */

            aQueenBitRes[0] = 0;
            aQueenBitCol[0] = aQueenBitPosDiag[0] = aQueenBitNegDiag[0] = 0;
        }
        else
        {
            /* Handle the middle column (of a odd-sized board).
               Set middle column bit to 1, then set
               half of next row.
               So we're processing first row (one element) & half of next.
               So if the board is 5 x 5, the first row will be: 00100, and
               the next row will be 00011.
            */
            bitfield = 1 << (board_size >> 1);
            numrows = 1; /* prob. already 0 */

            /* The first row just has one queen (in the middle column).*/
            aQueenBitRes[0] = bitfield;
            aQueenBitCol[0] = aQueenBitPosDiag[0] = aQueenBitNegDiag[0] = 0;
            aQueenBitCol[1] = bitfield;

            /* Now do the next row.  Only set bits in half of it, because we'll
               flip the results over the "Y-axis".  */
            aQueenBitNegDiag[1] = (bitfield >> 1);
            aQueenBitPosDiag[1] = (bitfield << 1);
            pnStack = aStack + 1; /* stack pointer */
            *pnStack++ = 0; /* we're done w/ this row -- only 1 element & we've done it */
            bitfield = (bitfield - 1) >> 1; /* bitfield -1 is all 1's to the left of the single 1 */
        }

        /* this is the critical loop */
        for (;;)
        {
            /* could use
               lsb = bitfield ^ (bitfield & (bitfield -1));
               to get first (least sig) "1" bit, but that's slower. */
            lsb = -((signed)bitfield) & bitfield; /* this assumes a 2's complement architecture */
            if (0 == bitfield)
            {
                bitfield = *--pnStack; /* get prev. bitfield from stack */
                if (pnStack == aStack) { /* if sentinel hit.... */
                    break ;
                }
                --numrows;
                continue;
            }
            bitfield &= ~lsb; /* toggle off this bit so we don't try it again */

            aQueenBitRes[numrows] = lsb; /* save the result */
            if (numrows < board_minus) /* we still have more rows to process? */
            {
                int n = numrows++;
                aQueenBitCol[numrows] = aQueenBitCol[n] | lsb;
                aQueenBitNegDiag[numrows] = (aQueenBitNegDiag[n] | lsb) >> 1;
                aQueenBitPosDiag[numrows] = (aQueenBitPosDiag[n] | lsb) << 1;
                *pnStack++ = bitfield;
                /* We can't consider positions for the queen which are in the same
                   column, same positive diagonal, or same negative diagonal as another
                   queen already on the board. */
                bitfield = mask & ~(aQueenBitCol[numrows] | aQueenBitNegDiag[numrows] | aQueenBitPosDiag[numrows]);
                continue;
            }
            else
            {
                /* We have no more rows to process; we found a solution. */
                /* Comment out the call to printtable in order to print the solutions as board position*/
                /* printtable(board_size, aQueenBitRes, solutionCount + 1);  */
				for(int i=0; i<depth; i++)
				{
					unsigned int adress = threadCount*depth + i;
					if(adress > dataLength)
					{
						printf("Internal Error. Arraw Overflow. Increase dataLength!");
						exit(1);
					}

					data[adress] = aQueenBitRes[i] ^ (aQueenBitRes[i] & (aQueenBitRes[i] - 1));
				}

				++threadCount;

                bitfield = *--pnStack;
                --numrows;
                continue;
            }
        }
    }

	return threadCount;
}

int* mallocInt(int size)
{
	int * result = (int*)malloc(size * sizeof(int));
	if(result == 0)
	{
		printf("Error allocating memory. (Size: %d)\n", size*sizeof(int));
		perror("Error: ");
		exit(EXIT_FAILURE);
	}

	return result;
}

int * initData(int * threadCount, int size, int depth)
{
	unsigned int allocationSize = guessAllocationSize(size, depth);

	int * data = mallocInt(allocationSize);
	*threadCount = initDataUsingNqueen(size, depth, data, allocationSize);

	allocationSize = getAllocationSize(*threadCount, depth);

	int * result = mallocInt(allocationSize);
	for(unsigned int i=0; i<allocationSize; i++)
		result[i] = data[i];

	free(data);

	return result;
}

/* The function which calculates the N queen solutions.
   We calculate one-half the solutions, then flip the results over
   the "Y axis" of the board.  Every solution can be reflected that
   way to generate another unique solution (assuming the board size
   isn't 1 x 1).  That's because a solution cannot be symmetrical
   across the Y-axis (because you can't have two queens in the same
   horizontal row).  A solution also cannot consist of queens
   down the middle column of a board with an odd number of columns,
   since you can't have two queens in the same vertical row.

   This is a backtracking algorithm.  We place a queen in the top
   row, then note the column and diagonals it occupies.  We then
   place a queen in the next row down, taking care not to place it
   in the same column or diagonal.  We then update the occupied
   columns & diagonals & move on to the next row.  If no position
   is open in the next row, we back track to the previous row & move
   the queen over to the next available spot in its row & the process
   starts over again.
*/

#ifdef CUDA
__global__
void NqueenFromDepthCUDA(int board_size, int threadCount, int depth, int * data)
{
	extern __shared__ int sharedData[]; // lenght = MAX_BOARDSIZE * 4 * threadCount

	int tx = threadIdx.x;

	int threadId = blockIdx.x * blockDim.x + tx;
	if(threadId >= threadCount || threadId < 0)
		return;

	int abs = board_size - depth;
	int iQueenBitCol = abs * 4 * tx + 0 * abs;
	int iQueenBitPosDiag = abs * 4 * tx + 1 * abs;
	int iQueenBitNegDiag = abs * 4 * tx + 2 * abs;
	int iStack = abs * 4 * tx + 3 * abs;

	ULL solutionCount = 0;

	register int nStack = iStack;
    register int numrows = 0; /* numrows redundant - could use stack */

    register unsigned int lsb; /* least significant bit */
    register unsigned int bitfield; /* bits which are set mark possible positions for a queen */
    int board_minus = board_size - 1; /* board size - 1 */
    register int mask = (1 << board_size) - 1; /* if board size is N, mask consists of N 1's */

    /* Initialize stack */
    sharedData[nStack] = -1; /* set sentinel -- signifies end of stack */

    /* We don't have to optimize this part; it ain't the
       critical loop */
    bitfield = 0;

    /* Handle half of the board, except the middle
       column. So if the board is 5 x 5, the first
       row will be: 00011, since we're not worrying
       about placing a queen in the center column (yet).
    */
    int half = board_size>>1; /* divide by two */
    /* fill in rightmost 1's in bitfield for half of board_size
       If board_size is 7, half of that is 3 (we're discarding the remainder)
       and bitfield will be set to 111 in binary. */
    bitfield = (1 << half) - 1;
    nStack += 1; /* stack pointer */

    sharedData[iQueenBitCol] = sharedData[iQueenBitPosDiag] = sharedData[iQueenBitNegDiag] = 0;

	for(int d = 0; d < depth; d++)
	{
		lsb = data[threadId * depth + d];
		bitfield &= ~lsb;

        sharedData[iQueenBitCol] = sharedData[iQueenBitCol] | lsb;
		sharedData[iQueenBitPosDiag] = (sharedData[iQueenBitPosDiag] | lsb) << 1;
        sharedData[iQueenBitNegDiag] = (sharedData[iQueenBitNegDiag] | lsb) >> 1;
        sharedData[nStack] = bitfield;
        /* We can't consider positions for the queen which are in the same
           column, same positive diagonal, or same negative diagonal as another
           queen already on the board. */
        bitfield = mask & ~(sharedData[iQueenBitCol] | sharedData[iQueenBitPosDiag] | sharedData[iQueenBitNegDiag]);
	}

    /* this is the critical loop */
    for (;;)
    {
        /* could use
           lsb = bitfield ^ (bitfield & (bitfield -1));
           to get first (least sig) "1" bit, but that's slower. */
        lsb = -((signed)bitfield) & bitfield; /* this assumes a 2's complement architecture */
        if (0 == bitfield)
        {
            bitfield = sharedData[--nStack]; /* get prev. bitfield from stack */
            if (nStack == iStack) { /* if sentinel hit.... */
                break ;
            }
            --numrows;
            continue;
        }
        bitfield &= ~lsb; /* toggle off this bit so we don't try it again */

        //aQueenBitRes[numrows] = lsb; /* save the result */
        if (numrows < board_minus - depth) /* we still have more rows to process? */
        {
			int n = numrows++;
            sharedData[iQueenBitCol + numrows] = sharedData[iQueenBitCol + n] | lsb;
			sharedData[iQueenBitPosDiag + numrows] = (sharedData[iQueenBitPosDiag + n] | lsb) << 1;
			sharedData[iQueenBitNegDiag + numrows] = (sharedData[iQueenBitNegDiag + n] | lsb) >> 1;
            sharedData[nStack++] = bitfield;
            /* We can't consider positions for the queen which are in the same
               column, same positive diagonal, or same negative diagonal as another
               queen already on the board. */
            bitfield = mask & ~(sharedData[iQueenBitCol + numrows] | sharedData[iQueenBitNegDiag + numrows] | sharedData[iQueenBitPosDiag + numrows]);
            continue;
        }
        else
        {
            /* We have no more rows to process; we found a solution. */
            /* Comment out the call to printtable in order to print the solutions as board position*/
            /* printtable(board_size, aQueenBitRes, solutionCount + 1);  */
            ++solutionCount;
            bitfield = sharedData[--nStack];
            --numrows;
            continue;
        }
    }

    /* multiply solutions by two, to count mirror images */
	data[threadId * depth] =  solutionCount * 2;
}
#endif

void NqueenFromDepth(int board_size, int threadId, int depth, int * data)
{
	ULL solutionCount = 0;
    //int aQueenBitRes[MAX_BOARDSIZE]; /* results */
    int aQueenBitCol[MAX_BOARDSIZE]; /* marks colummns which already have queens */
    int aQueenBitPosDiag[MAX_BOARDSIZE]; /* marks "positive diagonals" which already have queens */
    int aQueenBitNegDiag[MAX_BOARDSIZE]; /* marks "negative diagonals" which already have queens */
    int aStack[MAX_BOARDSIZE + 2]; /* we use a stack instead of recursion */
    register int* pnStack;

    register int numrows = 0; /* numrows redundant - could use stack */
    register unsigned int lsb; /* least significant bit */
    register unsigned int bitfield; /* bits which are set mark possible positions for a queen */
    int board_minus = board_size - 1; /* board size - 1 */
    int mask = (1 << board_size) - 1; /* if board size is N, mask consists of N 1's */

    /* Initialize stack */
    aStack[0] = -1; /* set sentinel -- signifies end of stack */

    /* We don't have to optimize this part; it ain't the
       critical loop */
    bitfield = 0;

    /* Handle half of the board, except the middle
       column. So if the board is 5 x 5, the first
       row will be: 00011, since we're not worrying
       about placing a queen in the center column (yet).
    */
    int half = board_size>>1; /* divide by two */
    /* fill in rightmost 1's in bitfield for half of board_size
       If board_size is 7, half of that is 3 (we're discarding the remainder)
       and bitfield will be set to 111 in binary. */
    bitfield = (1 << half) - 1;
    pnStack = aStack + 1; /* stack pointer */

//    aQueenBitRes[0] = 0;
    aQueenBitCol[0] = aQueenBitPosDiag[0] = aQueenBitNegDiag[0] = 0;

	for(;numrows < depth;)
	{
		lsb = data[threadId * depth + numrows];
		bitfield &= ~lsb;
		//aQueenBitRes[numrows] = lsb;

		int n = numrows++;
        aQueenBitCol[numrows] = aQueenBitCol[n] | lsb;
        aQueenBitNegDiag[numrows] = (aQueenBitNegDiag[n] | lsb) >> 1;
        aQueenBitPosDiag[numrows] = (aQueenBitPosDiag[n] | lsb) << 1;
        *pnStack++ = bitfield;
        /* We can't consider positions for the queen which are in the same
           column, same positive diagonal, or same negative diagonal as another
           queen already on the board. */
        bitfield = mask & ~(aQueenBitCol[numrows] | aQueenBitNegDiag[numrows] | aQueenBitPosDiag[numrows]);
	}

    /* this is the critical loop */
    for (;;)
    {
        /* could use
           lsb = bitfield ^ (bitfield & (bitfield -1));
           to get first (least sig) "1" bit, but that's slower. */
        lsb = -((signed)bitfield) & bitfield; /* this assumes a 2's complement architecture */
        if (0 == bitfield)
        {
            bitfield = *--pnStack; /* get prev. bitfield from stack */
            if (pnStack == aStack + depth) { /* if sentinel hit.... */
                break ;
            }
            --numrows;
            continue;
        }
        bitfield &= ~lsb; /* toggle off this bit so we don't try it again */

        //aQueenBitRes[numrows] = lsb; /* save the result */
        if (numrows < board_minus) /* we still have more rows to process? */
        {
            int n = numrows++;
            aQueenBitCol[numrows] = aQueenBitCol[n] | lsb;
            aQueenBitNegDiag[numrows] = (aQueenBitNegDiag[n] | lsb) >> 1;
            aQueenBitPosDiag[numrows] = (aQueenBitPosDiag[n] | lsb) << 1;
            *pnStack++ = bitfield;
            /* We can't consider positions for the queen which are in the same
               column, same positive diagonal, or same negative diagonal as another
               queen already on the board. */
            bitfield = mask & ~(aQueenBitCol[numrows] | aQueenBitNegDiag[numrows] | aQueenBitPosDiag[numrows]);
            continue;
        }
        else
        {
            /* We have no more rows to process; we found a solution. */
            /* Comment out the call to printtable in order to print the solutions as board position*/
            /* printtable(board_size, aQueenBitRes, solutionCount + 1);  */
            ++solutionCount;
            bitfield = *--pnStack;
            --numrows;
            continue;
        }
    }

    /* multiply solutions by two, to count mirror images */
	ULL * pdata = (ULL*)&data[threadId * depth];
	*pdata =  solutionCount * 2;
}

int sharedMemoryNeeded(int boardsize, int threadsPerBlock)
{
	return boardsize * 4 * threadsPerBlock * sizeof(int) // memory for 4 int arrays
		+ 2 * threadsPerBlock * sizeof(int); // stack array is 2 columns bigger
}

ULL calculateSolutions(int gpuIndex, int boardsize, int depth, int threadsPerBlock, bool verbose, bool cpuOnly)
{
	int threadCount = 0;
	int * data = initData(&threadCount, boardsize, depth);

	int blockSize = threadCount/threadsPerBlock+1;
	if(verbose)
	{
		printf("\nParameters: Depth(%d) Threads(%d) Blocks(%d)\n", depth, threadCount, blockSize);
		printf("SIZE: %d / %d\n", getAllocationSize(threadCount, depth), guessAllocationSize(boardsize, depth));
	}

	ULL solutionCount = 0;

#ifdef CUDA
	if(!cpuOnly)
	{
		// create events for time measurement
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaError_t err = cudaSetDevice(gpuIndex);
		CudaErrorInfo("cudaSetDevice", err);

		int mem_size = getAllocationSize(threadCount, depth) * sizeof(int);
		if(verbose) printf("mem_size = %d Bytes\n", mem_size);

		int *data_gpu;
		err = cudaMalloc((void**)&data_gpu, mem_size);
		CudaErrorInfo("cudaMalloc", err);

		err = cudaMemcpy(data_gpu, data, mem_size, cudaMemcpyHostToDevice);
		CudaErrorInfo("cudaMemcpy source->gpu", err);

		int sharedMemorySize = sharedMemoryNeeded(boardsize-depth, threadsPerBlock);

		if(sharedMemorySize + 40 > SharedMemoryPerBlock(0))
		{
			printf("SharedMemorySize(%d) is bigger than available shared memory per block (%d).", sharedMemorySize + 40, SharedMemoryPerBlock(0));
			exit(EXIT_FAILURE);
		}

		printf("\nNqueenFromDepthCUDA<<<%d, %d, %d>>>\n\n", blockSize, threadsPerBlock, sharedMemorySize);
		cudaEventRecord(start);
		NqueenFromDepthCUDA<<<blockSize, threadsPerBlock, sharedMemorySize>>>(boardsize, threadCount, depth, data_gpu); /* find solutions */
		cudaEventRecord(stop);
		CudaErrorInfo("NqueenFromDepthCUDA", err);

		err = cudaThreadSynchronize();
		CudaErrorInfo("cudaThreadSynchronize", err);

		err = cudaMemcpy(data, data_gpu, mem_size, cudaMemcpyDeviceToHost);
		CudaErrorInfo("cudaMemcpy gpu-> dest", err);

		// compute elapsed time
		float executionTime;
		cudaEventElapsedTime(&executionTime, start, stop);
		printf("\nGPU time: %f ms\n\n", executionTime);

		cudaThreadExit();
		CudaErrorInfo("cudaThreadExit", err);
	}
	else
	{
#endif
		printf("\nNqueenFromDepth\n\n");
		for(int threadId=0; threadId<threadCount; threadId++)
			NqueenFromDepth(boardsize, threadId, depth, data); /* find solutions */
#ifdef CUDA
	}
#endif

	for(int threadId=0; threadId<threadCount; threadId++)
		solutionCount += data[threadId * depth];

	return solutionCount;
}
