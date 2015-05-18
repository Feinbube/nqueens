/*  Jeff Somers
 *
 *  Copyright (c) 2002
 *
 *  jsomers@alumni.williams.edu
 *  or
 *  allagash98@yahoo.com
 *
 *  April, 2002
 *  
 *  Program:  nq
 *  
 *  Program to find number of solutions to the N queens problem.
 *  This program assumes a twos complement architecture.
 *
 *  For example, you can arrange 4 queens on 4 x 4 chess so that
 *  none of the queens can attack each other:
 *
 *  Two solutions:
 *     _ Q _ _        _ _ Q _
 *     _ _ _ Q        Q _ _ _
 *     Q _ _ _        _ _ _ Q
 *     _ _ Q _    and _ Q _ _
 *    
 *  Note that these are separate solutions, even though they
 *  are mirror images of each other.
 *
 *  Likewise, a 8 x 8 chess board has 92 solutions to the 8 queens
 *  problem.
 *
 *  Command Line Usage:
 *
 *          nq N
 *
 *       where N is the size of the N x N board.  For example,
 *       nq 4 will find the 4 queen solution for the 4 x 4 chess
 *       board.
 *
 *  By default, this program will only print the number of solutions,
 *  not board arrangements which are the solutions.  To print the 
 *  boards, uncomment the call to printtable in the Nqueen function.
 *  Note that printing the board arrangements slows down the program
 *  quite a bit, unless you pipe the output to a text file:
 *
 *  nq 10 > output.txt
 *
 *
 *  The number of solutions for the N queens problems are known for
 *  boards up to 23 x 23.  With this program, I've calculated the
 *  results for boards up to 21 x 21, and that took over a week on
 *  an 800 MHz PC.  The algorithm is approximated O(n!) (i.e. slow), 
 *  and calculating the results for a 22 x 22 board will take about 8.5 
 *  times the amount of time for the 21 x 21 board, or over 8 1/2 weeks.
 *  Even with a 10 GHz machine, calculating the results for a 23 x 23 
 *  board would take over a month.  Of course, setting up a cluster of
 *  machines (or a distributed client) would do the work in less time.
 *  
 *  (from Sloane's On-Line Encyclopedia of Integer Sequences,
 *   Sequence A000170
 *   http://www.research.att.com/cgi-bin/access.cgi/as/njas/sequences/eisA.cgi?Anum=000170
 *   )
 *
 *   Board Size:       Number of Solutions to          Time to calculate 
 *   (length of one        N queens problem:              on 800MHz PC
 *    side of N x N                                    (Hours:Mins:Secs)
 *    chessboard)
 *
 *     1                                  1                    n/a
 *     2                                  0                   < 0 seconds
 *     3                                  0                   < 0 seconds
 *     4                                  2                   < 0 seconds
 *     5                                 10                   < 0 seconds
 *     6                                  4                   < 0 seconds
 *     7                                 40                   < 0 seconds
 *     8                                 92                   < 0 seconds
 *     9                                352                   < 0 seconds 
 *    10                                724                   < 0 seconds
 *    11                               2680                   < 0 seconds
 *    12                              14200                   < 0 seconds
 *    13                              73712                   < 0 seconds
 *    14                             365596                  00:00:01
 *    15                            2279184                  00:00:04
 *    16                           14772512                  00:00:23
 *    17                           95815104                  00:02:38
 *    18                          666090624                  00:19:26
 *    19                         4968057848                  02:31:24
 *    20                        39029188884                  20:35:06
 *    21                       314666222712                 174:53:45
 *    22                      2691008701644                     ?
 *    23                     24233937684440                     ?
 *    24                                  ?                     ?
 */

#include "Shared.h"
#include "JmsNQueens.h"

int queenCount = 0;
int depth = DEPTH;
int threadsPerBlock = THREADS_PER_BLOCK;
bool verbose = false;
bool cpuOnly = false;
bool infoOnly = false;
int gpuIndex = 0;

void usage(const char *name)
{
		printf("Usage: %s [-i] [-g gpu_index] [-t threads_per_block] [-d depth] [-v] [-c] queen_count\n\n", name);
		printf("Options:\n");
		printf("\t-i Info mode. Displays information on the applied graphic cards.\n");
		printf("\t-g Index of the GPU device that is used.\n");
		printf("\t-t Used to set the count of threads for each block. [1..512]\n");
		printf("\t-d Indicates the depth to which the solutions shall be precalculated.\n");
		printf("\t   Influences the number of threads that are used. [1..queen_count]\n");
		printf("\t-v Verbose mode. Displays additional information.\n");
		printf("\t-c CPU mode. No GPU involved.\n");
		exit(0);
}

void checkArguments()
{
	if (1 > threadsPerBlock || 512 < threadsPerBlock)
    {
        printf("Threads per Block must be between %d and %d, inclusive.\n", 1, 512 );
        exit(0);
    }

	if (MIN_BOARDSIZE > queenCount || MAX_BOARDSIZE < queenCount)
    {
        printf("Queen count must be between %d and %d, inclusive.\n", MIN_BOARDSIZE, MAX_BOARDSIZE );
        exit(0);
    }

	if (1 > depth || queenCount < depth)
    {
        printf("Depth must be between %d and %d, inclusive.\n", 1, queenCount );
        exit(0);
    }
}

void printArguements()
{
	printf("Parameters: %d Queens / %d Threads per block / %d Depth / GPU %d / ", queenCount, threadsPerBlock, depth, gpuIndex);

	if(!verbose)
		printf("Not ");

	printf("Verbose / ");

	if(!cpuOnly)
		printf("GPU\n\n");
	else
		printf("CPU\n\n");
}

void parseArgument(int * argIndex, int argc, char *argv[])
{
	if(strcmp(argv[*argIndex], "-t") == 0)
	{
		(*argIndex)++;
		threadsPerBlock = atoi(argv[*argIndex]);
		return;
	}

	if(strcmp(argv[*argIndex], "-d") == 0)
	{
		(*argIndex)++;
		depth = atoi(argv[*argIndex]);
		return;
	}

	if(strcmp(argv[*argIndex], "-g") == 0)
	{
		(*argIndex)++;
		gpuIndex = atoi(argv[*argIndex]);
		return;
	}

	if(strcmp(argv[*argIndex], "-v") == 0)
	{
		verbose = true;
		return;
	}

	if(strcmp(argv[*argIndex], "-c") == 0)
	{
		cpuOnly = true;
		return;
	}

	if(strcmp(argv[*argIndex], "-i") == 0)
	{
		infoOnly = true;
		return;
	}

	if(*argIndex == argc -1)
		queenCount = atoi(argv[*argIndex]);
	else
		usage(argv[0]);
}

void parseCmdLine(int argc, char *argv[])
{
	if(argc < 2)
		usage(argv[0]);

	for(int i=1; i<argc; i++)
		parseArgument(&i, argc, argv);

	printArguements();
	checkArguments();
}


void greetings(int argc, char** argv)
{
	printf("\n");
	printf("N Queens\n");
	printf("--------\n");
	printf("This program calculates the total number of solutions to the N Queens problem.\n");
	printf("Program by Jeff Somers. Edited by Frank Feinbube.\n\n");

    parseCmdLine(argc, argv);

#ifdef CUDA
	if(!cpuOnly)
	{
		printf("\nThis Version is CUDA-enabled.\n");
		CudaInfo();
		printf("\n");
	}
#endif
}


char * getDuration(ULL clockDifference, bool showMillis)
{
	int milliseconds = (int)(clockDifference % CLOCKS_PER_SEC);
	int seconds = (int)((clockDifference / CLOCKS_PER_SEC) % 60);
	int minutes = (int)((clockDifference / (CLOCKS_PER_SEC*60ULL)) % 60);
	int hours = (int)((clockDifference / (CLOCKS_PER_SEC*60ULL*60)) % 24);
	int days = (int)(clockDifference / (CLOCKS_PER_SEC*60ULL*60*24));

	char result[20];

	if(showMillis)
		snprintf(result, 20, "%d:%02d:%02d:%02d:%03d", days, hours, minutes, seconds, milliseconds);
	else
		snprintf(result, 20, "%d:%02d:%02d:%02d", days, hours, minutes, seconds);

	return strdup(result);
}


char * getCurrentTime()
{
	 time_t currentTime;
	 time(&currentTime);

	 char * result = ctime(&currentTime);
	 result[strlen(result)-1] = '\0';

	 return result;
}


/* main routine for N Queens program.*/
int main(int argc, char** argv)
{
	ULL start, finish;

	greetings(argc, argv);

	if(infoOnly)
		return 0;

	printf("\n\n");
    printf("--[Started Calculation at %s]---------------------\n", getCurrentTime());

	start = clock();
	ULL solutionCount = calculateSolutions(gpuIndex, queenCount, depth, threadsPerBlock, verbose, cpuOnly);
	finish = clock();
	
	printf("--[Finished Calculation at %s]--------------------\n\n", getCurrentTime());
	printf("\n\n");

	printf("Solutions: %llu\n", solutionCount);
	printf("Duration:  %s\n", getDuration(finish - start, true));

    return 0;
}
