// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include "utils.h"
#include "types.h"
using namespace std;

// The effective block size, in actuality we only have BLOCK_H threads
#define BLOCK_W 16 
#define BLOCK_H 64 

__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B)
{
	int jBlock = blockIdx.x;
	int iBlock = blockIdx.y;

	int iThread = threadIdx.x; // each block's has BLOCK_W threads (one for each column of the C-block)

	int AsBegin = (N * BLOCK_H) * iBlock;
	int AsStep  = BLOCK_H;

	int BsBegin = BLOCK_W * jBlock;
	int BsStep  = BLOCK_H * N;

	__shared__ _DOUBLE_ Bs[BLOCK_W][BLOCK_W]; // 'Bs' holds the shared, parallel-loaded, B-block 

	_DOUBLE_ a; // the BLOCK_H a's (one for each thread), together hold a column of the A-block
	_DOUBLE_ c[BLOCK_W]; // the BLOCK_H c's (one for each thread) accumulate into a block of C

	#pragma unroll
	for (int j = 0; j < BLOCK_W; ++j) // initialize c[] to a row vector of all zeroes
	{
		c[j] = 0;
	}

	//////////////////////////////////////
	//// Perform the blocked multiply ////
	//////////////////////////////////////

	for (	int AsIndex = AsBegin, BsIndex = BsBegin;
			AsIndex < AsBegin + N;
			AsIndex += AsStep, BsIndex += BsStep
		)
	{
		for (int j = 0; j < BLOCK_W; ++j) // load shared B block in parallel
		{
			Bs[j][iThread] = B[BsIndex + N*iThread + j]; // transpose to avoid 'bank conflict'
		}
		__syncthreads();

		#pragma unroll
		for (int j = 0; j < BLOCK_W; ++j) // ... for each column of the A-block
		{
			a = A[AsIndex + N*iThread + j];

			#pragma unroll
			for (int jj = 0; jj < BLOCK_W; ++jj) // accumulate a*Bs[:][iThread] into c[:]
			{
				c[jj] += a * Bs[jj][iThread];
			}
		}
		__syncthreads();
	}

	////////////////////////////
	//// Copy c back into C ////
	////////////////////////////

	#pragma unroll
	for (int j = 0; j < BLOCK_W; ++j)
	{
		C[N * (BLOCK_H * iBlock + iThread) + BLOCK_W * jBlock + j] = c[j];
	}
}
