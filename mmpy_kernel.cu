// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include "utils.h"
#include "types.h"
using namespace std;

#define BLOCK_W 32
#define BLOCK_H 32

// the A-block is square with dimensions: BLOCK_H x BLOCK_H
// the B-block and C-block are rectangular with dimensions: BLOCK_H x BLOCK_W
__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B)
{
	int jBlock = blockIdx.x;
	int iBlock = blockIdx.y;

	int iThread = threadIdx.x; // each block's has BLOCK_H threads (one for each row of the C-block)

	int AsBegin = N * BLOCK_H * iBlock;
	int AsStep  = BLOCK_H;

	int BsBegin = BLOCK_W * jBlock;
	int BsStep  = BLOCK_H * N;

	__shared__ _DOUBLE_ As[BLOCK_H][BLOCK_H]; // 'As' holds the A-block 

	_DOUBLE_ b[BLOCK_W]; // 'b' holds a row of the B-block
	_DOUBLE_ c[BLOCK_W]; // we compute a row of the C-block and put the result in 'c' temporarily

	#pragma unroll
	for (int j = 0; j < BLOCK_W; ++j) // initialize c[] to a row vector of all zeroes
	{
		c[j] = 0;
	}

	//////////////////////////////////////
	//// Perform the blocked multiply ////
	//////////////////////////////////////

	for (	int AsIndex = AsBegin, BsIndex = BsBegin;
			AsIndex <= AsBegin + N - 1;
			AsIndex += AsStep, BsIndex += BsStep
		)
	{
		// Load a '1 by BLOCK_W' vector of the B-block 
		// Load a 'BLOCK_H by BLOCK_H' block of A 
		// These access global memory, so we want successive j's to correspond to contiguous memory
		for (int j = 0; j < BLOCK_W; ++j)
		{
			b[j] = B[BsIndex + N * iThread + j];
		}
		for (int j = 0; j < BLOCK_H; ++j)
		{
			As[j][iThread] = A[AsIndex + N * iThread + j]; // transpose to avoid 'bank conflict'
		}
		__syncthreads();

		#pragma unroll
		for (int j = 0; j < BLOCK_H; ++j) // for each element in a row of As
		{
			#pragma unroll
			for (int jj = 0; jj < BLOCK_W; ++jj)
			{
				c[jj] += As[j][iThread] * b[jj];
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
