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

	int iThread = threadIdx.x; // each block's has BLOCK_H threads (one for each row of the C-block)

	int AsBegin = (N * BLOCK_H) * iBlock;
	int AsStep  = BLOCK_W;

	int BsBegin = BLOCK_W * jBlock;
	int BsStep  = BLOCK_W * N;

	__shared__ _DOUBLE_ Bs[BLOCK_W][BLOCK_W]; // 'Bs' holds the shared, parallel-loaded, B-block 

	_DOUBLE_ a; // the BLOCK_H a's (one for each thread), together hold a column of the A-block
	_DOUBLE_ c[BLOCK_W] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

	//////////////////////////////////////
	//// Perform the blocked multiply ////
	//////////////////////////////////////

	for (	int AsIndex = AsBegin, BsIndex = BsBegin;
				AsIndex < AsBegin + N;
				AsIndex += AsStep, BsIndex += BsStep	)
	{
		int Bi = iThread / BLOCK_W;
		int Bj = iThread % BLOCK_W;

		Bs[Bi     ][Bj] = B[BsIndex + N* Bi       + Bj];
		Bs[Bi +  4][Bj] = B[BsIndex + N*(Bi +  4) + Bj];
		Bs[Bi +  8][Bj] = B[BsIndex + N*(Bi +  8) + Bj];
		Bs[Bi + 12][Bj] = B[BsIndex + N*(Bi + 12) + Bj];

		__syncthreads();

		#pragma unroll
		for (int k = 0; k < BLOCK_W; ++k) // ... for each column of the A-block
		{
			a = A[AsIndex + N*iThread + k];

			#pragma unroll
			for (int j = 0; j < BLOCK_W; ++j)
			{
				c[j] += a * Bs[k][j];
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
