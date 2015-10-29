// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include "utils.h"
#include "types.h"
using namespace std;

// The effective block size, in actuality we only have BLOCK_W threads

#define BLOCK_W 64 
#define BLOCK_H 16

__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B)
{
	int jBlock = blockIdx.x;
	int iBlock = blockIdx.y;

	int jThread = threadIdx.x;

	int AsBegin = (N * BLOCK_H) * iBlock;
	int AsStep  = BLOCK_H;

	int BsBegin = BLOCK_W * jBlock;
	int BsStep  = BLOCK_H * N;

	__shared__ _DOUBLE_ As[BLOCK_H][BLOCK_H + 1];

	_DOUBLE_ b;
	_DOUBLE_ c[BLOCK_H] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

	//////////////////////////////////////
	//// Perform the blocked multiply ////
	//////////////////////////////////////

	for (	int AsIndex = AsBegin, BsIndex = BsBegin;
				AsIndex < AsBegin + N;
				AsIndex += AsStep, BsIndex += BsStep	)
	{
		int Ai = jThread / BLOCK_H;
		int Aj = jThread % BLOCK_H;

		As[Ai     ][Aj] = A[AsIndex + N* Ai       + Aj];
		As[Ai +  4][Aj] = A[AsIndex + N*(Ai +  4) + Aj];
		As[Ai +  8][Aj] = A[AsIndex + N*(Ai +  8) + Aj];
		As[Ai + 12][Aj] = A[AsIndex + N*(Ai + 12) + Aj];

		__syncthreads();

		#pragma unroll
		for (int k = 0; k < BLOCK_H; ++k)
		{
			// b is loaded from main memory.
			// Between the threads in the warp, the loads are coalesced,
			// since jThread has unit coefficient in B[... + jThread].
			b = B[BsIndex + N*k + jThread];

			// Since k appears in the least significant index of As[...][k],
			// the threads in the warp will be distributed evenly among the banks
			c[ 0] += b * As[ 0][k];
			c[ 1] += b * As[ 1][k];
			c[ 2] += b * As[ 2][k];
			c[ 3] += b * As[ 3][k];
			c[ 4] += b * As[ 4][k];
			c[ 5] += b * As[ 5][k];
			c[ 6] += b * As[ 6][k];
			c[ 7] += b * As[ 7][k];
			c[ 8] += b * As[ 8][k];
			c[ 9] += b * As[ 9][k];
			c[10] += b * As[10][k];
			c[11] += b * As[11][k];
			c[12] += b * As[12][k];
			c[13] += b * As[13][k];
			c[14] += b * As[14][k];
			c[15] += b * As[15][k];
		}
		__syncthreads();
	}

	////////////////////////////
	//// Copy c back into C ////
	////////////////////////////

	#pragma unroll
	for (int i = 0; i < BLOCK_H; ++i)
	{
		C[N * (BLOCK_H * iBlock + i) + BLOCK_W * jBlock + jThread] = c[i];
	}
}
