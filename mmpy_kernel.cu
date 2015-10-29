// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include "utils.h"
#include "types.h"
using namespace std;

// The effective block size, in actuality we only have BLOCK_W threads
// NOTE: we stipulate that BLOCK_W must be divisible by BLOCK_w
#define BLOCK_W 16 
#define BLOCK_H 8 

__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B)
{
	int jBlock = blockIdx.x;
	int iBlock = blockIdx.y;

	int jThread = threadIdx.x;

	int AsBegin = (N * BLOCK_H) * iBlock;
	int AsStep  = BLOCK_H;

	int BsBegin = BLOCK_W * jBlock;
	int BsStep  = BLOCK_H * N;

	__shared__ _DOUBLE_ As[BLOCK_H][BLOCK_H];

	_DOUBLE_ b;
	_DOUBLE_ c[BLOCK_H];

	#pragma unroll
	for (int i = 0; i < BLOCK_H; ++i)
	{
		c[i] = 0;
	}

	//////////////////////////////////////
	//// Perform the blocked multiply ////
	//////////////////////////////////////

	for (	int AsIndex = AsBegin, BsIndex = BsBegin;
				AsIndex < AsBegin + N;
				AsIndex += AsStep, BsIndex += BsStep	)
	{
		int Ai = jThread / BLOCK_H;
		int Aj = jThread % BLOCK_H;

		#pragma unroll
		for (int i = 0; i < BLOCK_H; i += BLOCK_W/BLOCK_H)
		{
			As[Ai + i][Aj] = A[AsIndex + N*(Ai + i) + Aj];
		}
		__syncthreads();

		#pragma unroll
		for (int k = 0; k < BLOCK_H; ++k)
		{
			// b is loaded from main memory.
			// Between the threads in the warp, the loads are coalesced,
			// since jThread has unit coefficient in B[... + jThread].
			b = B[BsIndex + N*k + jThread];

			#pragma unroll
			for (int i = 0; i < BLOCK_H; ++i)
			{
				// Since k appears in the least significant index of As[...][k],
				// the threads in the warp will be distributed evenly among the banks
				c[i] += b * As[i][k];
			}
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
