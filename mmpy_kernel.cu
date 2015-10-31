// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include "utils.h"
#include "types.h"
using namespace std;

// The effective block size, in actuality we only have BLOCKDIM_X threads
// NOTE: we stipulate that BLOCKDIM_X must be divisible by BLOCKDIM_Y

__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B)
{
	int jBlock = blockIdx.x;
	int iBlock = blockIdx.y;

	int c_H = (iBlock == N/BLOCKDIM_Y) ? N%BLOCKDIM_Y : BLOCKDIM_Y;

	int jThread = threadIdx.x;

	int AsBegin = (N * BLOCKDIM_Y) * iBlock;
	int AsStep  = BLOCKDIM_Y;

	int BsBegin = BLOCKDIM_X * jBlock;
	int BsStep  = BLOCKDIM_Y * N;

	__shared__ _DOUBLE_ As[BLOCKDIM_Y][BLOCKDIM_Y];

	_DOUBLE_ b;
	_DOUBLE_ c[BLOCKDIM_Y];

	#pragma unroll
	for (int i = 0; i < BLOCKDIM_Y; ++i)
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
		int Ai = jThread / BLOCKDIM_Y;
		int Aj = jThread % BLOCKDIM_Y;

		int kMax = N - (AsIndex - AsBegin); // the range of the contraction index
		kMax = (kMax < BLOCKDIM_Y) ? kMax : BLOCKDIM_Y;

		if (Aj < kMax)
		{
			#pragma unroll
			for (int i = 0; i < BLOCKDIM_Y; i += BLOCKDIM_X/BLOCKDIM_Y)
			{
				As[Ai + i][Aj] = A[AsIndex + N*(Ai + i) + Aj];
			}
		}
		else
		{
			#pragma unroll
			for (int i = 0; i < BLOCKDIM_Y; i += BLOCKDIM_X/BLOCKDIM_Y)
			{
				As[Ai + i][Aj] = 0;
			}
		}
		__syncthreads();

		#pragma unroll
		for (int k = 0; k < BLOCKDIM_Y; ++k)
		{
			// b is loaded from main memory.
			// Between the threads in the warp, the loads are coalesced,
			// since jThread has unit coefficient in B[... + jThread].
			b = (k < kMax) ? B[BsIndex + N*k + jThread] : 0;

			#pragma unroll
			for (int i = 0; i < BLOCKDIM_Y; ++i)
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

	if (BLOCKDIM_X*jBlock + jThread < N)
	{
		#pragma unroll
		for (int i = 0; i < c_H; ++i)
		{
			C[N * (BLOCKDIM_Y * iBlock + i) + BLOCKDIM_X * jBlock + jThread] = c[i];
		}
	}
}
