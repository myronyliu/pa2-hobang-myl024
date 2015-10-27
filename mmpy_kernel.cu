// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include "utils.h"
#include "types.h"
using namespace std;

#define BLOCK_SIZE 32

__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

	int x = bx*BLOCK_SIZE + tx;
	int y = by*BLOCK_SIZE + ty;

    int wA = N;
    int wB = N;

    int aBegin = wA * BLOCK_SIZE * by;
    int aEnd   = aBegin + wA - 1;
    int aStep  = BLOCK_SIZE;
    int bBegin = BLOCK_SIZE * bx;
    int bStep  = BLOCK_SIZE * wB;
	
	__shared__ _DOUBLE_ As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ _DOUBLE_ Bs[BLOCK_SIZE][BLOCK_SIZE];

    _DOUBLE_ Csub[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep)
    {
		As[ty     ][tx] = A[a + wA *  ty       + tx];
        Bs[ty     ][tx] = B[b + wB *  ty       + tx];
		As[ty +  4][tx] = A[a + wA * (ty +  4) + tx];
 		Bs[ty +  4][tx] = B[b + wB * (ty +  4) + tx];
		As[ty +  8][tx] = A[a + wA * (ty +  8) + tx];
 		Bs[ty +  8][tx] = B[b + wB * (ty +  8) + tx];
		As[ty + 12][tx] = A[a + wA * (ty + 12) + tx];
 		Bs[ty + 12][tx] = B[b + wB * (ty + 12) + tx];
		As[ty + 16][tx] = A[a + wA * (ty + 16) + tx];
 		Bs[ty + 16][tx] = B[b + wB * (ty + 16) + tx];
		As[ty + 20][tx] = A[a + wA * (ty + 20) + tx];
 		Bs[ty + 20][tx] = B[b + wB * (ty + 20) + tx];
		As[ty + 24][tx] = A[a + wA * (ty + 24) + tx];
 		Bs[ty + 24][tx] = B[b + wB * (ty + 24) + tx];
		As[ty + 28][tx] = A[a + wA * (ty + 28) + tx];
 		Bs[ty + 28][tx] = B[b + wB * (ty + 28) + tx];

        __syncthreads();

		if (x < N && y < N)
		{
			int w = N - (a - aBegin);
			int kMax = (w < BLOCK_SIZE) ? w : BLOCK_SIZE;

			#pragma unroll
			for (int k = 0; k < kMax; ++k)
			{
				Csub[0] += As[ty][k] * Bs[k][tx];

				if (y + 28 < N)
				{
					Csub[1] += As[ty +  4][k] * Bs[k][tx];
					Csub[2] += As[ty +  8][k] * Bs[k][tx];
					Csub[3] += As[ty + 12][k] * Bs[k][tx];
					Csub[4] += As[ty + 16][k] * Bs[k][tx];
					Csub[5] += As[ty + 20][k] * Bs[k][tx];
					Csub[6] += As[ty + 24][k] * Bs[k][tx];
					Csub[7] += As[ty + 28][k] * Bs[k][tx];
				}
				else if (y + 24 < N)
				{
					Csub[1] += As[ty +  4][k] * Bs[k][tx];
					Csub[2] += As[ty +  8][k] * Bs[k][tx];
					Csub[3] += As[ty + 12][k] * Bs[k][tx];
					Csub[4] += As[ty + 16][k] * Bs[k][tx];
					Csub[5] += As[ty + 20][k] * Bs[k][tx];
					Csub[6] += As[ty + 24][k] * Bs[k][tx];
				}
				else if (y + 20 < N)
				{
					Csub[1] += As[ty +  4][k] * Bs[k][tx];
					Csub[2] += As[ty +  8][k] * Bs[k][tx];
					Csub[3] += As[ty + 12][k] * Bs[k][tx];
					Csub[4] += As[ty + 16][k] * Bs[k][tx];
					Csub[5] += As[ty + 20][k] * Bs[k][tx];
				}
				else if (y + 16 < N)
				{
					Csub[1] += As[ty +  4][k] * Bs[k][tx];
					Csub[2] += As[ty +  8][k] * Bs[k][tx];
					Csub[3] += As[ty + 12][k] * Bs[k][tx];
					Csub[4] += As[ty + 16][k] * Bs[k][tx];
				}
				else if (y + 12 < N)
				{
					Csub[1] += As[ty +  4][k] * Bs[k][tx];
					Csub[2] += As[ty +  8][k] * Bs[k][tx];
					Csub[3] += As[ty + 12][k] * Bs[k][tx];
				}
				else if (y + 8 < N)
				{
					Csub[1] += As[ty +  4][k] * Bs[k][tx];
					Csub[2] += As[ty +  8][k] * Bs[k][tx];
				}
				else if (y + 4 < N)
				{
					Csub[1] += As[ty +  4][k] * Bs[k][tx];
				}
			}
		}
        __syncthreads();
    }

	if (x < N)
	{
		int c = wB*BLOCK_SIZE*by + BLOCK_SIZE*bx;

		C[c + wB*ty + tx] = Csub[0];

		if (y + 28 < N)
		{
			C[c + wB*(ty +  4) + tx] = Csub[1];
			C[c + wB*(ty +  8) + tx] = Csub[2];
			C[c + wB*(ty + 12) + tx] = Csub[3];
			C[c + wB*(ty + 16) + tx] = Csub[4];
			C[c + wB*(ty + 20) + tx] = Csub[5];
			C[c + wB*(ty + 24) + tx] = Csub[6];
			C[c + wB*(ty + 28) + tx] = Csub[7];
		}
		else if (y + 24 < N)
		{
			C[c + wB*(ty +  4) + tx] = Csub[1];
			C[c + wB*(ty +  8) + tx] = Csub[2];
			C[c + wB*(ty + 12) + tx] = Csub[3];
			C[c + wB*(ty + 16) + tx] = Csub[4];
			C[c + wB*(ty + 20) + tx] = Csub[5];
			C[c + wB*(ty + 24) + tx] = Csub[6];
		}
		else if (y + 20 < N)
		{
			C[c + wB*(ty +  4) + tx] = Csub[1];
			C[c + wB*(ty +  8) + tx] = Csub[2];
			C[c + wB*(ty + 12) + tx] = Csub[3];
			C[c + wB*(ty + 16) + tx] = Csub[4];
			C[c + wB*(ty + 20) + tx] = Csub[5];
		}
		else if (y + 16 < N)
		{
			C[c + wB*(ty +  4) + tx] = Csub[1];
			C[c + wB*(ty +  8) + tx] = Csub[2];
			C[c + wB*(ty + 12) + tx] = Csub[3];
			C[c + wB*(ty + 16) + tx] = Csub[4];
		}
		else if (y + 12 < N)
		{
			C[c + wB*(ty +  4) + tx] = Csub[1];
			C[c + wB*(ty +  8) + tx] = Csub[2];
			C[c + wB*(ty + 12) + tx] = Csub[3];
		}
		else if (y + 8 < N)
		{
			C[c + wB*(ty +  4) + tx] = Csub[1];
			C[c + wB*(ty +  8) + tx] = Csub[2];
		}
		else if (y + 4 < N)
		{
			C[c + wB*(ty +  4) + tx] = Csub[1];
		}
	}
}
