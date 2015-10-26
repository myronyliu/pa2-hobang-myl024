// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include "utils.h"
#include "types.h"
using namespace std;
#define BLOCK_SIZE 32

__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int wA = N;
    int wB = N;

    int aBegin = wA * BLOCK_SIZE * by;
    int aEnd   = aBegin + wA - 1;
    int aStep  = BLOCK_SIZE;
    int bBegin = BLOCK_SIZE * bx;
    int bStep  = BLOCK_SIZE * wB;
	
	__shared__ _DOUBLE_ As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ _DOUBLE_ Bs[BLOCK_SIZE][BLOCK_SIZE];
    _DOUBLE_ Csub[2] = {0,0};

    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep)
    {
		As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];
		As[ty + 16][tx] = A[a + wA * (ty + 16) + tx];
 		Bs[ty + 16][tx] = B[b + wB * (ty + 16) + tx];

        __syncthreads();

		#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            Csub[0] += As[ty][k] * Bs[k][tx];
			Csub[1] += As[ty + 16][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub[0];
	C[c + wB * (ty + 16) + tx] = Csub[1];
}
