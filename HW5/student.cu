/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/
#include "utils.h"
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>

__global__
void naiveHisto(
  const unsigned int* const values,
	unsigned int* const histogram,
  int numVals
){
	if ((threadIdx.x + blockDim.x*blockIdx.x) >= numVals) return;
	atomicAdd(&(histogram[values[threadIdx.x + blockDim.x*blockIdx.x]]), 1);
}

__global__
void fastHistogram(
  const unsigned int* const values,
	unsigned int* const histogram,
  int numVals,int numBins
){
	extern __shared__ unsigned int sharedHistogram[];

	for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
		sharedHistogram[i] = 0;
	}

	__syncthreads();

	atomicAdd(&sharedHistogram[values[threadIdx.x + blockIdx.x*blockDim.x]], 1);
	
	__syncthreads();

	for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
		atomicAdd(&histogram[i], sharedHistogram[i]);
	}
}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
  const int NUM_THREADS = 1024;
	int numBlocks = ceil(numElems / NUM_THREADS);

	naiveHisto <<<numBlocks, NUM_THREADS>>> (d_vals, d_histo, numElems);

	// fastHistogram <<<numBlocks, NUM_THREADS, sizeof(unsigned int)*numBins>>> (d_vals, d_histo, numElems, numBins);

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}