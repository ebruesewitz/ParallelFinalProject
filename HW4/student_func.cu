//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

 __global__
 void setHistogramData(
        unsigned int iterationNumber,
        unsigned int * d_bins,
        unsigned int* const d_inputVals, 
        const int numEls
){  
        int mid = threadIdx.x + blockDim.x * blockIdx.x;
        if(mid >= numEls)
                return;

        int bin = ((d_inputVals[mid] & (1<<iterationNumber)) == (1<<iterationNumber)) ? 1 : 0;
        
        if(bin) 
                atomicAdd(&d_bins[1], 1);
        else
                atomicAdd(&d_bins[0], 1);
}

__global__
void exclusivePrefixSum(
        unsigned int iterationNumber,
        unsigned int const * d_inputVals,
        unsigned int * d_output,
        const int numElems,
        unsigned int base,
        unsigned int threadSize
) {
    int mid = threadIdx.x + threadSize * base;

    if(mid >= numElems)
        return;

    unsigned int value = 0;
    if(mid > 0)
        value = ((d_inputVals[mid-1] & (1<<iterationNumber))  == (1<<iterationNumber)) ? 1 : 0;
    else
        value = 0;

    d_output[mid] = value;
    
    __syncthreads();
    
    for(int i = 1; i <= threadSize; i *= 2) {
        int loc = mid - i; 
         
        if(loc >= 0 && loc >=  threadSize*base)
             value = d_output[loc];
        __syncthreads();
        if(loc >= 0 && loc >= threadSize*base)
            d_output[mid] += value;
        __syncthreads();
    }
    if(base > 0)
        d_output[mid] += d_output[base*threadSize - 1];
}

 __global__
void getRelativeOffset(
    unsigned int iterationNumber,
    unsigned int* const d_inputVals,
    unsigned int* const d_inputPos,
    unsigned int* d_outputVals,
    unsigned int* d_outputPos,
    unsigned int* d_outputMove,
    unsigned int* const d_visited,
    unsigned int  one_pos,
    const size_t numElems
){
    
    int mid = threadIdx.x + blockDim.x * blockIdx.x;
    if(mid >= numElems)
        return;
    
    unsigned int scan=0;
    unsigned int base=0;
    unsigned int one= 1;
    if( ( d_inputVals[mid] & (one<<iterationNumber)) == (1<<iterationNumber)) {
        scan = d_visited[mid]; 
        base = one_pos;
    } else {
        scan = (mid) - d_visited[mid];
        base = 0;
    }
    
    d_outputMove[mid] = base+scan;
    d_outputPos[base+scan]  = d_inputPos[mid];
    d_outputVals[base+scan] = d_inputVals[mid];
    
}


void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
    unsigned int* d_bins;
    unsigned int  h_bins[2];
    unsigned int* d_visited;
    unsigned int* d_moved;
    const size_t histSize = 2*sizeof(unsigned int);
    const size_t arraySize   = numElems*sizeof(unsigned int);
    
    // start by allocating the memory we'll need
    checkCudaErrors(cudaMalloc(&d_bins, histSize));
    checkCudaErrors(cudaMalloc(&d_visited, arraySize));
    checkCudaErrors(cudaMalloc(&d_moved, arraySize));
    
    // calculating thread and block dimensions
    dim3 threadSize(1024);
    dim3 histBlockSize( (int)ceil((float)numElems/(float)threadSize.x) + 1 );
    
    // start a pass through the data
    for(unsigned int iterationNumber = 0; iterationNumber < 32; iterationNumber++) {

        checkCudaErrors(cudaMemset(d_bins, 0, histSize));
        checkCudaErrors(cudaMemset(d_visited, 0, arraySize));
        checkCudaErrors(cudaMemset(d_outputVals, 0, arraySize));
        checkCudaErrors(cudaMemset(d_outputPos, 0, arraySize));
        
        // 1) Histogram of the number of occurrences of each digit
        setHistogramData<<<histBlockSize, threadSize>>>(iterationNumber, d_bins, d_inputVals, numElems);
        cudaDeviceSynchronize(); 
        checkCudaErrors(cudaGetLastError());
       
        // copy the histogram data back to the host
        checkCudaErrors(cudaMemcpy(&h_bins, d_bins, histSize, cudaMemcpyDeviceToHost));
        
        // 2) Exclusive Prefix Sum of Histogram
        for(int i = 0; i < ((int)ceil( (float)numElems/(float)threadSize.x ) + 1); i++) {
            exclusivePrefixSum<<<dim3(1), threadSize>>>(
                iterationNumber,
                d_inputVals,
                d_visited,
                numElems,
                i,
                threadSize.x
            );
            cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
        }        

        // 3) Determine relative offset of each digit
        //    and move it to the correct position
        getRelativeOffset<<<histBlockSize, threadSize>>>(
            iterationNumber,
            d_inputVals,
            d_inputPos,
            d_outputVals,
            d_outputPos,
            d_moved,
            d_visited,
            h_bins[0],
            numElems
        );
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
        
        
        // copy histogram data to output
        checkCudaErrors(cudaMemcpy(d_inputVals, d_outputVals, arraySize, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(d_inputPos, d_outputPos, arraySize, cudaMemcpyDeviceToDevice));

        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    }
    
    checkCudaErrors(cudaFree(d_moved));
    checkCudaErrors(cudaFree(d_visited));
    checkCudaErrors(cudaFree(d_bins));
}
