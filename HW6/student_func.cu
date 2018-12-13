//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.
   
      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */



#include "utils.h"
#include <thrust/host_vector.h>

// utility functions
__device__
int getm(int x, int y, size_t numColsSource) {
  return y*numColsSource + x;
}

__device__
bool isInBounds(const int x, const int y, const size_t numRowsSource, const size_t numColsSource) {
    return ((x < numColsSource) && (y < numRowsSource));
}

__device__
bool isMasked(uchar4 value) {
  return (value.x != 255 || value.y != 255 || value.z != 255);
}

// for STEP 1 & STEP 2
__global__
void computeMaskAndRegions(
  const uchar4* const d_sourceImg,
  int* d_border,
  int* d_interior,
  const size_t numRowsSource,
  const size_t numColsSource
){
  const int2 px = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    
  if(!isInBounds(px.x, px.y, numRowsSource, numColsSource))
      return;
  
  if(isMasked(d_sourceImg[px.y*numColsSource + px.x])) {
    int inbounds = 0;
    int interior = 0;

    // count how many of our neighbors are masked,
    // and how many neighbors we have
    if (isInBounds(px.x, px.y+1, numRowsSource, numColsSource)) {
      inbounds++;
      if(isMasked(d_sourceImg[(px.y+1)*numColsSource + px.x]))
        interior++;		
    }
    if (isInBounds(px.x, px.y-1, numRowsSource, numColsSource)) {
      inbounds++;
      if(isMasked(d_sourceImg[(px.y-1)*numColsSource + px.x]))
        interior++;		
    }
    if (isInBounds(px.x+1, px.y, numRowsSource, numColsSource)) {
      inbounds++;
      if(isMasked(d_sourceImg[(px.y)*numColsSource + px.x+1]))
        interior++;		
    }
    if (isInBounds(px.x-1, px.y, numRowsSource, numColsSource)) {
      inbounds++;
      if(isMasked(d_sourceImg[(px.y)*numColsSource + px.x-1]))
        interior++;		
    }

    d_interior[px.y*numColsSource + px.x] = 0;
    d_border[px.y*numColsSource + px.x]   = 0;
  
    // if all our neighbors are masked, then its interior
    // otherwise, if it is in the mask its a border
    if(inbounds == interior) {
      d_interior[px.y*numColsSource + px.x] = 1;
    } else if (interior > 0) {
      d_border[px.y*numColsSource + px.x] = 1;
    }
  }
}

// for STEP 3
__global__
void separateColorChannels(
  const uchar4* const inputImage,
  float* const redChannel,
  float* const greenChannel,
  float* const blueChannel,
  size_t numRows,
  size_t numCols
){
   const int2 px = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

   if(!isInBounds(px.x, px.y, numRows, numCols))
      return;

   redChannel[(px.y)*numCols + px.x] = (float)inputImage[(px.y)*numCols + px.x].x;
   greenChannel[(px.y)*numCols + px.x] = (float)inputImage[(px.y)*numCols + px.x].y;
   blueChannel[(px.y)*numCols + px.x] = (float)inputImage[(px.y)*numCols + px.x].z;
}

// for STEP 5
__global__
void jacobi(
   float* d_in,
   float* d_out,
   const int* d_border,
   const int* d_interior,
   float* d_sourceImg,
   float* d_destImg,
   size_t numRows,
   size_t numCols
){
   const int2 px = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
      
   if(!isInBounds(px.x, px.y, numRows, numCols))
      return;


   if(d_interior[(px.y)*numCols + px.x]==1) {
      float a = 0.f, b=0.f, c=0.0f, d=0.f;
      float sourceVal = d_sourceImg[(px.y)*numCols + px.x];

      if(isInBounds(px.x, px.y+1, numRows, numCols)) {
         d++;
         if(d_interior[(px.y+1)*numCols + px.x]==1) {
            a += d_in[(px.y+1)*numCols + px.x];
         } else if(d_border[(px.y+1)*numCols + px.x]==1) {
            b += d_destImg[(px.y+1)*numCols + px.x];
         }
         c += (sourceVal-d_sourceImg[(px.y+1)*numCols + px.x]);
      }
      
      if(isInBounds(px.x, px.y-1, numRows, numCols)) {
         d++;
         if(d_interior[(px.y-1)*numCols + px.x]==1) {
            a += d_in[(px.y-1)*numCols + px.x];
         } else if(d_border[(px.y-1)*numCols + px.x]==1) {
            b += d_destImg[(px.y-1)*numCols + px.x];
         }
         c += (sourceVal-d_sourceImg[(px.y-1)*numCols + px.x]);
      }
      
      if(isInBounds(px.x+1, px.y, numRows, numCols)) {
         d++;
         if(d_interior[(px.y)*numCols + px.x+1]==1) {
            a += d_in[(px.y)*numCols + px.x+1];
         } else if(d_border[(px.y)*numCols + px.x+1]==1) {
            b += d_destImg[(px.y)*numCols + px.x+1];
         }
         c += (sourceVal-d_sourceImg[(px.y)*numCols + px.x+1]);
      }
      
      if(isInBounds(px.x-1, px.y, numRows, numCols)) {
         d++;
         if(d_interior[(px.y)*numCols + px.x-1]==1) {
            a += d_in[(px.y)*numCols + px.x-1];
         } else if(d_border[(px.y)*numCols + px.x-1]==1) {
            b += d_destImg[(px.y)*numCols + px.x-1];
         }
         c += (sourceVal-d_sourceImg[(px.y)*numCols + px.x-1]);
      }
      
      d_out[(px.y)*numCols + px.x] = min(255.f, max(0.0, (a + b + c)/d));
   } else {
      d_out[(px.y)*numCols + px.x] = d_destImg[(px.y)*numCols + px.x];
   }
}

// for STEP 6
__global__
void createOutputImage(
   uchar4* finalImage,
   float* const redChannel,
   float* const greenChannel,
   float* const blueChannel,
   size_t numRows,
   size_t numCols
){
   const int2 px = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
      
   if(!isInBounds(px.x, px.y, numRows, numCols))
      return;
   
   finalImage[(px.y)*numCols + px.x].x = (char)redChannel[(px.y)*numCols + px.x];
   finalImage[(px.y)*numCols + px.x].y = (char)greenChannel[(px.y)*numCols + px.x];
   finalImage[(px.y)*numCols + px.x].z = (char)blueChannel[(px.y)*numCols + px.x];
}

void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{

  /* To Recap here are the steps you need to implement
      1) Compute a mask of the pixels from the source image to be copied
        The pixels that shouldn't be copied are completely white, they
        have R=255, G=255, B=255.  Any other pixels SHOULD be copied.

      2) Compute the interior and border regions of the mask.  An interior
         pixel has all 4 neighbors also inside the mask.  A border pixel is
         in the mask itself, but has at least one neighbor that isn't.
      
      3) Separate out the incoming image into three separate channels

      4) Create two float(!) buffers for each color channel that will
      act as our guesses.  Initialize them to the respective color
      channel of the source image since that will act as our intial guess.

      5) For each color channel perform the Jacobi iteration described 
         above 800 times.

      6) Create the output image by replacing all the interior pixels
         in the destination image with the result of the Jacobi iterations.
         Just cast the floating point values to unsigned chars since we have
         already made sure to clamp them to the correct range.
      

      Since this is final assignment we provide little boilerplate code to
      help you.  Notice that all the input/output pointers are HOST pointers.

      You will have to allocate all of your own GPU memory and perform your own
      memcopies to get data in and out of the GPU memory.

      Remember to wrap all of your calls with checkCudaErrors() to catch any
      thing that might go wrong.  After each kernel call do:

      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      to catch any errors that happened while executing the kernel.
  */

   size_t imageSize = numRowsSource*numColsSource*sizeof(uchar4);
     
   uchar4* d_sourceImg;
   uchar4* d_destImg;
   uchar4* d_finalImg;
   
   //first allocate memory for all of the images and copy them onto the GPU 
   checkCudaErrors(cudaMalloc(&d_sourceImg, imageSize));
   checkCudaErrors(cudaMalloc(&d_destImg, imageSize));
   checkCudaErrors(cudaMalloc(&d_finalImg, imageSize));
 
   checkCudaErrors(cudaMemcpy(d_sourceImg, h_sourceImg, imageSize, cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(d_destImg, h_destImg, imageSize, cudaMemcpyHostToDevice));
 
   // allocate memory for border/interior calculations
   size_t predSize = numRowsSource*numColsSource*sizeof(int);
   int* d_border;
   int* d_interior;
 
   checkCudaErrors(cudaMalloc(&d_border, predSize));
   checkCudaErrors(cudaMalloc(&d_interior, predSize));
 
   const dim3 blockSize(32, 32);
   const dim3 gridSize(numColsSource/blockSize.x + 1, numRowsSource/blockSize.y + 1);

   // 1) Compute a mask of the pixels from the source image to be copied
   //      The pixels that shouldn't be copied are completely white, they
   //      have R=255, G=255, B=255.  Any other pixels SHOULD be copied.

   // 2) Compute the interior and border regions of the mask.  An interior
   //      pixel has all 4 neighbors also inside the mask.  A border pixel is
   //      in the mask itself, but has at least one neighbor that isn't.
   
   computeMaskAndRegions<<<gridSize, blockSize>>>(
      d_sourceImg,
      d_border,
      d_interior,
      numRowsSource,
      numColsSource
   );
  
   cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

   size_t floatSize = numRowsSource*numColsSource*sizeof(float);
   float *d_sourceImgR, *d_sourceImgG, *d_sourceImgB; 
   float *d_destImgR,   *d_destImgG, 	*d_destImgB;
   
   // allocate memory for RGB channels
   checkCudaErrors(cudaMalloc(&d_sourceImgR, floatSize));
   checkCudaErrors(cudaMalloc(&d_sourceImgG, floatSize));
   checkCudaErrors(cudaMalloc(&d_sourceImgB, floatSize));
   
   checkCudaErrors(cudaMalloc(&d_destImgR, floatSize));
   checkCudaErrors(cudaMalloc(&d_destImgG, floatSize));
   checkCudaErrors(cudaMalloc(&d_destImgB, floatSize));
   
   // 3) Separate out the incoming image into three separate channels
   separateColorChannels<<<gridSize, blockSize>>>(
      d_sourceImg,
      d_sourceImgR,
      d_sourceImgG,
      d_sourceImgB,
      numRowsSource,
      numColsSource);

   cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

   separateColorChannels<<<gridSize, blockSize>>>(
      d_destImg,
      d_destImgR,
      d_destImgG,
      d_destImgB,
      numRowsSource,
      numColsSource);

   cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

   // 4) Create two float(!) buffers for each color channel that will
   //    act as our guesses.  Initialize them to the respective color
   //    channel of the source image since that will act as our intial guess.
   
   // allocate memory for the floats
   float *d_r0, *d_r1, *d_g0, *d_g1, *d_b0, *d_b1; 
   checkCudaErrors(cudaMalloc(&d_r0, floatSize));
   checkCudaErrors(cudaMalloc(&d_r1, floatSize));
   checkCudaErrors(cudaMalloc(&d_b0, floatSize));
   checkCudaErrors(cudaMalloc(&d_b1, floatSize));
   checkCudaErrors(cudaMalloc(&d_g0, floatSize));
   checkCudaErrors(cudaMalloc(&d_g1, floatSize));

   // initialize the buffers with the source image channels
   checkCudaErrors(cudaMemcpy(d_r0, d_sourceImgR, floatSize, cudaMemcpyDeviceToDevice));
   checkCudaErrors(cudaMemcpy(d_g0, d_sourceImgG, floatSize, cudaMemcpyDeviceToDevice));
   checkCudaErrors(cudaMemcpy(d_b0, d_sourceImgB, floatSize, cudaMemcpyDeviceToDevice));

   cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

   // 5) For each color channel perform the Jacobi iteration described 
   //    above 800 times.

   for(int i = 0; i < 800; i++) {

      //REDS
      jacobi<<<gridSize, blockSize>>>(
         d_r0, 
         d_r1,
         d_border,
         d_interior,
         d_sourceImgR,
         d_destImgR,
         numRowsSource,
         numColsSource
      );
      std::swap(d_r0, d_r1);

      //GREENS
      jacobi<<<gridSize, blockSize>>>(
         d_g0, 
         d_g1,
         d_border,
         d_interior,
         d_sourceImgG,
         d_destImgG,
         numRowsSource,
         numColsSource
      );
      std::swap(d_g0, d_g1);

      //BLUES
      jacobi<<<gridSize, blockSize>>>(
         d_b0, 
         d_b1,
         d_border,
         d_interior,
         d_sourceImgB,
         d_destImgB,
         numRowsSource,
         numColsSource
      );
      std::swap(d_b0, d_b1);
   }

   // 6) Create the output image by replacing all the interior pixels
   //      in the destination image with the result of the Jacobi iterations.
   //      Just cast the floating point values to unsigned chars since we have
   //      already made sure to clamp them to the correct range.
   
   createOutputImage<<<gridSize, blockSize>>>(
      d_finalImg,
      d_r0,
      d_g0,
      d_b0,
      numRowsSource,
      numColsSource
   );

   cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

   // copy final image to host
   checkCudaErrors(cudaMemcpy(h_blendedImg, d_finalImg, imageSize, cudaMemcpyDeviceToHost));

   // free everything up
   checkCudaErrors(cudaFree(d_sourceImg));
   checkCudaErrors(cudaFree(d_destImg));
   checkCudaErrors(cudaFree(d_finalImg));

   checkCudaErrors(cudaFree(d_sourceImgR));
   checkCudaErrors(cudaFree(d_sourceImgG));
   checkCudaErrors(cudaFree(d_sourceImgB));

   checkCudaErrors(cudaFree(d_destImgR));
   checkCudaErrors(cudaFree(d_destImgG));
   checkCudaErrors(cudaFree(d_destImgB));

   checkCudaErrors(cudaFree(d_r0));
   checkCudaErrors(cudaFree(d_r1));
   checkCudaErrors(cudaFree(d_g0));
   checkCudaErrors(cudaFree(d_g1));
   checkCudaErrors(cudaFree(d_b0));
   checkCudaErrors(cudaFree(d_b1));
}
