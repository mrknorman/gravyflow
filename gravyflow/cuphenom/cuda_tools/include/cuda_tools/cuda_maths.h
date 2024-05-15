#ifndef CUDA_MATHS_H
#define CUDA_MATHS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cufft.h>
#include <complex.h> 

#define TEST_LENGTH 10000000
#define MAX_ERR 1e-5

#define CUDA_CALL(e) do { if((e)!=cudaSuccess) { \
    printf("Error: %s at %s:%d\n",cudaGetErrorString(e),__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CUFFT_CALL(x) do { if((x)!=CUFFT_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

#ifndef FLOAT16_2
typedef struct {
	uint16_t x; 
	uint16_t y;
} float16_2_t;
#endif

cudaError_t CUDARTAPI cudaFree(void *devPtr);

int32_t cudaToHost16(
    const float16_2_t  *array_g, 
	const int32_t       num_elements,
          float       **ret_array
    );

int32_t cudaToDevice(
    const void     *array, 
    const size_t    element_size,
	const int32_t   num_elements,
          void    **ret_array_g
    );

int32_t cudaToHost(
    const void     *array_g, 
    const size_t    element_size,
	const int32_t   num_elements,
          void    **ret_array
    );

int32_t cudaPrintIntArray(
    const char     *title,
    const int32_t  *array_g,
    const int32_t   num_elements
    );

int32_t cudaPrintArray(
    const char    *title,
    const float   *array_g,
    const int32_t  num_elements
    );

int32_t cudaPrintArray64(
    const char    *title,
    const double   *array_g,
    const int32_t  num_elements
    );

void printTestResult(
    const int32_t  pass,
    const char    *test_name
    );

int32_t hostCudaMemCpy(
		  float   *array_1, 
		  float   *array_2,  
	const int32_t  num_elements
    );

int32_t hostCudaMemInsert(
		  void    *array_1, 
		  void    *array_2,  
    const size_t   element_size,
	const int32_t  num_elements
    );

int32_t hostCudaAddValue(
          float   *array, 
          double   value, 
    const int32_t  num_elements
    );

int32_t testCudaAdd(
    const int32_t verbosity
    );

int32_t testCudaSubtract(
    const int32_t verbosity
    );

int32_t hostCudaSubtract(
          float   *array_1, 
          float   *array_2, 
    const int32_t  num_elements
    );

int32_t hostCudaMultiply(
          float   *array_1, 
          float   *array_2, 
    const int32_t  num_elements
    );

int32_t hostCudaMultiply16(
          float16_2_t *array_1, 
          float16_2_t *array_2, 
    const int32_t      num_elements
    );

int32_t hostCudaMultiplyByValue(
          float   *array, 
          double   value, 
    const int32_t  num_elements
    );

int32_t testCudaMultiply(
    const int32_t verbosity
    );

int32_t testCudaDivide(
    const int32_t verbosity
    );

int32_t testCudaDivideByValue(
    const int32_t verbosity
    );

int32_t hostCudaACosF(
          float   *array, 
    const int32_t  num_elements
    );

int32_t initCurandGenerator(
    const int64_t  seed,
          void    *generator
    );

int32_t cudaGenerateRandomArray(
    const int32_t  num_elements,
          void    *generator,
          float   *array_g
    );

int32_t cudaGenerateRandomWeightedBoolArray(
    const int32_t  num_elements,
          void    *generator,
          double   weight,
          float   *array
    );

int32_t cudaGenerateRandomSignArray(
    const int32_t  num_elements,
          void    *generator,
          float   *array
    );

int32_t cudaGenerateRandomArrayBetween(
    const int32_t  num_elements,
          void    *generator,
          double   min,
          double   max,
          float   *array_g
    );

int32_t cudaGenerateRandomArrayBetweenArrays(
    const int32_t  num_elements,
          void    *generator,
          float   *min_array,
          float   *max_array,
          float   *array_g
    );

int32_t cudaGenerateRandomLogArrayBetween(
    const int32_t  num_elements,
          void    *generator,
          double   min,
          double   max,
          float   *array_g
    );

int32_t cudaGenerateRandomIntArrayBetween(
    const int32_t  num_elements,
          void    *generator,
          double   min,
          double   max,
          float   *array
    );

int32_t castToIntHost(
          float    *array_g,
    const int32_t   num_elements,
          int32_t **ret_array_g
    );

void cudaSetArrayValueIntHost(
          int32_t *array, 
	const int32_t  value,
    const int32_t  num_elements
    );

int32_t returnFloatArrayToHost(
    const float    *array_g,
    const int32_t   num_elements,
          float   **ret_array
    );

int32_t returnArrayToHost(
    const float    *array_g,
    const int32_t   num_elements,
          float   **ret_array
    );

int32_t returnIntArrayToHost(
    const int32_t  *array_g,
    const int32_t   num_elements,
          int32_t **ret_array
    );

int32_t cudaAllocateDeviceMemory(
	const size_t    size,
    const int32_t   num_elements,
          void    **ret_array
    );

int32_t cudaCallocateDeviceMemory(
	const size_t    size,
    const int32_t   num_elements,
          void    **ret_array
    );

int32_t cudaAllocateIntDeviceMemory(
    const int32_t   num_elements,
          int32_t **ret_array
    );
    
int32_t cudaGenerateRandomNormalArray(
    const int32_t  num_elements,
          void    *generator,
		  float    mean,
		  float    std_deviation,
          float   *array_g

    );

int32_t cudaGenerateNormalIntArray(
    const int32_t   num_elements,
          void     *generator,
          double     mean,
          double     std_deviation,
          double     min,
          double     max,
          float     *array
    );

int32_t cudaNormaliseArrayMulti16(
          float16_2_t *array,
    const int32_t      num_segments,
    const int32_t      num_elements_per_segment,
    const int32_t      total_num_elements
    );

int32_t testCudaGenerateRandomArray(
    const int32_t verbosity
    );

int32_t cudaRfft(
    const int32_t         num_elements,
    const int32_t         num_transforms,
          cuFloatComplex *array
    );

int32_t cudaIRfft(
    const int32_t         num_elements,
    const int32_t         num_transforms,
	      double          normalisation_factor,
          cuFloatComplex *input,
          cufftReal*     *output
    );

int32_t cudaIRfft64(
    const int32_t          num_elements,
    const int32_t          num_transforms,
    const double           normalisation_factor,
          cuDoubleComplex *data
    );

int32_t cudaInterlacedIRFFT(
    const int32_t          num_elements,
	      double           normalisation_factor,
          cuFloatComplex  *input,
          float          **ret_output
    );

int32_t testCudaRfft(
    const int32_t verbosity
    );

void generateHammingWindHost(
    const int32_t   intervals, 
          float   **data_ret
    );

int32_t testCudaGenerateHammingWindow(
    const int32_t verbosity
    );

void cudaSetArrayValueHost(
          float   *array, 
	const float    value,
    const int32_t  num_elements
    );

int32_t testCudaSetArrayValue(
    const int32_t verbosity
    );

int32_t cudaSumArray(
          float   *array, 
    const int32_t  num_elements,
          float   *ret_sum
	);

void cudaZeroArray(
          void    *array, 
	const size_t   size,
    const int32_t  num_elements
    );

void cudaFoldArray(
          float   *array, 
    const int32_t  num_elements
    );

void cudaFoldArrayC(
          void    *array, 
    const int32_t  num_elements
    );

int32_t testCudaFoldArray(
    const int32_t verbosity
    );

int32_t testCudaCExpf(
    const int32_t verbosity
    );

float cudaFindMaxHost(
          float   *array , 
    const int32_t  num_elements_o
	);

int32_t testCudaFindMax(
    const int32_t verbosity
    );

float cudaFindAbsMaxMulti(
          float   *array, 
          float   *max_array,
    const int32_t  num_segments,
    const int32_t  num_elements_per_segment,
    const int32_t  total_num_elements
	);

float cudaFindArgAbsMaxMulti(
          float   *array, 
          float   *arg_max_array,
    const int32_t  num_segments,
    const int32_t  num_elements_per_segment,
    const int32_t  total_num_elements
	);

int32_t cudaNormaliseArrayMulti(
          float   *array,
    const int32_t  num_segments,
    const int32_t  num_elements_per_segment,
    const int32_t  total_num_elements
    );

float cudaFindAbsMax(
          float   *array , 
    const int32_t  num_elements_o
	);

int32_t testCudaFindAbsMax(
    const int32_t verbosity
    );

void cudaResample(
	const int32_t   old_num_elements,
	const int32_t   new_num_elements,
	const int32_t   max_num_elements,
	const int32_t   block_size,
		  void     *window, 
		  void     *array
	);

int32_t cudaCallocateDeviceMemoryInt(
    const int32_t   num_elements,
          int32_t **ret_array
    );

void cudaSetArrayValueInt(
          int32_t *array, 
	const int32_t  value,
    const int32_t  num_elements
    );

#endif