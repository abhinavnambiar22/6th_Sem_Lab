#include <stdio.h>
#include <cuda_runtime.h>

#define FILTER_SIZE 7  // Filter size
#define BLOCK_SIZE 256 // Threads per block

// Constant memory for filter kernel
__constant__ float d_filter[FILTER_SIZE];

// 1D Convolution Kernel using constant memory and shared memory
__global__ void conv1D(float *d_input, float *d_output, int input_size, int filter_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int half_filter = filter_size / 2;

    // Shared memory tile (for input caching)
    __shared__ float tile[BLOCK_SIZE + FILTER_SIZE - 1];

    // Global index for input data
    int global_index = tid - half_filter;

    // Load data into shared memory with boundary handling
    if (global_index >= 0 && global_index < input_size)
        tile[threadIdx.x] = d_input[global_index];
    else
        tile[threadIdx.x] = 0.0f;

    // Load extra elements for boundary cases
    if (threadIdx.x < filter_size - 1) {
        int extra_index = tid + BLOCK_SIZE - half_filter;
        if (extra_index >= 0 && extra_index < input_size)
            tile[threadIdx.x + BLOCK_SIZE] = d_input[extra_index];
        else
            tile[threadIdx.x + BLOCK_SIZE] = 0.0f;
    }

    __syncthreads(); // Synchronize threads

    // Compute convolution if within bounds
    if (tid < input_size) {
        float sum = 0.0f;
        for (int i = 0; i < filter_size; i++) {
            sum += tile[threadIdx.x + i] * d_filter[i];
        }
        d_output[tid] = sum;

        // Print thread and block info
        printf("Block (%d), Thread (%d) -> Output[%d] = %.2f\n", blockIdx.x, threadIdx.x, tid, sum);
    }
}

// Function to measure execution time
float measureExecutionTime(float *d_input, float *d_output, int input_size, int filter_size, int grid_size) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    conv1D<<<grid_size, BLOCK_SIZE>>>(d_input, d_output, input_size, filter_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

void getInput(float *array, int size, const char *name) {
    printf("Enter %d elements for %s:\n", size, name);
    for (int i = 0; i < size; i++) {
        scanf("%f", &array[i]);
    }
}

void printArray(float *array, int size, const char *name) {
    printf("%s:\n", name);
    for (int i = 0; i < size; i++) {
        printf("%.2f ", array[i]);
    }
    printf("\n");
}

int main() {
    int input_size, filter_size = FILTER_SIZE;

    printf("Enter input size: ");
    scanf("%d", &input_size);

    int size_input = input_size * sizeof(float);
    int size_output = input_size * sizeof(float);
    int size_filter = filter_size * sizeof(float);

    // Allocate host memory
    float *h_input = (float *)malloc(size_input);
    float *h_output = (float *)malloc(size_output);
    float h_filter[FILTER_SIZE];

    // Get user input
    getInput(h_input, input_size, "Input");
    getInput(h_filter, filter_size, "Filter");

    // Copy filter to constant memory
    cudaMemcpyToSymbol(d_filter, h_filter, size_filter);

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, size_input);
    cudaMalloc((void **)&d_output, size_output);

    // Copy input to device
    cudaMemcpy(d_input, h_input, size_input, cudaMemcpyHostToDevice);

    // Grid size calculation
    int grid_size = (input_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Measure optimized kernel execution time
    float optimized_time = measureExecutionTime(d_input, d_output, input_size, filter_size, grid_size);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, size_output, cudaMemcpyDeviceToHost);

    // Print output
    printArray(h_output, input_size, "Output");

    // Print execution time
    printf("Optimized Kernel Execution Time: %.3f ms\n", optimized_time);

    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
