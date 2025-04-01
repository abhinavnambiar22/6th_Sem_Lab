#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16  // Block size (BLOCK_SIZE x BLOCK_SIZE)

// CUDA kernel for matrix multiplication
__global__ void matrixMulKernel(float *A, float *B, float *C, int M, int K, int N) {
    // Compute row and column index for C
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    if (row < M && col < N) {
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }

    // Synchronize threads within the block
    __syncthreads();

    // Print thread and block information along with matrix result
    if (row < M && col < N) {
        printf("Block (%d, %d), Thread (%d, %d) -> C[%d][%d] = %.2f\n",
               blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, row, col, sum);
    }
}

// Function to take user input for a matrix
void getMatrixFromUser(float *matrix, int rows, int cols, const char *name) {
    printf("Enter elements of matrix %s (%d x %d) row-wise:\n", name, rows, cols);
    for (int i = 0; i < rows * cols; i++) {
        scanf("%f", &matrix[i]);
    }
}

// Function to print the matrix
void printMatrix(float *matrix, int rows, int cols, const char *name) {
    printf("Matrix %s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%6.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

// Host function to launch the kernel
void matrixMultiplication(int M, int K, int N) {
    int sizeA = M * K * sizeof(float);
    int sizeB = K * N * sizeof(float);
    int sizeC = M * N * sizeof(float);

    // Allocate memory on host
    float *h_A = (float *)malloc(sizeA);
    float *h_B = (float *)malloc(sizeB);
    float *h_C = (float *)malloc(sizeC);

    // Get matrices from user
    getMatrixFromUser(h_A, M, K, "A");
    getMatrixFromUser(h_B, K, N, "B");

    // Allocate memory on GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, sizeA);
    cudaMalloc((void **)&d_B, sizeB);
    cudaMalloc((void **)&d_C, sizeC);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel
    matrixMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // Print output matrix
    printMatrix(h_C, M, N, "C (Result)");

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
}

int main() {
    int M, K, N;

    // Get matrix dimensions from user
    printf("Enter number of rows for Matrix A (M): ");
    scanf("%d", &M);
    printf("Enter number of columns for Matrix A / rows for Matrix B (K): ");
    scanf("%d", &K);
    printf("Enter number of columns for Matrix B (N): ");
    scanf("%d", &N);

    if (M <= 0 || K <= 0 || N <= 0) {
        printf("Invalid matrix dimensions. Exiting...\n");
        return 1;
    }

    // Perform matrix multiplication
    matrixMultiplication(M, K, N);

    printf("Matrix multiplication completed successfully.\n");
    return 0;
}
