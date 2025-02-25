#include<stdio.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

#define MAX_LEN 50  // Maximum length of each binary string

__global__ void one_comp(char *b_arr, char *one, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        char *src = b_arr + idx * MAX_LEN;  // Pointer to input string
        char *dst = one + idx * MAX_LEN;    // Pointer to output string

        int i;
        for (i = 0; i < MAX_LEN - 1 && src[i] != '\0'; i++)  // Process characters
        {
            dst[i] = (src[i] == '0') ? '1' : '0';
        }
        dst[i] = '\0';  // Null-terminate manually

        printf("Thread %d: bin[%d] = %s, one_comp[%d] = %s\n", idx, idx, src, idx, dst);
    }
}

int main()
{
    int N;
    printf("Enter number of elements: ");
    scanf("%d", &N);

    char h_b[N][MAX_LEN], h_one[N][MAX_LEN];  // Host arrays
    char *d_b, *d_one;  // Device pointers
    size_t size = N * MAX_LEN * sizeof(char);

    printf("Enter binary array elements:\n");
    for (int i = 0; i < N; i++)
    {
        scanf("%s", h_b[i]);
    }

    // Allocate memory on GPU
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_one, size);

    // Copy input strings to device
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Launch CUDA kernel
    one_comp<<<1, N>>>(d_b, d_one, N);

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(h_one, d_one, size, cudaMemcpyDeviceToHost);

    printf("One's complement array elements:\n");
    for (int i = 0; i < N; i++)
        printf("%s\n", h_one[i]);

    // Free GPU memory
    cudaFree(d_b);
    cudaFree(d_one);

    return 0;
}
