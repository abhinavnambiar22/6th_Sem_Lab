#include<stdio.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

__global__ void octal(int *a,int *o,int N)
{
  int ele,val,step;
  int idx=blockDim.x*blockIdx.x + threadIdx.x;
  if(idx<N)
  {
    ele=a[idx];
    val=0;
    step=1;
    do
    {

      val+=ele%8*step;
      step*=10;
      ele/=8;

    }while(ele!=0);
    printf("Thread %d: arr[%d] = %d, octal[%d] = %d\n",idx,idx,a[idx],idx,val);
    o[idx]=val;
  }
}

int main()
{
  int N;
  printf("Enter number of elements: ");
  scanf("%d",&N);

  int *h_a,*h_o;
  int *d_a,*d_o;
  size_t size=N*sizeof(int);

  h_a=(int *)malloc(size);
  h_o=(int *)malloc(size);

  printf("Enter array elements: ");
  for(int i=0;i<N;i++)
  scanf("%d",&h_a[i]);

  cudaMalloc((void **)&d_a,size);
  cudaMalloc((void **)&d_o,size);

  cudaMemcpy(d_a,h_a,size,cudaMemcpyHostToDevice);

  octal<<<1,N>>>(d_a,d_o,N);
  cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }
  cudaDeviceSynchronize();

  cudaMemcpy(h_o,d_o,size,cudaMemcpyDeviceToHost);

  printf("Octal array elements: ");
  for(int i=0;i<N;i++)
  printf("%d ",h_o[i]);
  printf("\n");

  free(h_a);
  free(h_o);
  cudaFree(d_a);
  cudaFree(d_o);

  return 0;
}
