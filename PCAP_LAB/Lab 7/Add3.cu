#include<stdio.h>
#include<string.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

__global__ void format(char *in, int len,char *out,int *p)
{
  int idx=blockIdx.x*blockDim.x + threadIdx.x;

  if(idx<len)
  {
    int pos=atomicAdd(p,idx+1);

    for(int i=0;i<idx+1;i++)
    {
      out[pos+i]=in[idx];
    }
    printf("Thread %d: Appended %c at position %d for %d times\n", idx, in[idx], pos,idx+1);
  }
}

int main()
{
  char Sin[50],T[50];
  printf("Enter a string: ");
  scanf("%s",Sin);
  int len=strlen(Sin);
  int outlen=len*(len+1);

  char *d_out,*d_in;
  int *p;

  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);
  cudaMalloc((void **)&d_in,len);
  cudaMalloc((void **)&d_out,outlen);
  cudaMalloc((void **)&p,sizeof(int));

  cudaMemcpy(d_in,Sin,len,cudaMemcpyHostToDevice);
  cudaMemset(p,0,sizeof(int));
  cudaMemset(d_out,0,outlen);

  format<<<1,len>>>(d_in,len,d_out,p);

  cudaError_t error=cudaGetLastError();
  if(error!=cudaSuccess)
  {
    printf("CUDA Error: %s\n",cudaGetErrorString(error));
  }
  
  cudaEventRecord(stop,0);
  float lapsed;
  cudaEventElapsedTime(&lapsed,start,stop);

  cudaMemcpy(T,d_out,outlen,cudaMemcpyDeviceToHost);
  printf("Final Result: %s\n",T);
  printf("Time Lapsed: %f\n",lapsed);

  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(p);

  return 0;
}
