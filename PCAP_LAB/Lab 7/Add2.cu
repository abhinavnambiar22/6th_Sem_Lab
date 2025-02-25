#include<stdio.h>
#include<string.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

__global__ void conc(char *in,int len,int N, char *out,int *p)
{
  int idx=blockIdx.x*blockDim.x + threadIdx.x;

  if(idx<N)
  {
    int pos=atomicAdd(p,len);

    for(int i=0;i<len;i++)
    {
      out[pos+i]=in[i];
    }
    printf("Thread %d: Wrote %s at position %d\n", idx, in, pos);
  }
}

int main()
{
  char Sin[50],Sout[50];
  printf("Enter a string: ");
  scanf("%s",Sin);

  int N;
  printf("Enter replication factor: ");
  scanf("%d",&N);

  int len=strlen(Sin);
  char *d_out,*d_in;
  int *p;

  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);
  cudaMalloc((void **)&d_in,len);
  cudaMalloc((void **)&d_out,N*len+1);
  cudaMalloc((void **)&p,sizeof(int));

  cudaMemcpy(d_in,Sin,len,cudaMemcpyHostToDevice);
  cudaMemset(p,0,sizeof(int));
  cudaMemset(d_out,0,N*len+1);

  conc<<<1,N>>>(d_in,len,N,d_out,p);

  cudaError_t error=cudaGetLastError();
  if(error!=cudaSuccess)
  {
    printf("CUDA Error: %s\n",cudaGetErrorString(error));
  }
  
  cudaEventRecord(stop,0);
  float lapsed;
  cudaEventElapsedTime(&lapsed,start,stop);

  cudaMemcpy(Sout,d_out,N*len+1,cudaMemcpyDeviceToHost);
  printf("Final Result: %s\n",Sout);
  printf("Time Lapsed: %f\n",lapsed);

  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(p);

  return 0;
}
