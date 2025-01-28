#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

int fact(int n)
{
    if(n==0)
        return 1;
    return n*fact(n-1);
}

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int arr[size];
    int num;

    if(rank==0)
    {
        printf("Enter %d numbers: ", size);
        for(int i=0; i<size; i++)
        {
            scanf("%d", &arr[i]);
        }
    }

    MPI_Scatter(arr, 1, MPI_INT, &num, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printf("Process %d received number: %d\n", rank, num);

    int f = fact(num);

    int *fact_arr=NULL;

    if(rank==0)
    {
        fact_arr = (int*)malloc(size*sizeof(int));
    }

    MPI_Gather(&f, 1, MPI_INT, fact_arr, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if(rank==0)
    {
        int sum=0;
        for(int i=0; i<size; i++)
        {
            printf("Process %d, factorial %d\n", i, fact_arr[i]);
            sum+=fact_arr[i];
        }
        printf("Sum of factorials: %d\n", sum);
    }

    MPI_Finalize();
    return 0;
}