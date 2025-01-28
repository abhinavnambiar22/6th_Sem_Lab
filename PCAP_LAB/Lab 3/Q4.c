#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>


int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char s1[100], s2[100];
    int l1,l2;
    int chunk_size;

    if(rank==0)
    {
        printf("Enter string 1: ");
        scanf("%s", s1);
        printf("Enter string 2: ");
        scanf("%s", s2);

        l1 = strlen(s1);
        l2 = strlen(s2);

        if(l1 != l2)
        {
            printf("Error: String lengths must be equal.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if(l1 % size != 0)
        {
            printf("Error: String length must be evenly divisible by the number of processes.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        chunk_size = l1 / size;
        printf("String length: %d, Chunk size: %d\n", l1, chunk_size);
    }

    // Broadcast the string length and chunk size to all processes
    MPI_Bcast(&l1, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&chunk_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    char *chunk1 = (char*)malloc((chunk_size + 1) * sizeof(char));
    char *chunk2 = (char*)malloc((chunk_size + 1) * sizeof(char));
    char *chunk_res=(char*)malloc((chunk_size*2 + 1) * sizeof(char));
    memset(chunk_res, 0, chunk_size*2 + 1);

    MPI_Scatter(s1, chunk_size, MPI_CHAR, chunk1, chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);
    chunk1[chunk_size] = '\0';

    MPI_Scatter(s2, chunk_size, MPI_CHAR, chunk2, chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);
    chunk2[chunk_size] = '\0';

    printf("Process %d received chunk 1: %s\n", rank, chunk1);
    printf("Process %d received chunk 2: %s\n", rank, chunk2);

    for(int i=0;i<chunk_size;i++)
    {
        chunk_res[i*2]=chunk1[i];
        chunk_res[i*2+1]=chunk2[i];
    }
    printf("Process %d sent chunk result: %s\n", rank, chunk_res);

    char *result=NULL;
    if(rank==0)
    {
        result=(char*)malloc((l1*2 + 1) * sizeof(char));
        memset(result, 0, l1*2 + 1);
    }
    MPI_Gather(chunk_res, chunk_size*2, MPI_CHAR, result, chunk_size*2, MPI_CHAR, 0, MPI_COMM_WORLD);

    if(rank==0)
    {
        printf("Final result: %s\n", result);
        free(result);
    }

    free(chunk1);
    free(chunk2);
    free(chunk_res);

    MPI_Finalize();
    return 0;
}