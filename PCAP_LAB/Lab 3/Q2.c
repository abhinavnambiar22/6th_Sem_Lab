// 

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int m, *mat = NULL, *row = NULL;

    if (rank == 0) {
        // Input matrix dimensions and elements
        printf("Enter the value of m: ");
        scanf("%d", &m);
        mat = (int*)malloc(m * size * sizeof(int));
        printf("Enter %d elements: ", m * size);
        for (int i = 0; i < m * size; i++) {
            scanf("%d", &mat[i]);
        }
    }

    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate memory for each process's row
    row = (int*)malloc(m * sizeof(int));

    // Scatter rows of the matrix to all processes
    MPI_Scatter(mat, m, MPI_INT, row, m, MPI_INT, 0, MPI_COMM_WORLD);

    // Display the received row
    printf("Process %d received row: ", rank);
    for (int i = 0; i < m; i++) {
        printf("%d ", row[i]);
    }
    printf("\n");

    // Compute the average of the received row
    float avg = 0.0;
    for (int i = 0; i < m; i++) {
        avg += row[i];
    }
    avg /= m;

    // Gather all averages to the root process
    float* avgarr = NULL;
    if (rank == 0) {
        avgarr = (float*)malloc(size * sizeof(float));
    }
    MPI_Gather(&avg, 1, MPI_FLOAT, avgarr, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Root process calculates the total average
    if (rank == 0) {
        float total_avg = 0.0;
        printf("Averages received from each process:\n");
        for (int i = 0; i < size; i++) {
            printf("Process %d, average: %f\n", i, avgarr[i]);
            total_avg += avgarr[i];
        }
        total_avg /= size;
        printf("Sum of averages: %f\n", total_avg);

        // Free memory on root process
        free(mat);
        free(avgarr);
    }

    // Free memory on all processes
    free(row);

    MPI_Finalize();
    return 0;
}
