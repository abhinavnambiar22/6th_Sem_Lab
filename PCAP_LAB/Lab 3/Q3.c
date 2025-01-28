#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// Function to check if a character is a vowel
int is_vowel(char c) {
    c = tolower(c);
    return (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u');
}

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char* input_string = NULL;
    int string_length;
    int chunk_size;

    if (rank == 0) {
        // Root process reads the input string
        input_string = (char*)malloc(100 * sizeof(char));
        printf("Enter a string: ");
        scanf("%s", input_string);

        string_length = strlen(input_string);
        if (string_length % size != 0) {
            printf("Error: String length must be evenly divisible by the number of processes.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        chunk_size = string_length / size;
        printf("String length: %d, Chunk size: %d\n", string_length, chunk_size);
    }

    // Broadcast the string length and chunk size to all processes
    MPI_Bcast(&string_length, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&chunk_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate memory for each process's chunk
    char* chunk = (char*)malloc((chunk_size + 1) * sizeof(char));

    // Scatter the string to all processes
    MPI_Scatter(input_string, chunk_size, MPI_CHAR, chunk, chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);
    chunk[chunk_size] = '\0'; // Null-terminate the chunk

    printf("Process %d received chunk: %s\n", rank, chunk);

    // Each process counts the number of non-vowels in its chunk
    int local_non_vowel_count = 0;
    for (int i = 0; i < chunk_size; i++) {
        if (!is_vowel(chunk[i])) {
            local_non_vowel_count++;
        }
    }

    // Gather the non-vowel counts from all processes to the root
    int* non_vowel_counts = NULL;
    if (rank == 0) {
        non_vowel_counts = (int*)malloc(size * sizeof(int));
    }

    MPI_Gather(&local_non_vowel_count, 1, MPI_INT, non_vowel_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Root process prints the results
    if (rank == 0) {
        int total_non_vowels = 0;
        printf("Non-vowel counts from each process:\n");
        for (int i = 0; i < size; i++) {
            printf("Process %d: %d\n", i, non_vowel_counts[i]);
            total_non_vowels += non_vowel_counts[i];
        }
        printf("Total number of non-vowels: %d\n", total_non_vowels);
        free(non_vowel_counts);
        free(input_string);
    }

    free(chunk);
    MPI_Finalize();
    return 0;
}
