#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include <stdio.h>
#include <string.h>

#define THREADS_PER_BLOCK 29

__device__ bool is_word_match(char *sentence, int index, char *word, int word_len) {
    for (int i = 0; i < word_len; i++) {
        if (sentence[index + i] != word[i]) {
            return false;
        }
    }
    return (sentence[index + word_len] == ' ' || sentence[index + word_len] == '\0');
}

__global__ void count_word_occurrences(char *sentence, char *word, int *count, int sent_len, int word_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < sent_len) {
        if ((idx == 0 || sentence[idx - 1] == ' ') && is_word_match(sentence, idx, word, word_len)) {
          printf("Block %d, Thread %d, Count of word %s found at index %d\n",blockIdx.x,threadIdx.x,word,idx);
            atomicAdd(count, 1);
        }
    }
}

int main() {
    char h_sentence[] = "cuda is fast and cuda is powerful but cuda needs knowledge";
    char h_word[] = "cuda";
    int h_count = 0;

    printf("Sentence: %s\nWord: %s\n",h_sentence,h_word);
    char *d_sentence, *d_word;
    int *d_count;
    int sent_len = strlen(h_sentence);
    printf("Sentence_Length: %d\n",sent_len);
    int word_len = strlen(h_word);

    cudaMalloc((void **)&d_sentence, sent_len + 1);
    cudaMalloc((void **)&d_word, word_len + 1);
    cudaMalloc((void **)&d_count, sizeof(int));

    cudaMemcpy(d_sentence, h_sentence, sent_len + 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_word, h_word, word_len + 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_count, &h_count, sizeof(int), cudaMemcpyHostToDevice);

    int num_blocks = (sent_len + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    printf("Number of blocks: %d\n",num_blocks);
    printf("Number of threads per block: %d\n",THREADS_PER_BLOCK);
    count_word_occurrences<<<num_blocks, THREADS_PER_BLOCK>>>(d_sentence, d_word, d_count, sent_len, word_len);

    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    printf("The word '%s' appears %d times in the sentence.\n", h_word, h_count);

    cudaFree(d_sentence);
    cudaFree(d_word);
    cudaFree(d_count);
    
    return 0;
}
