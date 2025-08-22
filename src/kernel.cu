#include "spmm.h"
#include <cub/cub.cuh>

#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void spmv_csr_kernel(int num_rows, const int* row_ptr, const int* col_idx,
                               const float* values, const float* dense_matrix,
                               float* result, int dense_cols, int dense_lda) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < num_rows) {
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];
        
        for (int col = 0; col < dense_cols; col++) {
            float sum = 0.0f;
            
            for (int j = row_start; j < row_end; j++) {
                int sparse_col = col_idx[j];
                float sparse_val = values[j];
                float dense_val = dense_matrix[sparse_col * dense_lda + col];
                sum += sparse_val * dense_val;
            }
            
            result[row * dense_cols + col] = sum;
        }
    }
}

template<int BLOCK_SIZE, int TILE_SIZE>
__global__ void spmm_csr_optimized_kernel(int num_rows, const int* row_ptr, 
                                         const int* col_idx, const float* values,
                                         const float* dense_matrix, float* result,
                                         int dense_cols, int dense_lda) {
    __shared__ float dense_tile[TILE_SIZE][BLOCK_SIZE];
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col_tile = blockIdx.y * TILE_SIZE;
    
    if (row < num_rows) {
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];
        
        float row_result[TILE_SIZE] = {0.0f};
        
        for (int elem = row_start; elem < row_end; elem++) {
            int sparse_col = col_idx[elem];
            float sparse_val = values[elem];
            
            if (threadIdx.x < TILE_SIZE && col_tile + threadIdx.x < dense_cols) {
                for (int i = 0; i < BLOCK_SIZE; i++) {
                    int load_col = col_tile + threadIdx.x;
                    dense_tile[threadIdx.x][i] = dense_matrix[sparse_col * dense_lda + load_col];
                }
            }
            __syncthreads();
            
            for (int i = 0; i < TILE_SIZE; i++) {
                if (col_tile + i < dense_cols) {
                    row_result[i] += sparse_val * dense_tile[i][threadIdx.x];
                }
            }
            __syncthreads();
        }
        
        for (int i = 0; i < TILE_SIZE; i++) {
            if (col_tile + i < dense_cols) {
                result[row * dense_cols + col_tile + i] = row_result[i];
            }
        }
    }
}