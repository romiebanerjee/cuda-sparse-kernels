#include "spmm.h"
#include <iostream>
#include <cstdlib>

void spmm_csr(const CSRMatrix& sparse_mat, const float* dense_matrix, 
              int dense_cols, float* result, int threads_per_block) {
    
    // Device pointers
    int *d_row_ptr, *d_col_idx;
    float *d_values, *d_dense, *d_result;
    
    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_row_ptr, (sparse_mat.num_rows + 1) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_col_idx, sparse_mat.nnz * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_values, sparse_mat.nnz * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_dense, sparse_mat.num_cols * dense_cols * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_result, sparse_mat.num_rows * dense_cols * sizeof(float)));
    
    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_row_ptr, sparse_mat.row_ptr.data(), 
                               (sparse_mat.num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_col_idx, sparse_mat.col_idx.data(), 
                               sparse_mat.nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_values, sparse_mat.values.data(), 
                               sparse_mat.nnz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_dense, dense_matrix, 
                               sparse_mat.num_cols * dense_cols * sizeof(float), cudaMemcpyHostToDevice));
    
    // Configure kernel launch
    dim3 block_size(threads_per_block);
    dim3 grid_size((sparse_mat.num_rows + block_size.x - 1) / block_size.x);
    
    // Launch kernel
    spmv_csr_kernel<<<grid_size, block_size>>>(sparse_mat.num_rows, d_row_ptr, d_col_idx, 
                                              d_values, d_dense, d_result, 
                                              dense_cols, dense_cols);
    
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(result, d_result, 
                               sparse_mat.num_rows * dense_cols * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_row_ptr));
    CHECK_CUDA_ERROR(cudaFree(d_col_idx));
    CHECK_CUDA_ERROR(cudaFree(d_values));
    CHECK_CUDA_ERROR(cudaFree(d_dense));
    CHECK_CUDA_ERROR(cudaFree(d_result));
}