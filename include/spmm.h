#ifndef SPMM_CUDA_H
#define SPMM_CUDA_H

#include <cuda_runtime.h>
#include <vector>

struct CSRMatrix {
    int num_rows;
    int num_cols;
    int nnz;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<float> values;
};

// Basic SpMM function
void spmm_csr(const CSRMatrix& sparse_mat, const float* dense_matrix, 
              int dense_cols, float* result, int threads_per_block = 256);

// Optimized SpMM with shared memory
void spmm_csr_optimized(const CSRMatrix& sparse_mat, const float* dense_matrix,
                       int dense_cols, float* result, int tile_size = 32);

// Matrix generation utilities
CSRMatrix generate_random_sparse_matrix(int rows, int cols, float sparsity);
void generate_random_dense_matrix(float* matrix, int rows, int cols);

// Validation functions
bool validate_results(const float* result_cuda, const float* result_cpu, 
                     int rows, int cols, float tolerance = 1e-6);

// Performance measurement
double measure_performance(const CSRMatrix& sparse_mat, const float* dense_matrix,
                          int dense_cols, float* result, int num_runs = 10);

#endif