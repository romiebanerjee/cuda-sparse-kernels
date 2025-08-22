# CUDA Sparse Matrix-Dense Matrix Multiplication

A high-performance CUDA implementation of sparse matrix-dense matrix multiplication (SpMM) using CSR format.

## Features

- CSR format sparse matrix multiplication
- Multiple optimization strategies
- Performance benchmarking
- Validation utilities
- Easy-to-use API

## Building

```bash
mkdir build
cd build
cmake ..
make
```

## Usage
```bash
#include "spmm.h"

CSRMatrix sparse_mat = generate_random_sparse_matrix(1000, 1000, 0.01);
float* dense_matrix = new float[1000 * 50];
float* result = new float[1000 * 50];

spmm_csr(sparse_mat, dense_matrix, 50, result);
```