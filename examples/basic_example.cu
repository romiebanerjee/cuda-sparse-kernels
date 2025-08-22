#include "../include/spmm.h"
#include <iostream>

int main() {
    // Create a simple sparse matrix in CSR format
    CSRMatrix sparse_mat;
    sparse_mat.num_rows = 3;
    sparse_mat.num_cols = 4;
    sparse_mat.nnz = 5;
    sparse_mat.row_ptr = {0, 2, 3, 5};
    sparse_mat.col_idx = {0, 2, 1, 0, 3};
    sparse_mat.values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    
    // Create dense matrix
    int dense_cols = 2;
    float dense[] = {1.0f, 2.0f,
                    3.0f, 4.0f,
                    5.0f, 6.0f,
                    7.0f, 8.0f};
    
    float result[6] = {0}; // 3x2 result matrix
    
    // Perform multiplication
    spmm_csr(sparse_mat, dense, dense_cols, result);
    
    // Print results
    std::cout << "Result matrix:" << std::endl;
    for (int i = 0; i < sparse_mat.num_rows; i++) {
        for (int j = 0; j < dense_cols; j++) {
            std::cout << result[i * dense_cols + j] << " ";
        }
        std::cout << std::endl;
    }
    
    return 0;
}