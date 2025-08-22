#include <torch/extension.h>
#include <vector>
#include "../include/spmm.h"

torch::Tensor spmm_csr_torch(torch::Tensor row_ptr, torch::Tensor col_idx,
                            torch::Tensor values, torch::Tensor dense_matrix,
                            int num_rows, int num_cols) {
    
    // Validate input tensors
    TORCH_CHECK(row_ptr.device().is_cuda(), "row_ptr must be on CUDA");
    TORCH_CHECK(col_idx.device().is_cuda(), "col_idx must be on CUDA");
    TORCH_CHECK(values.device().is_cuda(), "values must be on CUDA");
    TORCH_CHECK(dense_matrix.device().is_cuda(), "dense_matrix must be on CUDA");
    
    TORCH_CHECK(row_ptr.dtype() == torch::kInt32, "row_ptr must be int32");
    TORCH_CHECK(col_idx.dtype() == torch::kInt32, "col_idx must be int32");
    TORCH_CHECK(values.dtype() == torch::kFloat32, "values must be float32");
    TORCH_CHECK(dense_matrix.dtype() == torch::kFloat32, "dense_matrix must be float32");
    
    int dense_cols = dense_matrix.size(1);
    auto result = torch::zeros({num_rows, dense_cols}, 
                              torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    // Convert to CSRMatrix structure
    CSRMatrix sparse_mat;
    sparse_mat.num_rows = num_rows;
    sparse_mat.num_cols = num_cols;
    sparse_mat.nnz = values.size(0);
    
    // Get raw pointers
    int* row_ptr_data = row_ptr.data_ptr<int>();
    int* col_idx_data = col_idx.data_ptr<int>();
    float* values_data = values.data_ptr<float>();
    float* dense_data = dense_matrix.data_ptr<float>();
    float* result_data = result.data_ptr<float>();
    
    // Convert row_ptr to vector (needed for our existing interface)
    std::vector<int> row_ptr_vec(row_ptr_data, row_ptr_data + num_rows + 1);
    std::vector<int> col_idx_vec(col_idx_data, col_idx_data + sparse_mat.nnz);
    std::vector<float> values_vec(values_data, values_data + sparse_mat.nnz);
    
    sparse_mat.row_ptr = row_ptr_vec;
    sparse_mat.col_idx = col_idx_vec;
    sparse_mat.values = values_vec;
    
    // Call our CUDA implementation
    spmm_csr(sparse_mat, dense_data, dense_cols, result_data);
    
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("spmm_csr", &spmm_csr_torch, "Sparse matrix-dense matrix multiplication (CSR format)");
}