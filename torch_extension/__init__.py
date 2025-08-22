import torch
from torch.autograd import Function

try:
    from . import spmm_cuda
except ImportError:
    raise ImportError("Please build the CUDA extension with: python setup.py build_ext --inplace")

class SparseMMFunction(Function):
    @staticmethod
    def forward(ctx, row_ptr, col_idx, values, dense_matrix, shape):
        num_rows, num_cols = shape
        ctx.save_for_backward(row_ptr, col_idx, values, dense_matrix)
        ctx.shape = shape
        
        return spmm_cuda.spmm_csr(row_ptr, col_idx, values, dense_matrix, num_rows, num_cols)
    
    @staticmethod
    def backward(ctx, grad_output):
        row_ptr, col_idx, values, dense_matrix = ctx.saved_tensors
        num_rows, num_cols = ctx.shape
        
        # For backward pass: dL/dDense = Sparse^T @ dL/dOutput
        # dL/dSparse values would be more complex, typically handled differently
        grad_dense = torch.matmul(values.unsqueeze(1), grad_output.unsqueeze(0))
        
        # For sparse matrix gradients, we usually use different techniques
        # Returning None for sparse gradients as they're typically handled separately
        return None, None, None, grad_dense, None

def sparse_mm(row_ptr, col_idx, values, dense_matrix, shape):
    """
    Sparse matrix multiplication with dense matrix
    
    Args:
        row_ptr: CSR row pointer tensor (int32, CUDA)
        col_idx: CSR column indices tensor (int32, CUDA)
        values: CSR values tensor (float32, CUDA)
        dense_matrix: Dense matrix tensor (float32, CUDA)
        shape: Tuple of (num_rows, num_cols) for sparse matrix
    
    Returns:
        Result of sparse @ dense multiplication
    """
    return SparseMMFunction.apply(row_ptr, col_idx, values, dense_matrix, shape)