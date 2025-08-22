import torch
import torch.nn as nn
from . import sparse_mm

class SparseLinear(nn.Module):
    def __init__(self, in_features, out_features, sparsity=0.9):
        super(SparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Create a sparse weight matrix in CSR format
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        
        # Create mask for sparsity
        mask = torch.rand(out_features, in_features) > sparsity
        self.register_buffer('mask', mask)
        
        # Convert to CSR format
        self._update_csr_format()
    
    def _update_csr_format(self):
        # Apply mask to create sparse matrix
        sparse_weight = self.weight * self.mask.float()
        
        # Convert to CSR format
        non_zero_mask = sparse_weight != 0
        self.values = sparse_weight[non_zero_mask]
        self.col_idx = torch.nonzero(non_zero_mask, as_tuple=True)[1]
        
        # Create row_ptr
        row_counts = non_zero_mask.sum(dim=1)
        self.row_ptr = torch.cat([torch.tensor([0], device=self.weight.device), 
                                 row_counts.cumsum(dim=0)])
        
        self.shape = (self.out_features, self.in_features)
    
    def forward(self, x):
        # Update CSR format if weights changed
        if self.training:
            self._update_csr_format()
        
        return sparse_mm(self.row_ptr, self.col_idx, self.values, x.t(), self.shape).t()
    
    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}'

class SparseNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, sparsity=0.95):
        super(SparseNetwork, self).__init__()
        
        self.sparse1 = SparseLinear(input_size, hidden_size, sparsity)
        self.relu = nn.ReLU()
        self.sparse2 = SparseLinear(hidden_size, output_size, sparsity)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.relu(self.sparse1(x))
        x = self.dropout(x)
        x = self.sparse2(x)
        return x