import torch
import torch.nn as nn
import torch.nn.functional as F
from .torch_integration import SparseLinear

class TopKActivation(nn.Module):
    """
    Top-K activation function that keeps only the top k values active
    and sets others to zero.
    """
    def __init__(self, k, dim=1):
        super(TopKActivation, self).__init__()
        self.k = k
        self.dim = dim
    
    def forward(self, x):
        # Get top k values and their indices
        topk_values, topk_indices = torch.topk(x, self.k, dim=self.dim)
        
        # Create mask for top k elements
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask.scatter_(self.dim, topk_indices, True)
        
        # Apply mask: keep top k values, zero out others
        return x * mask.float()
    
    def extra_repr(self):
        return f'k={self.k}, dim={self.dim}'

class SparseEncoder(nn.Module):
    """
    Encoder with dense layers and TopK activation for sparsity
    """
    def __init__(self, input_dim, hidden_dims, bottleneck_dim, k_sparse):
        super(SparseEncoder, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build encoder layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Bottleneck layer with TopK activation
        layers.extend([
            nn.Linear(prev_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            TopKActivation(k=k_sparse, dim=1)  # Apply TopK sparsity
        ])
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.encoder(x)

class SparseDecoder(nn.Module):
    """
    Decoder using our custom sparse linear layers
    """
    def __init__(self, bottleneck_dim, hidden_dims, output_dim, sparsity=0.98):
        super(SparseDecoder, self).__init__()
        
        layers = []
        prev_dim = bottleneck_dim
        
        # Build decoder layers with sparse linear
        for hidden_dim in hidden_dims:
            layers.extend([
                SparseLinear(prev_dim, hidden_dim, sparsity=sparsity),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Final output layer (can be dense or sparse)
        layers.extend([
            SparseLinear(prev_dim, output_dim, sparsity=sparsity),
            # No activation for final layer (output depends on task)
        ])
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.decoder(x)

class SparseAutoencoder(nn.Module):
    """
    Complete sparse autoencoder with TopK activation and sparse decoder
    """
    def __init__(self, input_dim, encoder_dims, bottleneck_dim, 
                 decoder_dims=None, k_sparse=50, decoder_sparsity=0.98):
        super(SparseAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.k_sparse = k_sparse
        
        # Use symmetric architecture if decoder_dims not specified
        if decoder_dims is None:
            decoder_dims = list(reversed(encoder_dims))
        
        self.encoder = SparseEncoder(input_dim, encoder_dims, bottleneck_dim, k_sparse)
        self.decoder = SparseDecoder(bottleneck_dim, decoder_dims, input_dim, decoder_sparsity)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization for better training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Encode with TopK sparsity
        encoded = self.encoder(x)
        
        # Decode with sparse weights
        decoded = self.decoder(encoded)
        
        return encoded, decoded
    
    def get_sparsity_stats(self):
        """Get statistics about sparsity in the model"""
        stats = {
            'encoder_params': sum(p.numel() for p in self.encoder.parameters()),
            'decoder_params': sum(p.numel() for p in self.decoder.parameters()),
            'total_params': sum(p.numel() for p in self.parameters()),
            'bottleneck_sparsity': f"{self.k_sparse}/{self.bottleneck_dim} "
                                  f"({self.k_sparse/self.bottleneck_dim*100:.1f}%)"
        }
        
        # Calculate actual sparsity in decoder
        decoder_sparse_layers = [m for m in self.decoder.modules() 
                               if isinstance(m, SparseLinear)]
        
        for i, layer in enumerate(decoder_sparse_layers):
            non_zero = (layer.weight != 0).sum().item()
            total = layer.weight.numel()
            stats[f'decoder_layer_{i}_sparsity'] = f"{non_zero}/{total} ({non_zero/total*100:.1f}%)"
        
        return stats
