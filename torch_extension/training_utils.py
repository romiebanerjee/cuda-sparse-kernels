import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

class SparseAETrainer:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
    
    def reconstruction_loss(self, x_recon, x_original):
        """MSE reconstruction loss"""
        return F.mse_loss(x_recon, x_original)
    
    def sparsity_loss(self, encoded):
        """L1 sparsity constraint on bottleneck layer"""
        return torch.mean(torch.abs(encoded))
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        recon_loss_total = 0
        sparsity_loss_total = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            
            x = x.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            encoded, decoded = self.model(x)
            
            # Calculate losses
            recon_loss = self.reconstruction_loss(decoded, x)
            sparse_loss = self.sparsity_loss(encoded) * 0.01  # Weighting factor
            loss = recon_loss + sparse_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            recon_loss_total += recon_loss.item()
            sparsity_loss_total += sparse_loss.item()
        
        return {
            'total_loss': total_loss / len(dataloader),
            'recon_loss': recon_loss_total / len(dataloader),
            'sparsity_loss': sparsity_loss_total / len(dataloader)
        }
    
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        recon_loss_total = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                
                x = x.to(self.device)
                
                encoded, decoded = self.model(x)
                recon_loss = self.reconstruction_loss(decoded, x)
                total_loss += recon_loss.item()
                recon_loss_total += recon_loss.item()
        
        return {
            'val_loss': total_loss / len(dataloader),
            'val_recon_loss': recon_loss_total / len(dataloader)
        }
    
    def train(self, train_loader, val_loader, epochs=100):
        history = {
            'train_loss': [], 'train_recon': [], 'train_sparsity': [],
            'val_loss': [], 'val_recon': []
        }
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_metrics['val_loss'])
            
            # Save history
            for k, v in train_metrics.items():
                history[k].append(v)
            for k, v in val_metrics.items():
                history[k].append(v)
            
            # Print metrics
            print(f"Train Loss: {train_metrics['total_loss']:.4f} "
                  f"(Recon: {train_metrics['recon_loss']:.4f}, "
                  f"Sparsity: {train_metrics['sparsity_loss']:.4f})")
            print(f"Val Loss: {val_metrics['val_loss']:.4f} "
                  f"(Recon: {val_metrics['val_recon_loss']:.4f})")
            
            # Save best model
            if val_metrics['val_loss'] < best_loss:
                best_loss = val_metrics['val_loss']
                torch.save(self.model.state_dict(), 'best_sparse_ae.pth')
                print("Saved best model!")
            
            # Print sparsity stats occasionally
            if (epoch + 1) % 10 == 0:
                stats = self.model.get_sparsity_stats()
                print("Sparsity Statistics:")
                for k, v in stats.items():
                    print(f"  {k}: {v}")
        
        return history

def create_mnist_autoencoder(k_sparse=20, decoder_sparsity=0.98):
    """Create autoencoder for MNIST-like data"""
    return SparseAutoencoder(
        input_dim=784,
        encoder_dims=[512, 256],
        bottleneck_dim=128,
        decoder_dims=[256, 512],
        k_sparse=k_sparse,
        decoder_sparsity=decoder_sparsity
    )

def create_cifar_autoencoder(k_sparse=100, decoder_sparsity=0.95):
    """Create autoencoder for CIFAR-like data"""
    return SparseAutoencoder(
        input_dim=3072,  # 32x32x3
        encoder_dims=[2048, 1024],
        bottleneck_dim=512,
        decoder_dims=[1024, 2048],
        k_sparse=k_sparse,
        decoder_sparsity=decoder_sparsity
    )