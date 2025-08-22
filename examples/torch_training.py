import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
sys.path.append('../torch_extension')
from torch_integration import SparseNetwork

# Build the extension
def build_extension():
    import subprocess
    import os
    old_dir = os.getcwd()
    os.chdir('../torch_extension')
    subprocess.run(['python', 'setup.py', 'build_ext', '--inplace'])
    os.chdir(old_dir)

# Build before running
build_extension()

# Create sample data
def create_sample_data(batch_size=32, input_size=784, num_classes=10):
    x = torch.randn(batch_size * 100, input_size)
    y = torch.randint(0, num_classes, (batch_size * 100,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)

def train_sparse_network():
    # Hyperparameters
    input_size = 784
    hidden_size = 512
    output_size = 10
    sparsity = 0.98  # 98% sparse weights
    learning_rate = 0.001
    epochs = 10
    
    # Model, loss, optimizer
    model = SparseNetwork(input_size, hidden_size, output_size, sparsity)
    model = model.cuda()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Data
    train_loader = create_sample_data()
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        print(f'Epoch {epoch}, Average Loss: {total_loss/len(train_loader):.4f}')
    
    print("Training completed!")

if __name__ == "__main__":
    train_sparse_network()