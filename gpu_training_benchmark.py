import torch
import torch.nn as nn
import torch.optim as optim
import time

# --- 1. Check GPU availability ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ MPS GPU is available. Using M4 GPU.")
else:
    device = torch.device("cpu")
    print("❌ MPS GPU not available. Using CPU.")

# --- 2. Define a LARGER model ---
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1000, 4000),  # Increased from 2000 to 4000
            nn.ReLU(),
            nn.Linear(4000, 4000),  # Added another large layer
            nn.ReLU(),
            nn.Linear(4000, 10)     # Increased from 1000 to 4000
        )
    def forward(self, x):
        return self.layers(x)

# --- 3. Create data ---
batch_size = 1024  # Increased from 256
input_size = 1000
num_batches = 100

# --- 4. Function to train on a specific device ---
def train_on_device(device_name):
    print(f"\n--- Training on {device_name} (Batch Size: {batch_size}) ---")
    device = torch.device(device_name)
    
    model = SimpleModel().to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    
    # Create dummy data
    inputs = torch.randn(batch_size, input_size).to(device)
    labels = torch.randn(batch_size, 10).to(device)
    
    # Warm-up (This is crucial for the GPU)
    _ = model(inputs)
    if device_name == "mps":
        torch.mps.synchronize()
    
    start = time.time()
    
    for _ in range(num_batches):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
    
    if device_name == "mps":
        torch.mps.synchronize() # Wait for GPU to finish
    
    end = time.time()
    print(f"{device_name.upper()} training time: {end - start:.2f} seconds")

# --- 5. Run the benchmarks ---
train_on_device("cpu")

if torch.backends.mps.is_available():
    train_on_device("mps")