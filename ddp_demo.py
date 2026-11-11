import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

# A simple model
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def setup(rank, world_size):
    """Initialize the process group."""
    # Set the environment variables for rendezvous
    os.environ['MASTER_ADDR'] = '127.0.0.1'  # localhost
    os.environ['MASTER_PORT'] = '29500'       # a free port
    
    # Now initialize the process group
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
    

def cleanup():
    """Destroy the process group."""
    dist.destroy_process_group()

def demo_basic(rank, world_size):
    print(f"Running DDP on rank {rank}.")
    setup(rank, world_size)

    # Create model and move it to GPU if available (we will use CPU for this demo)
    model = ToyModel()
    
    # TODO 2: Wrap the model in DDP.
    # HINT: ddp_model = DDP(model)
    ddp_model = DDP(model)

    # Create a dummy loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # Create dummy data
    # We use the rank to make sure each process gets DIFFERENT data
    inputs = torch.randn(20, 10) + rank
    labels = torch.randn(20, 5)

    # Training loop (just one step for this demo)
    optimizer.zero_grad()
    outputs = ddp_model(inputs)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()

    print(f"Rank {rank} finished. Loss: {loss.item()}")

    cleanup()

if __name__ == "__main__":
    # This is a trick to run multiple processes from one script
    # In a real scenario, you would use torchrun or a similar launcher
    world_size = 2  # Simulate 2 GPUs
    import torch.multiprocessing as mp
    mp.spawn(
        demo_basic,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )