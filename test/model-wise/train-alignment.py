import os
import argparse
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

# Define the SimpleModel
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(512, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, 512)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

# Set up the process group
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# Clean up the process group
def cleanup():
    dist.destroy_process_group()

# Prepare model and random data
def prepare(rank, world_size, cap_size):
    setup(rank, world_size)

    # Use SimpleModel instead of BERT
    model = SimpleModel().to(rank)
    
    # Wrap model with DDP, using the bucket_cap_mb from command line
    model = DDP(model, device_ids=[rank], bucket_cap_mb=cap_size)

    # Generate random data tensors
    input_data = torch.randn(64, 512).to(rank)  # Batch size of 64, input features of 512
    labels = torch.randint(0, 10, (64,)).to(rank)  # Random labels for a 10-class classification problem
    
    # Use a simple data loader-like structure
    data_loader = [(input_data, labels)] * 100  # Simulate 100 batches
    
    return model, data_loader

# Training function
def train(model, data_loader, rank, epochs):
    criterion = nn.CrossEntropyLoss().to(rank)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    nb_iters = len(data_loader) * epochs  # Total number of iterations for the specified epochs
    last_epoch_start = nb_iters - len(data_loader)  # Start iteration of the last epoch
    middle_of_last_epoch = last_epoch_start + (len(data_loader) // 2)  # Middle iteration of the last epoch

    model.train()
    for epoch in range(epochs):  # Use the epochs argument to control training duration
        for i, (inputs, labels) in enumerate(data_loader):
            global_iter = i + epoch * len(data_loader)

            optimizer.zero_grad()

            # Start profiling at the middle of the last epoch
            if global_iter == middle_of_last_epoch:
                torch.cuda.cudart().cudaProfilerStart()
            
            # Forward pass
            if global_iter == middle_of_last_epoch:
                torch.cuda.nvtx.range_push("forward")
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if global_iter == middle_of_last_epoch:
                torch.cuda.nvtx.range_pop()

            # Backward pass
            if global_iter == middle_of_last_epoch:
                torch.cuda.nvtx.range_push("backward")
            loss.backward()
            if global_iter == middle_of_last_epoch:
                torch.cuda.nvtx.range_pop()
            
            # Optimize & update gradient
            if global_iter == middle_of_last_epoch:
                torch.cuda.nvtx.range_push("optimize")
            optimizer.step()
            if global_iter == middle_of_last_epoch:
                torch.cuda.nvtx.range_pop()

            # Stop profiling
            if global_iter == middle_of_last_epoch:
                torch.cuda.cudart().cudaProfilerStop()

        print(f"Rank {rank}, Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")

def run_training(rank, world_size, cap_size, epochs):
    model, data_loader = prepare(rank, world_size, cap_size)
    train(model, data_loader, rank, epochs)
    
    cleanup()

# Main function
def main():
    parser = argparse.ArgumentParser(description="Profile SimpleModel training with DDP")
    parser.add_argument("--cap_size", type=int, default=2, help="Bucket cap size for DDP (in MB)")
    parser.add_argument("--epoch", type=int, default=3, help="Number of epochs to train")
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count(), help="Number of GPUs to use for training")
    args = parser.parse_args()

    world_size = args.world_size
    mp.spawn(run_training, args=(world_size, args.cap_size, args.epoch), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()
