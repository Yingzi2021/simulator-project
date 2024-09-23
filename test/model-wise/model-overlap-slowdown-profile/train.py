import os
import argparse
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torchvision import models
import transformers
import json  # For reading the config file

'''
2024/9/14
from train-align-simplified.py
chen boying
'''

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
def prepare(rank, world_size, cap_size, model_name, model_type, batchsize):
    setup(rank, world_size)

    if model_type == 'CV':
        model = getattr(models, model_name)().to(rank)
        example = torch.rand(batchsize, 3, 224, 224).to(rank)
    elif model_type == 'NLP':
        model = getattr(transformers, model_name)().to(rank)
        example = (torch.LongTensor(batchsize, 512).random_() % 1000).to(rank)
    else:
        raise ValueError("Unsupported model type. Use 'CV' for computer vision or 'NLP' for natural language processing.")

    # Wrap model with DDP, using the bucket_cap_mb from the config
    model = DDP(model, device_ids=[rank], bucket_cap_mb=cap_size)

    # Create fake labels (assuming 10 classes for CV and NLP tasks)
    labels = torch.randint(0, 10, (batchsize,)).to(rank)
    
    # Use a simple data loader-like structure
    data_loader = [(example, labels)] * 100  # Simulate 100 batches
    
    return model, data_loader

# Training function
def train(model, data_loader, rank, epochs, model_type):
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
                dist.barrier()  # Synchronize all processes
                torch.cuda.synchronize()  # Ensure all CUDA operations are completed
                torch.cuda.cudart().cudaProfilerStart()
            
            # Forward pass
            if global_iter == middle_of_last_epoch:
                dist.barrier()  # Synchronize all processes
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_push("forward")
            output = model(inputs)
            if global_iter == middle_of_last_epoch:
                dist.barrier()  # Synchronize all processes
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()

            # Backward pass
            if global_iter == middle_of_last_epoch:
                dist.barrier()  # Synchronize all processes
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_push("backward")

            if model_type == 'CV':
                output.backward(output)
            elif model_type == 'NLP':
                if hasattr(output, 'pooler_output'):
                    output.pooler_output.backward(output.pooler_output)
                else:
                    output.last_hidden_state.backward(output.last_hidden_state)

            if global_iter == middle_of_last_epoch:
                dist.barrier()  # Synchronize all processes
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()

            # Stop profiling
            if global_iter == middle_of_last_epoch:
                dist.barrier()  # Synchronize all processes
                torch.cuda.synchronize()  # Ensure all CUDA operations are completed
                torch.cuda.cudart().cudaProfilerStop()

            # Optimize & update gradient (don't profile)
            optimizer.step()
        
        # Optional: Clear cache to manage memory usage
        torch.cuda.empty_cache()
        print(f"Rank {rank}, Epoch [{epoch+1}/{epochs}], Finished epoch.")

def run_training(rank, world_size, cap_size, epochs, model_name, model_type, batchsize):
    model, data_loader = prepare(rank, world_size, cap_size, model_name, model_type, batchsize)
    #with torch.autograd.profiler.emit_nvtx(enabled=True, record_shapes=False):# a PyTorch utility that automatically inserts NVTX markers into the CUDA execution stream.
    train(model, data_loader, rank, epochs, model_type)
    cleanup()

# Main function
def main():
    parser = argparse.ArgumentParser(description="Profile training with DDP")
    parser.add_argument("--config_file", type=str, default="config.json", help="Path to the configuration file")
    args = parser.parse_args()

    # Load the configuration from the JSON file
    with open(args.config_file, 'r') as f:
        config = json.load(f)

    # Extract parameters from the configuration
    world_size = config.get('world_size', torch.cuda.device_count())
    cap_size = config.get('cap_size', 2)
    epochs = config.get('epochs', 3)
    model_name = config.get('model_name', 'resnet152')
    model_type = config.get('model_type', 'CV')
    batchsize = config.get('batchsize', 32)

    mp.spawn(run_training, args=(world_size, cap_size, epochs, model_name, model_type, batchsize), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()
