import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import os
import argparse
import json  # For reading the config file
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

# Set up the process group
def setup(rank, world_size):
    dist.init_process_group(
        "nccl", 
        rank=rank, 
        world_size=world_size
    )
    torch.cuda.set_device(rank)

def initialize_workload(rank, world_size, matrix_size, tensor_size_mb, linear_input_size, attention_config):
    # Initialize matrices for basic operations
    tensor_size = 1024 * 1024 * tensor_size_mb
    matrix1 = torch.randn(matrix_size, matrix_size, device='cuda')
    matrix2 = torch.randn(matrix_size, matrix_size, device='cuda')
    extra_tensor = torch.randn(tensor_size, device='cuda')

    # Initialize layer normalization
    layer_norm = torch.nn.LayerNorm(matrix1.shape[0]).to(device='cuda') 
    
    # Initialize linear layer with customizable input size
    nn_linear = torch.nn.Linear(linear_input_size, 30).to(device='cuda') 
    linear_input = torch.randn(128, linear_input_size).to(device='cuda')

    # Generate input tensors for flash attention with parameters from config
    def test_gqa(Z, H, KVH, N_CTX, D_HEAD, causal, dtype=torch.float16):
        q = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
        k = torch.empty((Z, KVH, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
        v = torch.empty((Z, KVH, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
        sm_scale = 0.5
        dout = torch.randn_like(q)
        flash_q = q.transpose(1, 2).clone().detach().requires_grad_(True)
        flash_k = k.transpose(1, 2).clone().detach().requires_grad_(True)
        flash_v = v.transpose(1, 2).clone().detach().requires_grad_(True)
        return flash_q, flash_k, flash_v
    
    flash_q, flash_k, flash_v = test_gqa(
        attention_config['Z'], attention_config['H'], attention_config['KVH'], attention_config['N_CTX'], attention_config['D_HEAD'], causal=True
    )

    return {
        'matrix1': matrix1,
        'matrix2': matrix2,
        'extra_tensor': extra_tensor,
        'layer_norm': layer_norm,
        'nn_linear': nn_linear,
        'linear_input': linear_input,
        'flash_q': flash_q,
        'flash_k': flash_k,
        'flash_v': flash_v
    }

def torch_combined(workload, stream, all_reduce_stream, overlap, operation):
    repeat = 10
    torch.cuda.nvtx.range_push("combined")
    
    operations_list = ["sum", "add", "layer_norm", "softmax", "linear", "flash_attn"]
    
    if operation == "all":
        ops_to_run = operations_list
    else:
        ops_to_run = [operation]

    for op in ops_to_run:
        for i in range(repeat):
            with torch.cuda.stream(stream):
                if op == "sum":
                    result = torch.sum(workload['matrix1'])  # 1. Sum
                elif op == "add":
                    result = torch.add(workload['matrix1'], workload['matrix2'])  # 2. Add
                elif op == "layer_norm":
                    result = workload['layer_norm'](workload['matrix1'])  # 3. LayerNorm
                elif op == "softmax":
                    result = torch.nn.functional.softmax(workload['matrix1'], dim=0)  # 4. Softmax
                elif op == "linear":
                    result = workload['nn_linear'](workload['linear_input'])  # 5. Linear Layer
                elif op == "flash_attn":
                    result = flash_attn_func(workload['flash_q'], workload['flash_k'], workload['flash_v'], 0, 0.5, True)  # 6. Flash Attention
                else:
                    raise ValueError(f"Unsupported operation: {op}")

            if overlap:
                with torch.cuda.stream(all_reduce_stream):
                    all_reduce(workload['extra_tensor'], all_reduce_stream)

    torch.cuda.nvtx.range_pop()
    return result

def all_reduce(tensor, stream):
    with torch.cuda.stream(stream):
        dist.all_reduce(tensor)

def main(rank, world_size, matrix_size, tensor_size_mb, iterations, overlap, linear_input_size, config, operation):
    setup(rank, world_size)

    # Initialize workload with user-specified parameters
    workload = initialize_workload(
        rank, world_size, matrix_size, tensor_size_mb, linear_input_size, config['attention_config']
    )

    operations_stream = torch.cuda.Stream()
    all_reduce_stream = torch.cuda.Stream()

    warmup_iter = 50

    with torch.cuda.stream(operations_stream):
        
        for i in range(iterations):
            if i == warmup_iter:
                torch.cuda.cudart().cudaProfilerStart()
                start_time = time.time()
            
                torch_combined(workload, operations_stream, all_reduce_stream, overlap, operation)

            if overlap:
                with torch.cuda.stream(all_reduce_stream):
                    all_reduce(workload['extra_tensor'], all_reduce_stream)
                    
            if i == iterations - 1:
                torch.cuda.cudart().cudaProfilerStop()
                end_time = time.time()

    operations_stream.synchronize()
    all_reduce_stream.synchronize()

    print(f"Rank {rank}: {operation} operation {iterations} times took {end_time - start_time} seconds.")

    del operations_stream
    del all_reduce_stream

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Script to run DDP with configurable parameters")

    # Add arguments for the config file
    parser.add_argument('--config_file', type=str, default='config.json', help='Path to the configuration file')

    # Parse the arguments
    args = parser.parse_args()

    # Load the configuration from the JSON file
    with open(args.config_file, 'r') as f:
        config = json.load(f)

    # Extract parameters from the configuration
    # general
    world_size = config.get('world_size', 2)
    iterations = config.get('iterations', 1000)
    overlap = config.get('overlap', False)
    
    # communication workload
    tensor_size_mb = config.get('tensor_size', 6)
    
    # computation workload
    matrix_size = config.get('matrix_size', 4096)
    linear_input_size = config.get('linear_input_size', 20)
    attention_config = config.get('attention_config', {})
    Z = attention_config.get('Z', 1)
    H = attention_config.get('H', 2)
    KVH = attention_config.get('KVH', 1)
    N_CTX = attention_config.get('N_CTX', 1)
    D_HEAD = attention_config.get('D_HEAD', 1)
    operation = config.get('operation', 'sum')  # Default to 'sum' if not specified
    # Can be "sum", "add", "layer_norm", "softmax", "linear", "flash_attn", or "all"

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    # Make sure to include `operation` in the args passed to `main`
    mp.spawn(main, args=(world_size, matrix_size, tensor_size_mb, iterations, overlap, linear_input_size, config, operation), nprocs=world_size, join=True)
