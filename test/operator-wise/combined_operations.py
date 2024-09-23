import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import os
import argparse
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

def setup(rank, world_size):
    dist.init_process_group(
        "nccl", 
        rank=rank, 
        world_size=world_size
    )
    torch.cuda.set_device(rank)

def create_tensors(rank, world_size, matrix_size, tensor_size):
    matrix1 = torch.randn(matrix_size, matrix_size, device='cuda')
    matrix2 = torch.randn(matrix_size, matrix_size, device='cuda')
    extra_tensor = torch.randn(tensor_size, device='cuda')
    return matrix1, matrix2, extra_tensor

def torch_combined(matrix1, matrix2, extra_tensor, stream, all_reduce_stream, overlap):
    repeat = 10
    layer_norm = torch.nn.LayerNorm(matrix1.shape[0]).to(device='cuda')
    nn_linear = torch.nn.Linear(20, 30).to(device='cuda')
    linear_input = torch.randn(128, 20).to(device='cuda')
    def test_gqa(Z, H, KVH, N_CTX, D_HEAD, causal, dtype=torch.float16):
        q = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
        k = torch.empty((Z, KVH, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
        v = torch.empty((Z, KVH, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
        sm_scale = 0.5
        dout = torch.randn_like(q)
        flash_q = q.transpose(1,2).clone().detach().requires_grad_(True)
        flash_k = k.transpose(1,2).clone().detach().requires_grad_(True)
        flash_v = v.transpose(1,2).clone().detach().requires_grad_(True)
        return flash_q, flash_k, flash_v
    flash_q, flash_k, flash_v = test_gqa(Z=1, H=2, KVH=1, N_CTX=1, D_HEAD=1, causal=True)

    torch.cuda.nvtx.range_push("combined")
    
    for i in range(repeat):
        with torch.cuda.stream(stream):
            result = torch.sum(matrix1)
        if overlap:
            with torch.cuda.stream(all_reduce_stream):
                all_reduce(extra_tensor, all_reduce_stream)
    for i in range(repeat):
        with torch.cuda.stream(stream):
            result = torch.add(matrix1, matrix2)
        if overlap:
            with torch.cuda.stream(all_reduce_stream):
                all_reduce(extra_tensor, all_reduce_stream)
    for i in range(repeat):
        with torch.cuda.stream(stream):
            result = layer_norm(matrix1)
        if overlap:
            with torch.cuda.stream(all_reduce_stream):
                all_reduce(extra_tensor, all_reduce_stream)
    for i in range(repeat):
        with torch.cuda.stream(stream):
            result = torch.nn.functional.softmax(matrix1, dim=0)
        if overlap:
            with torch.cuda.stream(all_reduce_stream):
                all_reduce(extra_tensor, all_reduce_stream)
    for i in range(repeat):
        with torch.cuda.stream(stream):
            result = nn_linear(linear_input)
        if overlap:
            with torch.cuda.stream(all_reduce_stream):
                all_reduce(extra_tensor, all_reduce_stream)
    for i in range(repeat):
        with torch.cuda.stream(stream):
            result = flash_attn_func(flash_q, flash_k, flash_v, 0, 0.5, True)
        if overlap:
            with torch.cuda.stream(all_reduce_stream):
                all_reduce(extra_tensor, all_reduce_stream)
    torch.cuda.nvtx.range_pop()

    return result

def all_reduce(tensor, stream):
    with torch.cuda.stream(stream):
        dist.all_reduce(tensor)

def main(rank, world_size, matrix_size, tensor_size, iterations, overlap):
    setup(rank, world_size)

    matrix1, matrix2, extra_tensor = create_tensors(rank, world_size, matrix_size, tensor_size)

    operations_stream = torch.cuda.Stream()
    all_reduce_stream = torch.cuda.Stream()

    warmup_iter = 50

    with torch.cuda.stream(operations_stream):
        
        idx = 0
        cnt = 0
        for i in range(iterations):
            if i == warmup_iter:
                torch.cuda.cudart().cudaProfilerStart()
                start_time = time.time()
            
                torch_combined(matrix1, matrix2, extra_tensor, operations_stream, all_reduce_stream, overlap)

            if overlap:
                with torch.cuda.stream(all_reduce_stream):
                    all_reduce(extra_tensor, all_reduce_stream)
                    
            if i == iterations - 1:
                torch.cuda.cudart().cudaProfilerStop()
                end_time = time.time()

    operations_stream.synchronize()
    all_reduce_stream.synchronize()

    print(f"Rank {rank}: Matrix multiplication and All-Reduce {iterations} times took {end_time - start_time} seconds.")

    del operations_stream
    del all_reduce_stream

if __name__ == "__main__":
    world_size = 2
    matrix_size = 4096
    tensor_size = 1024 * 1024 * 6
    iterations = 1000

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    # Create the parser
    parser = argparse.ArgumentParser(description="Script with a boolean command line parameter")

    # Add a boolean argument
    parser.add_argument('--overlap', action='store_true', help='Enable overlap')

    # Parse the arguments
    args = parser.parse_args()

    # Check the value of the boolean argument
    overlap = False
    if args.overlap:
        overlap = True

    mp.spawn(main, args=(world_size, matrix_size, tensor_size, iterations, overlap), nprocs=world_size, join=True)
