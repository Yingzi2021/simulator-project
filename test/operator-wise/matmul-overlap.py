import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import os

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

def matrix_multiply(mat1, mat2, stream):
    torch.cuda.nvtx.range_push("matmul")
    with torch.cuda.stream(stream):
        result = torch.matmul(mat1, mat2)
    torch.cuda.nvtx.range_pop()
    return result

def all_reduce(tensor, stream):
    with torch.cuda.stream(stream):
        dist.all_reduce(tensor)

def main(rank, world_size, matrix_size, tensor_size, iterations):
    setup(rank, world_size)
    matrix1, matrix2, extra_tensor = create_tensors(rank, world_size, matrix_size, tensor_size)

    matrix_multiply_stream = torch.cuda.Stream()
    all_reduce_stream = torch.cuda.Stream()

    warmup_iter = 50
    with torch.cuda.stream(matrix_multiply_stream):
        for i in range(iterations):
            if i == warmup_iter:
                torch.cuda.cudart().cudaProfilerStart()
                start_time = time.time()
            
            result = matrix_multiply(matrix1, matrix2, matrix_multiply_stream)
            matrix1, matrix2 = result, matrix1  

            with torch.cuda.stream(all_reduce_stream):
                all_reduce(extra_tensor, all_reduce_stream)
            # 如何弄一个开关似的东西让计算和通信总量不变，但是一会overlap，一会不overlap？
            if i == iterations - 1:
                torch.cuda.cudart().cudaProfilerStop()
                end_time = time.time()

    matrix_multiply_stream.synchronize()
    all_reduce_stream.synchronize()

    print(f"Rank {rank}: Matrix multiplication and All-Reduce {iterations} times took {end_time - start_time} seconds.")

    del matrix_multiply_stream
    del all_reduce_stream

if __name__ == "__main__":
    world_size = 2
    matrix_size = 4096
    tensor_size = 1024 * 1024 * 6
    iterations = 100

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    mp.spawn(main, args=(world_size, matrix_size, tensor_size, iterations), nprocs=world_size, join=True)
