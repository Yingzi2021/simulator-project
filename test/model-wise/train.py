import os
import argparse
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
import torch.multiprocessing as mp

# Set up the process group
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# Clean up the process group
def cleanup():
    dist.destroy_process_group()

# Prepare model, dataset, and data loader
def prepare(rank, world_size, cap_size, dataset_name):
    setup(rank, world_size)

    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    model.to(rank)
    
    # Wrap model with DDP, using the bucket_cap_mb from command line
    model = DDP(model, device_ids=[rank], bucket_cap_mb=cap_size)

    # Download the specified dataset
    dataset = load_dataset(dataset_name)

    # Data preprocessing function
    def preprocess_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)
    
    encoded_dataset = dataset.map(preprocess_function, batched=True)
    encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Create a distributed data sampler and data loader
    train_sampler = DistributedSampler(encoded_dataset['train'], num_replicas=world_size, rank=rank)
    train_loader = DataLoader(encoded_dataset['train'], batch_size=8, sampler=train_sampler)
    
    return model, train_loader, tokenizer

# Training function
def train(model, train_loader, rank):
    criterion = nn.CrossEntropyLoss().to(rank)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    nb_iters = len(train_loader) * 3  # Total number of iterations for 3 epochs
    last_epoch_start = nb_iters - len(train_loader)  # Start iteration of the last epoch
    middle_of_last_epoch = last_epoch_start + (len(train_loader) // 2)  # Middle iteration of the last epoch

    model.train()
    for epoch in range(3):  # Train for 3 epochs
        train_loader.sampler.set_epoch(epoch)  # Shuffle data for each epoch
        for i, batch in enumerate(train_loader):
            global_iter = i + epoch * len(train_loader)

            inputs = batch['input_ids'].to(rank)
            attention_mask = batch['attention_mask'].to(rank)
            labels = batch['label'].to(rank)

            optimizer.zero_grad()

            # Start profiling at the middle of the last epoch
            if global_iter == middle_of_last_epoch:
                torch.cuda.cudart().cudaProfilerStart()
            
            # Forward pass
            if global_iter == middle_of_last_epoch:
                torch.cuda.nvtx.range_push("forward")
            outputs = model(inputs, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
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

        print(f"Rank {rank}, Epoch [{epoch+1}/3], Loss: {loss.item()}")

def run_training(rank, world_size, cap_size, dataset_name):
    model, train_loader, tokenizer = prepare(rank, world_size, cap_size, dataset_name)
    train(model, train_loader, rank)
    
    if rank == 0:
        model.module.save_pretrained('./fine_tuned_bert')
        tokenizer.save_pretrained('./fine_tuned_bert')
    
    cleanup()

# Main function
def main():
    parser = argparse.ArgumentParser(description="Fine-tune BERT on a dataset")
    parser.add_argument("--cap_size", type=int, default=2, help="Bucket cap size for DDP (in MB)")
    parser.add_argument("--dataset", type=str, default="imdb", help="Dataset to use for training")
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count(), help="Number of GPUs to use for training")
    args = parser.parse_args()

    world_size = args.world_size
    mp.spawn(run_training, args=(world_size, args.cap_size, args.dataset), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()
