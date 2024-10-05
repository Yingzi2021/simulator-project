import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from flash_attn import flash_attn_func

# Load GPT-2 model
config = GPT2Config.from_pretrained('gpt2')
model = GPT2Model(config)

# Sample input for GPT-like model
batch_size = 8
sequence_length = 128
input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_length))

# Enable profiler to capture low-level operations
with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=True) as prof:
    with torch.no_grad():
        # Run the forward pass
        outputs = model(input_ids)

# Iterate through each event to capture the shapes
for evt in prof.function_events:
    if evt.cpu_time_total > 0:  # Filter out very small operations
        print(f"Operation: {evt.name}, Input Shapes: {evt.input_shapes}, CPU Time: {evt.cpu_time_total}us, CUDA Time: {evt.cuda_time_total}us")

# You can still export to a Chrome trace to visualize the entire execution
prof.export_chrome_trace("trace.json")
