import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config
from flash_attn import flash_attn_func

# Hook function for layers and operations
def hook_fn(module, input, output):
    # Print input shapes
    input_shapes = [i.shape for i in input if isinstance(i, torch.Tensor)]
    print(f"Layer: {module.__class__.__name__}, Input Shape: {input_shapes}")

    # Check if output is a tensor or tuple
    if isinstance(output, torch.Tensor):
        print(f"Output Shape: {output.shape}")
    elif isinstance(output, tuple):
        output_shapes = [o.shape for o in output if isinstance(o, torch.Tensor)]
        print(f"Output Shapes: {output_shapes}")
    else:
        print(f"Output Type: {type(output)}")

# Load GPT-2 model
config = GPT2Config.from_pretrained('gpt2')
model = GPT2Model(config)

# Register hooks for relevant modules within the GPT2 model
for name, layer in model.named_modules():
    if isinstance(layer, (nn.Linear, nn.LayerNorm, nn.Softmax)):
        layer.register_forward_hook(hook_fn)
    elif "attn" in name:  # Specifically targeting GPT2Attention layers
        layer.register_forward_hook(hook_fn)

# Sample input for GPT-like model
batch_size = 8
sequence_length = 128
input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_length))

# Forward pass
with torch.no_grad():
    outputs = model(input_ids)
