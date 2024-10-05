# 获取给定sample input情况下6个感兴趣op的input size

## 目标：

已知，在GPT2模型训练过程中，输入一个sample（size = x），在forward的阶段会经历一系列的计算。这些计算可能就包括我们所要测试的6个operator("sum", "add", "layer_norm", "softmax", "linear", "flash_attn"),或者与之类似的算子。

求解：当sample的size=x时，这些算子的input size又是多少？

## 期望结果：

6个tensor size，分别与6个operator相对应。

## 解决方案：

（1）对于本身就是包装好的模型层的算子（如layer_norm）：注册hooks

```python
# test-hooks.py
# Register hooks for relevant modules within the GPT2 model
for name, layer in model.named_modules():
    if isinstance(layer, (nn.Linear, nn.LayerNorm, nn.Softmax)):
        layer.register_forward_hook(hook_fn)
    elif "attn" in name:  # Specifically targeting GPT2Attention layers
        layer.register_forward_hook(hook_fn)
```

（2）对于更低级的操作（如add、sum）：使用`torch.autograd.profiler`

```python
# test-profiler.py
# Enable profiler to capture low-level operations
with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=True) as prof:
    with torch.no_grad():
        # Run the forward pass
        outputs = model(input_ids)

# Iterate through each event to capture the shapes
for evt in prof.function_events:
    if evt.cpu_time_total > 0:  # Filter out very small operations
        print(f"Operation: {evt.name}, Input Shapes: {evt.input_shapes}, CPU Time: {evt.cpu_time_total}us, CUDA Time: {evt.cuda_time_total}us")
```

## 实验结果：

**sample input:**

```python
# Sample input for GPT-like model
batch_size = 8
sequence_length = 128
input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_length))
```

**统计结果：**

| operator name           | input tensor size                                            |
| ----------------------- | ------------------------------------------------------------ |
| aten::native_layer_norm | [[8, 128, 768], [], [768], [768], []]                        |
| aten::softmax           | [[8, 12, 128, 128], [], []]                                  |
| aten::add               | [[8, 128, 768], [1, 128, 768], []]                           |
| aten::addmm             | [[768], [1024, 768], [768, 768], [], []]，<br>[[2304], [1024, 768], [768, 2304], [], []], <br>[[3072], [1024, 768], [768, 3072], [], []]. |
| GPT2Attention           | [8, 128, 768]                                                |
| sum                     | 没有出现                                                     |
| flash_attn              | 没有出现                                                     |

> ChatGPT的解释：
>
> 在 PyTorch 的操作统计中，`Linear` 层的操作通常不会直接以 `Linear` 的名称出现，而是通过底层的操作符来实现。在 `Linear` 层的计算过程中，最主要的两个操作是：
>
> 1. **矩阵乘法**：这个操作通过 `aten::addmm` 实现。
> 2. **加法**（偏置项的加法）：这个操作通过 `aten::add` 实现。
>
> 因此，当你看到 `aten::addmm` 和 `aten::add` 出现在统计结果中时，它们实际上就是 `Linear` 层的核心操作。

详细统计结果见`layer-full.txt`, `profiler-full.txt`和`profiler-brief.txt`