# pytorch分桶机制原理与测试

### 1. DDP 总体实现

我们把论文和 https://pytorch.org/docs/master/notes/ddp.html 结合起来，看看 DDP 总体实现。

![img](https://img2020.cnblogs.com/blog/1850883/202111/1850883-20211121121038645-1873197734.png)

我们总结一次DistributedDataParallel迭代中的步骤如下（与上图不完全一致，有部分细化）：

- **Prerequisite：**
  - DDP 依赖 c10d`ProcessGroup`进行通信。因此，应用程序必须`ProcessGroup`在构建 DDP 之前创建实例。
- **Constuctor：**
  - rank 0 进程会引用本地模块，把模型`state_dict()`参数广播到所有进程之中，这样可以保证所有进程使用同样初始化数值和模型副本进行训练。
  - 每个 DDP 进程创建一个 local `Reducer`，稍后将在向后传递期间处理梯度同步。
  - **为了提高通信效率，`Reducer`将参数梯度组织成桶，一次规约一个桶。**
    - 初始化桶，按照逆序把 parameters 分配到桶之中，这样可以提高通信效率。
    - 可以通过设置DDP 构造函数中的参数bucket_cap_mb来配置桶的大小。
    - 从参数梯度到桶的映射是在构建时根据桶大小限制和参数大小确定的。模型参数以（大致）`Model.parameters()`与给定模型相反的顺序分配到桶中 。使用相反顺序的原因是因为 DDP 期望梯度在反向传递期间以大约该顺序准备就绪。
    - 下图显示了一个示例。请注意，`grad0`和`grad1`在 `bucket1`中，另外两个梯度在 `bucket0`中。当然，这种假设可能并不总是正确的，当这种情况发生时，它可能会损害 DDP 后向速度，因为它无法 `Reducer`尽早开始通信。
    - ![img](https://img2020.cnblogs.com/blog/1850883/202111/1850883-20211121121133146-1254841983.png)
  - 除了分桶，`Reducer`还在构造期间注册 autograd 钩子，每个参数一个钩子。当梯度准备好时，将在向后传递期间触发这些钩子。具体就是遍历参数，为每个参数加上 grad_accumulator 和 autograd_hook。
- **Forward Pass**:
  - 每个进程读去自己的训练数据，DistributedSampler确保每个进程读到的数据不同。
  - DDP 获取输入并将其传递给本地模型。
  - 模型进行前向计算，结果设置为 out。现在计算都是在每个进程（CUDA设备）上完成。
  - 如果`find_unused_parameters`设置为`True`，DDP 会分析本地模型的输出，从 out 开始遍历计算图，把未使用参数标示为 ready，因为每次计算图都会改变，所以每次都要遍历。
    - 此模式（Mode）允许在模型的子图上向后运行，并且 DDP 通过从模型输出out遍历 autograd 图并将所有未使用的参数标记为就绪，以减少反向传递中涉及的参数。
    - 在向后传递期间，`Reducer`只会等待未准备好的参数，但它仍然会规约所有桶。将参数梯度标记为就绪并不能帮助 DDP 跳过桶，但它会阻止 DDP 在向后传递期间永远等待不存在的梯度。
    - 请注意，遍历 autograd 图会引入额外的开销，因此应用程序仅应必要时才设置 `find_unused_parameters`为`True` 。
  - 返回out。模型网络输出不需要gather到 rank 0进程了，这与 DP不同。
- **Backward Pass**:
  - `backward()`在 loss 上直接调用该函数 `Tensor`，这是 DDP 无法控制的，DDP 使用构造时注册的 autograd hooks 来触发梯度同步。当一个梯度准备好时，它在该梯度累加器上的相应 DDP 钩子将触发。
  - 在 autograd_hook 之中进行all-reduce。假设参数index是param_index，则利用param_index获取到参数，标示为ready，如果某个桶里面梯度都ready，则该桶是ready。
  - **当一个桶中的梯度都准备好时，会 在该桶上`Reducer`启动异步`allreduce`以计算所有进程的梯度平均值。**
  - **如果所有桶都ready，则等待所有 all-reduce 完成。当所有桶都准备好时，`Reducer`将阻塞等待所有`allreduce`操作完成。完成此操作后，将平均梯度写入`param.grad`所有参数的字段。**
  - 所有进程的梯度都会reduce，更新之后，大家的模型权重都相同。所以在向后传播完成之后，跨不同DDP进程的对应的相同参数上的 grad 字段应该是相等的。
  - 不需要像 DP 那样每次迭代之后还要广播参数。但是 Buffers 还是需要在每次迭代由 rank 0 进程广播到其他进程之上。
- **Optimizer Step**:
  - 从优化器的角度来看，它正在优化本地模型。
  - 所有 DDP 进程上的模型副本都可以保持同步，因为它们都从相同的状态开始，并且在每次迭代中都具有相同的平均梯度。