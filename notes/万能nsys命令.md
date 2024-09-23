# 万能nsys命令

```
nsys profile --trace=cuda,nvtx,osrt,python-gil --sample=cpu --python-sampling=true --python-backtrace=cuda --gpuctxsw=true --output=cxx_thread_v3_report --export=none --force-overwrite true  --cuda-graph-trace=node --capture-range=cudaProfilerApi
```

`nsys` 是 NVIDIA 的一个性能分析工具，它允许用户分析应用程序的性能，特别是 GPU 加速应用程序。`nsys` 命令行提供了多种选项来定制性能分析的范围和深度。下面是你提供的 `nsys` 命令参数的详细解释：

1. `profile`：这是 `nsys` 的一个子命令，用于启动性能分析。
2. `--trace=cuda,nvtx,osrt,python-gil`：这个参数指定了要跟踪的资源类型。这里包括：
   - `cuda`：CUDA API 调用。
   - `nvtx`：NVIDIA 工具扩展（用于标记 CUDA 应用程序的不同阶段）。
   - `osrt`：操作系统运行时（例如线程和进程）。
   - `python-gil`：Python 全局解释器锁（GIL）。
3. `--sample=cpu`：这个参数指定 CPU 级别采样，意味着 `nsys` 将记录 CPU 活动，如线程的上下文切换。
4. `--python-sampling=true`：启用 Python 函数调用的采样。
5. `--python-backtrace=cuda`：当 CUDA 事件被触发时，收集 Python 的调用栈跟踪。
6. `--gpuctxsw=true`：记录 GPU 上下文切换，这有助于分析 GPU 上下文创建和销毁的性能影响。
7. `--output=cxx_thread_v3_report`：指定输出报告的名称。
8. `--export=none`：不导出任何额外的数据，只生成报告。
9. `--force-overwrite true`：如果输出文件已存在，强制覆盖它。
10. `--cuda-graph-trace=node`：记录 CUDA 图形的节点级别的信息。
11. `--capture-range=cudaProfilerApi`：指定要捕获的范围，这里使用 `cudaProfilerApi` 作为范围的开始和结束。
12. 命令行的最后部分 `--force-overwrite true` 后面应该有一个空格分隔，但根据你提供的命令，这部分语法有误。正确的语法应该是 `--force-overwrite true`。

将这些参数组合起来，整个命令的作用是使用 `nsys` 工具来分析一个应用程序的性能，特别关注 CPU 和 GPU 的活动，Python 相关的性能问题，以及 CUDA 相关的详细事件。这可以帮助用户识别性能瓶颈，如 GPU 上下文切换开销、CUDA API 调用的性能，以及 Python 层面的问题。

请注意，为了运行这个命令，你需要有一个支持 CUDA 的 NVIDIA GPU，并且 `nsys` 工具应该已经安装在你的系统上。此外，`--python-sampling` 和 `--python-backtrace` 参数可能需要你的应用程序使用特定的 Python 环境或配置，以便正确地收集数据。