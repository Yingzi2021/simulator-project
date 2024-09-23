# 安装nVidia NVTX 库

NVTX，全称NVIDIA Tools Extension，是一组旨在为nVidia工具（如nsight system）提供扩展功能的API。如果应用程序未附加任何工具，则该库几乎不会带来任何开销。附加工具时的开销则特定于该工具。

> By default, NVTX API calls do *nothing*. When you launch a program from a developer tool, NVTX calls in that program are redirected to functions in the tool. Developer tools are free to implement NVTX API calls however they wish.
>
> 说明NVTX 库相当于一组上层的包装？

### NVTX 库能做什么？

##### （1）Markers

Markers annotate **a specific point** in a program's execution with a message. Optional extra fields may be provided: a category, a color, and a payload value.

##### （2）Ranges（重要！！！）

Ranges annotate **a range between two points in a program's execution**, like a related pair of markers. 

> 正是我想要的，使用NVTX 库提供的范围功能把nsight system的追踪范围限定在最后几个iteration

There are **two types** of ranges:

Push/Pop ranges, which can be nested to form a stack
The Pop call is automatically associated with a prior Push call on the same thread

Start/End ranges, which may overlap with other ranges arbitrarily
The Start call returns a handle which must be passed to the End call

Push/Pop ranges是自动按照规则结合的，而Start/End ranges更精确，需要指定，可以与Push/Pop ranges重合

**These ranges can start and end on different threads**
The C++ and Python interfaces provide objects and decorators for automatically managing the lifetimes of ranges.

##### （3）Resource naming/tracking

Resource naming associates a displayable name string with an object. For example, naming CPU threads allows a tool that displays thread activity on a timeline to have more meaningful labels for its rows than a numeric thread ID.

NVTX 库还可以用来给资源对象起名，比如某个线程；可以实现对象的生命周期追踪；

Resource tracking extends the idea of naming to include **object lifetime tracking**, as well as important usage of the object. For example, a mutex provided by platform API (e.g. pthread_mutex, CriticalSection) can be tracked by a tool that intercepts its lock/unlock API calls, so using NVTX to name these mutex objects would be sufficient to see the names of mutexes being locked/unlocked on a timeline. However, manually implemented spin-locks may not have an interceptible API, so tools can't automatically detect when they are used. Use NVTX to annotate these types of mutexes where they are locked/unlocked to enable tools to track them just like standard platform API mutexes.

##### （4）高级用法：Domains

Domains enable developers to scope annotations. By default, all events and annotations are in the default domain. Additional domains can be registered. This allows developers to scope markers, ranges, and resources names to avoid conflicts.

相当于权限与优先级管理机制？避免太多的range和markers相互冲突

### 能够与NVTX 库配合使用的工具

- Nsight Systems 在时间线上记录 NVTX 调用，并将其与驱动程序/操作系统/硬件事件一起显示。
- Nsight Compute 使用 NVTX 范围来确定深入进行 GPU 性能分析的位置。
- Nsight Graphics 使用 NVTX 范围在帧调试器中设置范围分析的界限。
- CUPTI API 支持记录 NVTX 调用的跟踪。

支持的平台：Windows、Linux and other POSIX-like platforms (including cygwin)、Android

### 检验是否安装：

```python
try:
    import nvtx
    print("NVTX is installed.")
except ImportError:
    print("NVTX is not installed.")
```

### 安装NVTX 库

> https://github.com/NVIDIA/NVTX/tree/release-v3/python python专门版文档
>
> 版本要求：NVTX Python requires **Python 3.6 or newer**. It has been tested on Linux, with Python 3.6 to 3.9.

```
conda install -c conda-forge nvtx
```

### python nvtx库文档

> [nvtx - Annotate code ranges and events in Python — nvtx 0.2 ...](https://nvtx.readthedocs.io/en/latest/index.html)