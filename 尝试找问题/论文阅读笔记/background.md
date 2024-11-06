# 读论文过程中学到的背景信息

分布式训练是如何“并行”的

In addition to SIMD (single instruction multiple data) and thread parallelism on a single machine, these systems also exploit model parallelism and data parallelism across machines. Model parallelism partitions DNN across machines that we call workers. Each worker trains a portion of the DNN concurrently and a collection of workers that make up a DNN is called a replica. Data parallelism partitions training data to enable parallel training of multiple DNN replicas. To ensure convegence, replicas periodically exchange weight values through parameter servers, which maintains an updated global copy of the weights.

