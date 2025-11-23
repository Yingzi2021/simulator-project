# Detailed Explanation of the Simulation Process

> https://github.com/flexflow/FlexFlow.git
>
> commit `fc9c6b7`
>
> `./script/simulator.cc`

The DNN Execution Simulator models the execution of deep neural networks by simulating computation tasks and data transfers across multiple workers (e.g., GPUs). After constructing a specific DNN network and defining its configuration, the simulator performs a detailed simulation to estimate the total execution time. This process involves several key steps:

## Overview of the Simulation Workflow

1. **Model Construction**: Build the DNN model using the provided building blocks (operations like convolution, pooling, LSTM, etc.).
2. **Configuration Assignment**: Assign configurations (`OpConfig`) to each operation, specifying how the operation is partitioned and mapped to workers.
3. **Task Generation**: Convert operations into computational tasks, respecting the configurations and dependencies.
4. **Dependency Resolution**: Establish dependencies between tasks based on data flow between operations.
5. **Scheduling and Execution Simulation**:
   - Schedule tasks respecting dependencies and worker availability.
   - Simulate computation times and communication overheads.
   - Update the ready queue and track task completion times.
6. **Performance Metrics Calculation**: Calculate total execution time, data transfer volumes, and other relevant metrics.

Let's delve deeper into each of these steps, focusing on how the simulation operates after the DNN network and configurations are known.

## Step-by-Step Simulation Process

### 1. Model Construction

The DNN model is built using classes derived from the `Op` base class. Each operation (e.g., `Conv2D`, `LSTM`, `Softmax`) represents a layer or component in the DNN. Tensors (`Tensor` class) represent the data flowing between operations.

**Example**:

```cpp
Tensor x(4, BATCH_SIZE, 3, 224, 224, NULL, 0);
Tensor t = add_conv_layer(x, 64, 11, 11, 4, 4, 2, 2, "conv1");
// ... additional layers ...
```

### 2. Configuration Assignment

Each operation is assigned an `OpConfig` that specifies:

- **Partitioning Dimensions (`dim`)**: How the operation's workload is divided.
- **Number of Parts (`nParts`)**: The total number of partitions.
- **Mapping to Workers (`map`)**: Which worker (GPU) handles each partition.

**Example**:

```cpp
OpConfig config;
config.nDims = 1;
config.nParts = NUM_PARTITIONS;
config.dim[0] = NUM_PARTITIONS;
for (int j = 0; j < config.nParts; j++)
    config.map[j] = j % NUM_WORKERS;
current[guidToOp[i]] = config;
```

### 3. Task Generation

Operations are converted into computational tasks (`Task` class). For each partition specified in the operation's configuration, a corresponding task is created.

**Key Points**:

- **Tasks per Operation**: The number of tasks for an operation equals the number of partitions (`nParts`) in its configuration.
- **Worker Assignment**: Each task is assigned to a worker based on the mapping in the `OpConfig`.

**Example**:

```cpp
for (int j = 0; j < config.nParts; j++) {
    tasks[i][j] = new Task(config.map[j], op->compute(config));
}
```

### 4. Dependency Resolution

Dependencies between tasks are established based on the data flow between operations. If an operation (`Op`) consumes the output of another operation, the corresponding tasks must respect this dependency.

**Process**:

- **Tensor Overlaps**: Determine if the output tensor partitions from the producer operation intersect with the input tensor partitions of the consumer operation.
- **Dependency Links**: If there is an overlap, a dependency is added between the producer's task and the consumer's task.

**Example**:

```cpp
for (int dstId = 0; dstId < config.nParts; dstId++) {
    Rect dstR = op->get_tensor_shape(config, dstId, t, true);
    Task* dstT = tasks[i][dstId];
    for (int srcId = 0; srcId < preConfig.nParts; srcId++) {
        Rect srcR = preOp->get_tensor_shape(preConfig, srcId, t, false);
        Task* srcT = tasks[preOp->guid][srcId];
        if (intersect(srcR, dstR) > 0) {
            // Add dependency from srcT to dstT
            srcT->add_next_task(dstT);
        }
    }
}
```

### 5. Scheduling and Execution Simulation

The simulation progresses by scheduling tasks while respecting dependencies and simulating computation and communication times.

#### a. Initialization

- **Ready Queue**: A priority queue (`readyQueue`) is initialized with tasks that have no dependencies (i.e., their `preTasks` list is empty).
- **Worker Timelines**: An array (`gpuTime`) keeps track of the current time on each worker, representing when it becomes available.

#### b. Task Execution Loop

While the ready queue is not empty:

1. **Select Task**: The task with the earliest `readyTime` is selected from the ready queue.
2. **Update Start Time**: The task's `startTime` is set to the maximum of its `readyTime` and the current time on its assigned worker.
3. **Update Worker Timeline**: The worker's current time is updated to `startTime + computeTime`.
4. **Process Next Tasks**:
   - For each task that depends on the current task (`nextTasks`), update its `readyTime` based on the completion of the current task.
   - If a task's dependencies are all satisfied, add it to the ready queue.

**Example**:

```cpp
while (!readyQueue.empty()) {
    Task* t = *readyQueue.begin();
    readyQueue.erase(readyQueue.begin());
    int gpuId = t->workerId;
    gpuTime[gpuId] = std::max(gpuTime[gpuId], t->readyTime);
    t->startTime = gpuTime[gpuId];
    gpuTime[gpuId] += t->computeTime;
    for (Edge& edge : t->nextTasks) {
        Task* next = edge.task;
        next->readyTime = std::max(next->readyTime, t->startTime + t->computeTime);
        next->counter++;
        if (next->counter == next->preTasks.size()) {
            readyQueue.insert(next);
        }
    }
}
```

#### c. Communication Time Calculation

- **Data Transfers**: When tasks are assigned to different workers and there is data dependency, the simulator calculates the communication time based on data size and bandwidth.
- **Bandwidth Considerations**: The bandwidth between workers depends on whether they are on the same node (intra-node) or different nodes (cross-node).

**Example**:

```cpp
if (srcT->workerId != dstT->workerId) {
    float cost = dataSize / bandwidth(srcT->workerId, dstT->workerId);
    totalDataXfer += dataSize;
    Task* commTask = new Task(communicationDeviceId, cost);
    srcT->add_next_task(commTask);
    commTask->add_next_task(dstT);
}
```

### 6. Performance Metrics Calculation

After all tasks have been scheduled and executed in the simulation:

- **Total Execution Time**: The maximum of the completion times across all workers.
- **Data Transfer Volume**: Sum of all data transferred between workers.
- **Computation Time**: Sum of computation times for all tasks.

**Example**:

```cpp
float totalTime = 0.0f;
for (int i = 0; i < NUM_WORKERS; i++)
    totalTime = std::max(totalTime, gpuTime[i]);

// Include update costs for parameter synchronization
for (const auto& paramOps : parameters) {
    float updateCost = paramOps[0]->update(configs);
    totalTime += updateCost;
}
```

## Factors Affecting the Simulation

### Operation Configurations (`OpConfig`)

- **Partitioning Strategy**: How operations are partitioned affects the number of tasks and data dependencies.
- **Worker Mapping**: Assigning partitions to different workers influences communication costs due to data transfers.

### Data Dependencies

- **Tensor Overlaps**: The intersection of tensor partitions between producer and consumer tasks determines if data needs to be transferred.
- **Task Dependencies**: Dependencies dictate the order of task execution and can create bottlenecks if not managed properly.

### Resource Constraints

- **Worker Availability**: Tasks are scheduled on workers based on their availability, which can cause delays if workers are busy.
- **Bandwidth Limitations**: Limited bandwidth between workers can increase communication times, especially for cross-node transfers.

## Optimization Using Simulated Annealing

The simulator employs a simulated annealing approach to explore different configurations and optimize execution time.

### Process

1. **Initialization**: Start with an initial configuration and calculate its execution time.
2. **Iteration**:
   - **Modify Configuration**: Randomly alter the configuration of one operation.
   - **Simulate**: Run the simulation with the new configuration.
   - **Acceptance Criteria**: Decide whether to accept the new configuration based on:
     - Improvement in execution time.
     - A probability function that allows for occasional acceptance of worse configurations to escape local minima.
3. **Loop**: Repeat the iteration for a predefined number of steps or until convergence.

**Example**:

```cpp
for (int i = 0; i <= MAX_ITERATIONS; i++) {
    Op* updOp = rewrite(current, next);
    float next_runtime = simulate_time(next);
    float diff = next_runtime - cur_runtime;
    if (next_runtime < optimal_runtime) {
        optimal_runtime = next_runtime;
        optimal = next;
    }
    if (next_runtime < cur_runtime || accept_worse_solution(diff, temperature)) {
        current = next;
        cur_runtime = next_runtime;
    }
    // Update temperature if using an annealing schedule
}
```

### Benefits

- **Exploration of Configuration Space**: Allows the simulator to test a wide range of configurations.
- **Avoiding Local Minima**: The probabilistic acceptance of worse configurations helps in finding a global optimum.

## Example Simulation Scenario

Assume we have built an LSTM-based neural machine translation (NMT) model using the simulator:

1. **Building the Model**:

   ```cpp
   void build_nmt_model() {
       // Initialize tensors and operations
       // Create embedding layers, LSTM layers, attention mechanisms, etc.
   }
   ```

2. **Assigning Initial Configurations**:

   ```cpp
   generate_init_config(current);
   ```

3. **Running the Simulation**:

   ```cpp
   float initial_runtime = simulate_time(current);
   ```

4. **Optimizing Configurations**:

   ```cpp
   for (int i = 0; i <= MAX_ITERATIONS; i++) {
       Op* updOp = rewrite(current, next);
       float next_runtime = simulate_time(next);
       // Decide whether to accept next configuration
   }
   ```

5. **Analyzing Results**:

   - **Total Execution Time**: The optimized total time for processing the batch.
   - **Data Transfer Volumes**: Amount of data moved between workers.
   - **Optimal Configurations**: The best `OpConfig` settings found for each operation.

## Conclusion

The simulation works by converting the DNN model and its configurations into a set of computational tasks with dependencies. By simulating the execution of these tasks across multiple workers, considering both computation and communication costs, the simulator estimates the total execution time of the model. Through optimization techniques like simulated annealing, it explores different configurations to find the most efficient execution plan.

This detailed simulation process allows users to understand the performance implications of different partitioning strategies, worker mappings, and resource allocations, enabling them to optimize DNN execution on distributed systems effectively.