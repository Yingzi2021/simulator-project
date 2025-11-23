# Insight From The Survey

[TOC]

## 1. Different Motivations

### Find a problem

**Discover an important problem and try to solve it** 

In such papers, a dedicated section is often used to present the key observations, supported by experimental results that convincingly demonstrate the problem's significance and validity. Conducting such research is challenging, as it requires a combination of strong analytical skills and, at times, some luck. 

**take-away:** Not suitable for me.

> **Enabling  Compute-Communication Overlap in     Distributed Deep Learning Training Platforms**
>
> **ARK:  GPU-driven Code Execution for Distributed Deep Learning**

### Solve a problem

**Propose a novel solution for an important problem**

In this type of paper, the target problem is not entirely new. Various approaches may have been proposed to address it, yet no definitive or optimal solution has been achieved. Such papers contribute by introducing a novel solution that, while not flawless, offers a foundational idea to inspire subsequent research and provides a new direction for the community.  

**take-away:** The "first" paper in a small field.

> **Poseidon:  An Efficient Communication Architecture for Distributed Deep Learning on GPU  Clusters**
>
> - The inefficiency of distributed training in existing ML frameworks is **a well-recognized issue**, especially as DNN models grow increasingly large. This problem is critical and has drawn significant attention.
> - This paper proposes a novel approach to address the problem by overlapping computation and communication. Although their solution is not perfect (later studies reveal inefficiency in their layer-by-layer overlapping strategy for deeper DNNs), it was the first work to explore this concept in the context of DNN training, laying the groundwork for further research.
>
> **Paleo:  A Performance Model for Deep Neural Networks**
>
> - As DNN models scale, locating the optimal parallel configurations from the vast design space of scalable deep learning systems remains a challenging and unresolved problem. First raised in 2015, this issue has driven the development of performance modeling for distributed DNN training, highlighting its significance and need for innovative solutions.
> - this paper address this problem by making a key observation: the architecture of a neural network inherently defines computational dependencies associated with its training and evaluation. Based on this insight, the author proposes a novel graph-based model to concisely capture these dependencies. As the first work to apply a graph-based approach in this context, it has inspired subsequent research, leading to a widespread adoption of graph-based models for performance modeling.

**Propose the best solution for an important problem.**

In such papers, the authors present a unified approach to address a critical issue by optimizing and integrating existing tools to a single framework. This type of work is typically implementation-intensive and often better suited for large research groups in industrial settings.

**take-away:** Not suitable for me. it's too difficult for individuals to compete with large group.

> **SimAI:  Unifying Architecture Design and Performance Tunning for Large-Scale Large  Language Model Training with Scalability and Precision**

**Propose a novel solution for an important problem in a highly-specific scenario.**

In such papers, the author focus on addressing a critical problem within a highly specific scenario that shares similarities with an existing, well-studied problem. The novelty of these works lie in their ability to adapt traditional solutions through scenario-specific modifications, leveraging the unique characteristics of the target context to achieve superior performance. 

> **Srifty:  Swift and Thrifty Distributed Neural Network Training on the Cloud**

**take-away:** While it requires expertise in that specific field, it worth trying.

**Propose a novel solution for an important problem by adapting well-established methods from a different field**

In such papers, the authors address a critical problem within their field by leveraging well-established method from another discipline. Although the problem may initially appear unrelated to the external field, it is often found that integrating multiple mature techniques from that field provides an effective solution. This cross-disciplinary approach serves as a valuable source of innovation. However, if too much solutions are combined, leading to a complex system with suboptimal performance, the approach may face skepticism regarding its efficacy.

**take-away:**  他山之石，可以攻玉

> **FasterMoE:  modeling and optimizing training of large-scale dynamic pre-trained models**

### Follow up work

**Propose direct enhancement to an existing method**

In such papers, the introduction often mirrors the previous work they aims to build upon, acknowledging its contributions while identifying specific limitations. The authors argue that by introducing thoughtful modifications, their work addresses these shortcomings and achieves improved performance. One common concern regarding the type of works is that they tend to be method-driven, focusing on incremental improvements to existing techniques rather than addressing fundamentally important problems. While this approach can be more accessible for beginners, it may risk lacking novelty—unless the performance gains are exceptionally significant.

> **Overlapping Communication With Computation in Parameter Server for Scalable DL Training**

## 2. Research Topic

> utilize open-source project

**Candidate 1**: Add more detailed comp & comm overlap modeling to https://github.com/JF-D/Proteus

**Candidate 2**: Use early version of [FlexFlow](https://flexflow.ai/) to build a toy performance simulator, probably add some enhancement related to comp & comm overlap.

**Candidate 3**: Use [SimAI](https://github.com/aliyun/SimAI) to simulate distributed training performance on a large cluster. Do some profile with standard workload https://github.com/mlcommons/chakra

**Candidate 4**: Try to understand the "bandwidth fragmentation" problem as DNN training scales. Simulator becomes a module in the training system and its online performance estimation result serves as some metric for dynamic scheduling, etc.

> **ARK:  GPU-driven Code Execution for Distributed Deep Learning (NSDI 2023):**
>
> The communication overhead mainly arises in two different aspects. First, collective communication (e.g., all-reduce, splitand-gather, all-to-all, etc.), which is widely adopted in most of popular DL algorithms, often splits the data for transfer into multiple small chunks for pipelining or for sending to multiple different destinations. The chunk size tends to get smaller as we scale out, which is detrimental to efficient utilization of networking bandwidth.
>
> **Centauri: Enabling Efficient Scheduling for Communication-Computation Overlap in Large Model Training via Communication Partitioning (ASPLOS 2024 ⭐)**
>
> In contrast to the static and intensive computation pattern observed in LLMs training, parallel LLMs inference faces challenges associated with small-volume but frequent communication overheads.

## 3. Writing

- Title should be **concise** and **precise**. It tells reader what is your work and why it is special.  
- Every word in abstract serves for a purpose.
- Background is important but could be infinite. Only mention the part which is most related to your work in the introduction. 
