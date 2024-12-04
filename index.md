---
layout: page
title: "Proposal"
---

#### Summary
This project focuses on developing a parallel, dynamic tree-based speculative decoding algorithm tailored for efficient large language model (LLM) inference across multiple GPUs. Unlike SpecInfer, which relies on static tree speculative decoding, this project aims to implement a dynamic approach that constructs decoding trees on the fly, leveraging parallelism across GPUs. This design will allow simultaneous management of multiple decoding requests, dynamically scheduling them to maximize throughput and minimize latency.

***
#### Background
Speculative decoding algorithms are essential for accelerating LLM inference by enabling predictions based on a precomputed tree structure that guesses possible paths of text generation. SpecInfer, as a static tree speculative decoding approach, builds this tree once and uses it across requests, which can speed up certain types of workloads. However, static trees do not adapt dynamically to the variations in request patterns, model states, or user interactions, which can limit their efficiency when handling multiple, concurrent requests with diverse language models. A dynamic tree, adaptable in real time, enables finer control over GPU workload distribution and improves computational efficiency by adjusting to the needs of target and draft models across multiple GPUs.

***
#### Challenges

Implementing a dynamic tree speculative decoding system requires parallelizing several components:
1. Tree Construction: Building and updating decoding trees dynamically across GPUs in response to incoming requests. Each GPU must manage its share of nodes while communicating with other GPUs to maintain synchronization.
2. Multi-GPU Load Balancing: Efficiently distributing tasks for target and draft language models across GPUs to handle numerous requests in parallel.
3. Request Scheduling: Enabling sequential request processing, with the system intelligently routing tasks to GPUs and adjusting workloads in real time to reduce response time and maximize GPU utilization.

Each of these tasks involves significant inter-GPU communication and optimization of memory access patterns, making efficient parallelization a complex challenge.

***
#### Resources
This project will leverage existing speculative decoding frameworks like SpecInfer as a starting point but will implement core algorithms for dynamic tree construction and multi-GPU communication from scratch. The primary tools will include:
1. CUDA: For managing parallel operations across GPUs.
2. PyTorch or TensorFlow: For integrating the speculative decoding tree with transformer-based language models.
3. Multi-GPU cluster access at CMU (potentially Gates Cluster Machines and PSC Bridges-2) for testing scalability and performance.

***
#### Goals and Deliverables
1. Primary Goals:
    * Implement a parallel, dynamic tree-based speculative decoding algorithm for multi-GPU inference.
    * Achieve efficient load balancing between target and draft models across GPUs.
    * Benchmark and compare the speedup of dynamic speculative decoding against a static tree-based approach.
2. Stretch Goals:
    * Provide a graphical or statistical representation of the speedup, illustrating efficiency gains under various workloads.
    * Explore advanced scheduling techniques that adapt in real time to changing request patterns and model states.


***
#### Platform Choice
This project will be implemented in C++ with CUDA and OpenMPI for the low-level GPU and inter-GPU communication, alongside Python for high-level model and system integration.

***
#### Schedule

* Week 1: Literature Review; Design and set up initial benchmarking with SpecInfer (a static tree-based decoding) for baseline performance.
* Week 2: Implement a sequential version of our dynamic tree-based speculative decoding algorithm
* Week 3: Derive a simple parallel approach using tensor parallelism provided by PyTorch
* Week 4: Provide a parallel implementation of the algorithm using CUDA
* Week 5: Benchmark and compare performance between different versions of the algorithm
* Week 6: Complete final performance evaluation, prepare final report, and create any visualizations for demonstration.

***
#### Sources
* Original SpecInfer documentation and academic papers on speculative decoding.
* References on tree-based and mesh-based parallel algorithms for dynamic GPU scheduling and load balancing.
