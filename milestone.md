---
layout: page
title: "Milestone"
---

#### Original Schedule 
* Week 1: Literature Review; Design and set up initial benchmarking with SpecInfer (a static tree-based decoding) for baseline performance.
* Week 2: Implement a sequential version of our dynamic tree-based speculative decoding algorithm
* Week 3: Derive a simple parallel approach using data parallelism provided by PyTorch Distributed Data Parallel (DDP)
* Week 4: Provide a parallel implementation of the algorithm using CUDA
* Week 5: Benchmark and compare performance between different versions of the algorithm
* Week 6: Complete final performance evaluation, prepare final report, and create any visualizations for demonstration.

***
#### Project Progress
At this point, we have conducted the necessary literature review and finished setting up the initial benchmarking experiment with SpecInfer, a static tree-based speculative-decoding algorithm that serves as a baseline. We also implemented a sequential version of the proposed dynamic tree-based speculative-decoding algorithm. Additionally, we were able to implement a simple parallel version of the algorithm that leverages the data-parallelism capabilities provided by PyTorch. Initial benchmarking shows that the parallel version exhibits speedup against the sequential one. 

Compared with the schedule in the project proposal, we believe we are currently on track to finish the project on time. We are currently implementing and debugging a CUDA parallel version of the algorithm, which we expect to finish by the end of this week. We aim to conduct more comprehensive benchmark testings to measure the speed-up of the parallel implementation, and make incremental improvements based on analysis.

***
#### Updated upcoming schedule

| Schedule | Task | Person In Charge |
|----------|------|-----------------|
| Dec 1st - Dec 8th | Finish implementing the CUDA version of the proposed algorithm | Mingxiao & Tengda |
| Dec 3rd - Dec 10th | Performance benchmarking of the sequential and different parallel versions of the implementation | Mingxiao & Tengda |
| Dec 10th - Dec 13th | Analyze benchmark statistics, identify potential bottlenecks, and make iterative improvements | Mingxiao & Tengda |
| Dec 13th - Dec 15th | Write final report and prepare for poster session | Mingxiao & Tengda |

***
#### Poster Deliverables

In the poster session, we plan to present the following
* A figure illustrating how tree-based speculative decoding works
* A diagram modeling our framework, highlighting the parallel component
* Several plots showing the speed up benchmark of our parallel implementation compared to the chosen baselines
* Discussion about the current solution and future directions
  

***

#### Concerns

For our dynamic tree speculative decoding algorithm, there are many components that can be parallelized. At the moment, we focus our efforts on tree construction, where we build and update decoding-trees across GPUs in response to encoding requests. However, there are other parallelizable components that we may not have sufficient time to explore. We don’t foresee any other major blockers as of now.