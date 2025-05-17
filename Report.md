Report
===

## Part 1

結果部分截圖
![image](https://hackmd.io/_uploads/HJYAkEsrR.png)
![image](https://hackmd.io/_uploads/HkqMeVjH0.png)


## Part 2
### Q1
#### Q1-1
> A naive way of splitting the model is to split into equal size (pippy.split_into_equal_size), is this a good policy? What are the split points of this policy?

1. Even with the same number of parameters, each split part may require different amounts of computation, causing the pipeline to get stuck at the bottleneck.
2. If the split points are located between layers that require frequent communication, it will significantly impact performance.

#### Q1-2
> Please describe the split policy you have used in your implementation. Explain why it is better.

Time consumption for each stage is more correlated to MACs (multiply-accumulate) Therefore, splitting the model into n stages with approximately the same MACs is better.


### Q2

> In the previous setup, we split the model into 4 stages across 4 devices. Now, let's try splitting the model into more stages or less stages. Please compare the speed up between 2, 3, 4, and 6-stage pipeline.

#### 2 stage
![image](https://hackmd.io/_uploads/SyGPQ23H0.png)
![image](https://hackmd.io/_uploads/Hk_Omh3H0.png)

<!-- 4-stage pipeline's fps is 1.3890, and 2-stage pipeline's fps is 2.6900. -->

According to the result in serial mode from Part 1, the speedup is **1.937**

#### 3 stage

![image](https://hackmd.io/_uploads/ry9EtYprC.png)

According to the result in serial mode from Part 1, the speedup is **3.065**

#### 4 stage

![image](https://hackmd.io/_uploads/HkqMeVjH0.png)

The speedup is **4.0879**

### Q3
> Ideally, in a n-stage pipeline, the speedup should be close to n. However, this is not the case in practice. Please examine the model's execution and share your thoughts on why the speedup isn't close to n.

Although splitting the model into n stages increases parallelization, the pipeline also introduces additional overhead, such as data transfer costs.
#### pipeline profile
![image](https://hackmd.io/_uploads/B11c7_nHC.png)

From the profiler results of pipeline mode, it can be observed that during execution, data transfer between nodes consumes CPU time. Therefore, the speedup will not be n.

![image](https://hackmd.io/_uploads/SyBXxF2rR.png)

#### serial profile
You can see that in serial mode, there is no data transfer.

![image](https://hackmd.io/_uploads/r1XoqonSR.png)




## Part 3

> Improve Speed Up

Initially, we split the model into n stages with approximately the same MACs. However, after multiple attempts, we found that if we split the model based on transformer blocks, the performance is better.

![image](https://hackmd.io/_uploads/B1k8GF6S0.png)

Before:

![image](https://hackmd.io/_uploads/rJEiztprR.png)


After:

![image](https://hackmd.io/_uploads/SypFMKaBA.png)

We speculate the following reasons: a smaller number of MACs does not necessarily mean shorter execution time, as other factors must be considered, such as memory data transfer and data transmission bandwidth. If we split the model into n stages with approximately the same MACs, it does not mean that the execution time for each stage will be equal. As long as the times are unequal, the bottleneck of the entire pipeline will be at the stage with the longest execution time.