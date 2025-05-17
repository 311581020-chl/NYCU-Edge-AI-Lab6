# **Lab 6: DeiT-S Pipeline Parallelism**

<span style="color:Red;">**Due Date: 6/20 23:55**</span>

:::danger
**Update (6/18)**

Accuracy in Part 3 must exceed 88%
:::

:::danger
**Update (6/3)** 

For the **<master_ip>** flag in torchrun command, please use the ip **192.168.1.1xx** instead of 140.113.194.102. For the replacement of **xx**, please refer to the machine port we assigned to your team.

Example: 
-    Machine Port **8**22: xx = **08**
-    Machine Port **16**22: xx = **16**
:::

## Introduction

In the previous labs, we have learnt about the skills of quantization and model pruning to reduce model size and speed up model inferencing. In this lab, we are going to try another speed up skill: **Pipeline Parallelism**

## Part 1: Pipeline Parallelism Implementation (40%)

In this part, you first need to know what [Pipeline Parallelism](https://huggingface.co/docs/transformers/v4.15.0/parallelism#naive-model-parallel-vertical-and-pipeline-parallel) is, and try to implement it using [torch.distributed](https://pytorch.org/docs/stable/distributed.html) and [PiPPy](https://github.com/pytorch/PiPPy). 

The zip below contains template code and environment requirements:
> [lab6.zip](https://drive.google.com/file/d/1wUuT_t-Kf-6mvINbcnc9MpePjowl6bzL/view?usp=drive_link)

### Overview
Let's think of a situation that we want to inference a model with so much parameter that an edge device can barely load them all. **In this case, pipeline parallelsim may be a solution.** Supposed that we have 4 R-Pis on hand, we can split the model into 4 stages. Each R-Pi is in-charge of 1 stage. In each iteration, a chunk of data goes through the weights of current stage and then be fowarded to the next stage. 

![image](https://hackmd.io/_uploads/ryZuHE9XR.png)

$F_i,_j: i\ indicates\ the\ stage\ id.\ j\ indicates\ the\ chunk\ id.$ 

We are using deit-s on classification task, which does not have the issues we mentioned earlier, but it is still a good opportunity to learn how pipeline parallelism works. And since the execution is in pipeline with 4 stages now, you should observe **about 3-4 times** throughput compared to running the entire model on a single machine.

### Venv tutorial
In this lab, **multiple teams** will be utilizing the same Raspberry Pi simultaneously. To avoid any potential conflicts stemming from different team environments, **each group is mandated to employ Python's venv.** Here's a concise tutorial on how to utilize venv:
* Create a Virtual Environment: 
Utilize ```python3 -m venv <env_name>``` to generate a virtual environment. Replace <env_name> with your desired name.

* Activate the Virtual Environment: 
Activate your virtual environment by running ```source <env_name>/bin/activate```.

* Deactivate the Virtual Environment:
When finished, deactivate the environment by simply running ```deactivate```.

* Optional - Delete the Virtual Environment:
If needed, delete the virtual environment directory with ```rm -r <env_name>```.


### Pipeline Tools tutorial

* [PiPPy](https://github.com/pytorch/PiPPy) - A tool for splitting models and manage communications between each stage.
* [torch.distributed](https://pytorch.org/docs/stable/distributed.html) - Distributed communication package. 
Here are some useful functions you might need:
    ```python=
    dist.init_process_group()
    dist.destroy_process_group()
    dist.barrier()
    dist.reduce()
    ```

* [torchrun](https://pytorch.org/docs/stable/elastic/run.html) - Facilitates running PyTorch code on multiple machines.
torchrun command example (run on all machine):
    ```
    torchrun\
    --nnodes=<number of machines>\
    --nproc-per-node=<num of processors per machine>\
    --node-rank=<machine rank>\
    --master-addr=<master_ip_addr>\
    --master-port=<master_port>\
    <your_program>.py
    ```
    
Additional references that might be helpful:
- [Distributed Data Parallel in PyTorch Tutorial Series](https://www.youtube.com/watch?v=-K3bZYHYHEA&list=PL_lsbAsL_o2CSuhUhJIiW0IkdT5C2wGWj&index=2)

### Parallel-scp turtorial
Since we will be running code on 4 R-Pis, we need to first set up parallel-scp:
1. ssh to one of the rpi allocated to you
2. *mkdir -p ~/.ssh*
3. *ssh-keygen -t rsa*  (Leave all empty, i.e. Press Enter)
4. Create ssh configuration file: ~/.ssh/config **(Template is given below)**
5. *cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys*
6. SSH to all 4 RPIs from the current RPI (including current one)
7. Add a host.txt which contains the user & ip & port **(Template is given below)**
8. *parallel-scp -A -h host.txt -r ~/.ssh ~*
9. Now, ssh to one of the other RPI, you should find that it doesn't require password

- Template ssh configuration file:
```javascript=
Host 140.113.194.102
        HostName 140.113.194.102
        Port ( Port We Assigned )
        User ( Your Username )

Host 140.113.194.102
        HostName 140.113.194.102
        Port ( Port We Assigned )
        User ( Your Username )

Host 140.113.194.102
        HostName 140.113.194.102
        Port ( Port We Assigned )
        User ( Your Username )

Host 140.113.194.102
        HostName 140.113.194.102
        Port ( Port We Assigned )
        User ( Your Username )
```

- Template host file (host.txt):
```javascript=
( Your Username )@140.113.194.102:( Port We Assigned )
( Your Username )@140.113.194.102:( Port We Assigned )
( Your Username )@140.113.194.102:( Port We Assigned )
( Your Username )@140.113.194.102:( Port We Assigned )
```

After all these setup are done, test the functionality of parallel-scp
```parallel-scp -h host.txt ~/tmp.txt ~```

You will see the file be copied to other RPIs


### Test your code
Run the script with your pipeline parallelism code to test speed up and screen shot the result:
```shell=
chmod +x run.sh
./run.sh <nnodes> <nproc_per_node> <node_rank> <master_ip_addr>
```

:::danger
Note that, for the comparison, TAs have set the chunk size in pipeline parallelism mode to **1** and the batch size in the serial mode to **1**. Do not modify these two setting. The speedup should be greater than **3.2**. Also, the accuracy should be same as running the whole model on single RPI.
:::

## Part 2: Experiment (40%)

**Q1(15%)** 

Q1-1(5%): A naive way of splitting the model is to split into equal size (pippy.split_into_equal_size), is this a good policy? What are the split points of this policy?

Q1-2(10%): Please describe the split policy you have used in your implementation. Explain why it is better.

*Hints: To answer this question, you can present the structure of the split model and use the trace event profiling tool (torch.profiler) to illustrate the model's execution for better explanation.*

**Q2(15%)**
In the previous setup, we split the model into 4 stages across 4 devices. Now, let's try splitting the model into more stages or less stages. Please compare the speed up between 2, 3, 4, and 6-stage pipeline. 


**Q3(10%)**
Ideally, in a *n*-stage pipeline, the speedup should be close to *n*. However, this is not the case in practice. Please examine the model's execution and share your thoughts on why the speedup isn't close to *n*.

*Hint: You can use `torch.profiler` to analyze the execution in both serial mode and pipeline mode and compare the graphs from the two modes.*


### [torch.profiler](https://pytorch.org/docs/stable/profiler.html) - a tool to analyze the model's execution

We can analyze the model's execution using torch.profiler. An illustrative example of such an analysis is presented below:
```pyhton=
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU],
    profile_memory=True,
    record_shapes=True
) as prof:

     # Your Code

prof.export_chrome_trace(
    f"{os.path.splitext(os.path.basename(__file__))[0]}_{rank}.json"
)
```


Upon completion, a .json file will be generated. Please proceed to place the file at [chrome://tracing](chrome://tracing). You will observe an interface resembling the following screenshot: 
![image](https://hackmd.io/_uploads/S1XoON9QA.png)

## Part 3: Improve Speed Up (20%)

### Report(15%)
Describe your implementation on how to further improve the performance and your result. You can use any method to improve the throughput with 4 R-Pis.

You must compare the throughput to the result of running `serial_deit.py`, which is the throughput of running the model on single R-Pi. Run the following command:
```shell
torchrun   --nnodes=4   --nproc-per-node=1   --node-rank=<node_rank>   --master-addr=<master_ip_addr>   --master-port=50000   serial_deit.py
```

### Performance Evaluation of your work(5%)
See *Evaluation Criteria*

## Hand In Policy

You will need to hand-in:
* **Any code** you used to implement pipelining
* **DeiT Model** if you have modified the original one we provided
* **url.txt** should include the URL of your HackMD report.

Please organize your submission files into a zip archive structured as follows:

```scss
YourID.zip
    ├── code/
    │     └── (any code you use)
    │
    ├── model/
    │     └── deit.pth
    │
    └── url.txt
```

## Evaluation Criteria: 
```
More than 4 times speedup:     5 points
3.8 to 4 times speedup:        3 points
3.6 to 3.8 times speedup:      1 point
Less than 3.6 times speedup:   0 points
```