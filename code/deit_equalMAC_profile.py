# Copyright (c) Meta Platforms, Inc. and affiliates
# test
import torch
from torch import nn
import time
import numpy as np
from tqdm.auto import tqdm
import timm
import pippy
from pippy.IR import *
from pippy.PipelineStage import PipelineStage

from util import *

import os
import copy
import sys

import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity
import logging

import argparse

# parallel-scp -h ~/hosts.txt -r ~/<code dir> ~/
# torchrun   --nnodes=4   --nproc-per-node=1   --node-rank=0   --master-addr=192.168.1.xxx   --master-port=50000   template.py

def getVeryMiniTestDataset():
    # Create a test_loader with batch size = 1
    _, test_loader, _ = prepare_data(batch_size=1)

    # Prepare to collect 2 images per class
    class_images = [[] for _ in range(100)]

    # Iterate through the data
    for (image, label) in test_loader:
        if len(class_images[label]) < 2:
            class_images[label].append((image, label))
        if all(len(images) == 2 for images in class_images):
            break  # Stop once we have 2 images per class

    # flatten class_images
    mini_test_dataset = []
    for images in class_images:
        mini_test_dataset.extend(images)

    
    images, labels = zip(*mini_test_dataset)

    images = torch.stack(images)
    labels = torch.tensor(labels)

    return (images, labels)

def main():

    # Do Not Modify !!!
    #########################
    CHUNK_SIZE = 1
    # 113/06/16 SmallDragon: profile needs a lot of memory, so I reduce the amount of testing data
    # NUM_CHUNKS = 500
    # NUM_IMGS = 500
    NUM_CHUNKS = 200
    NUM_IMGS = 200
    # WARMUP = 1
    WARMUP = 0
    # NUM_TEST = 5
    NUM_TEST = 1
    #########################


    
    DEVICE = torch.device("cpu")
    torch.manual_seed(0)
        
    import os
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    os.environ["TP_SOCKET_IFNAME"]="eth0" 
    os.environ["GLOO_SOCKET_IFNAME"]="eth0"
    os.environ["GLOO_TIMEOUT_SECONDS"] = "3600"

    # TODO: Init process group
    ############### YOUR CODE STARTS HERE #################
    dist.init_process_group(backend='gloo', init_method='env://', world_size=world_size, rank=rank)
    #######################################################    


    print(f"\n**************** My Rank: {rank} ****************", file=sys.stderr)
    print(f'RANK:{os.environ["RANK"]}', file=sys.stderr)
    print(f'LOCAL_RANK:{os.environ["LOCAL_RANK"]}', file=sys.stderr)
    print(f'WORLD_SIZE:{os.environ["WORLD_SIZE"]}', file=sys.stderr)
    print(f'LOCAL_WORLD_SIZE:{os.environ["LOCAL_WORLD_SIZE"]}', file=sys.stderr)
    print(f'intra op threads num: {torch.get_num_threads()} | inter op threads num: {torch.get_num_interop_threads()}', file=sys.stderr, end='\n\n')  # You can set number of threads on your own

    # 113/06/16 SmallDragon: profile needs a lot of memory, so I reduce the amount of testing data
    # images, labels = getMiniTestDataset()
    images, labels = getVeryMiniTestDataset()
    
    # Image data for pipeline
    one_batch_images = images.squeeze(1)


    # TODO: Split the model and build the pipeline
    ############### YOUR CODE STARTS HERE #################
    model = torch.load('./0.9099_deit3_small_patch16_224.pth', map_location='cpu')
    model = model.to(DEVICE)

    # split by equal MAC
    # annotate_split_points(model, {f"blocks.{(i + 1) * 3}": PipeSplitWrapper.SplitPoint.BEGINNING for i in range(3)})
    split_points = {'Block.2.Mlp': PipeSplitWrapper.SplitPoint.END,
                    'Block.5.Mlp': PipeSplitWrapper.SplitPoint.END,
                    'Block.8.Mlp': PipeSplitWrapper.SplitPoint.END,}
    annotate_split_points(model, split_points)

    example_input = one_batch_images
    pipe = Pipe.from_tracing(model, NUM_CHUNKS, example_args=(example_input,))
    # print(pipe)

    stage = PipelineStage(pipe, rank, DEVICE)




    #######################################################

    '''
    Running Pipeline
    '''

    fps_list = []
        
    print("Testing Pipeline...", file=sys.stderr)
    with torch.no_grad():

        for i in range(1, NUM_TEST+WARMUP+1):
            
            '''
            To be fair, all threads has to be on same point
            '''

            if i <= WARMUP:
                print(f"Warmup Epoch {i}/{WARMUP}", file=sys.stderr)
            else:
                print(f"Epoch {i-WARMUP}/{NUM_TEST}", file=sys.stderr)
            
            dist.barrier()

            start_time = time.perf_counter()
            # pipeline_output = run_stage(stage=stage, rank=rank, world_size=world_size, imgs=one_batch_images)
            pipeline_output = run_stage_with_profiler(stage=stage, rank=rank, world_size=world_size, dataset=one_batch_images)
            end_time = time.perf_counter()

            elapsed_time = torch.tensor(end_time - start_time)

            dist.barrier()

            dist.reduce(elapsed_time, dst=world_size-1, op=torch.distributed.ReduceOp.MAX)

            if rank == world_size-1:
                print(f"Elapsed Time: {elapsed_time.item()}", file=sys.stderr)

            if i <= WARMUP:
                continue

            if rank == world_size - 1:
                fps = NUM_IMGS / elapsed_time.item()
                fps_list.append(fps)

            dist.barrier()
            time.sleep(5)

    if rank == world_size - 1:
        pipeline_fps = np.mean(fps_list)
        print('Throughput with %d pipeline stages: %.4f (fps)'%(world_size, pipeline_fps), file=sys.stdout)
        

    dist.barrier()


    # '''
    # Reference output
    # '''

    # print("Generating Reference Output...", file=sys.stderr)

    # with torch.no_grad():
    #     reference_output = run_serial(model=model, imgs=images)

    if rank == world_size - 1:
    #     torch.testing.assert_close(pipeline_output, reference_output)

    #     print(" Pipeline parallel model ran successfully! ".center(80, "*"), file=sys.stderr, end='\n\n')

        acc = evaluate_output(pipeline_output, labels)

    dist.barrier()


    # TODO: destroy process group
    ############### YOUR CODE STARTS HERE #################
    dist.destroy_process_group()
    #######################################################    
   

if __name__ == "__main__":
    
    # rank = int(os.environ["RANK"])

    # with profile(
    #     activities=[ProfilerActivity.CPU],
    #     profile_memory=True,
    #     record_shapes=True
    # ) as prof:
    #     # Your Code
    #     main()

    # prof.export_chrome_trace(
    #     f"{os.path.splitext(os.path.basename(__file__))[0]}_{rank}.json"
    # )
    main()