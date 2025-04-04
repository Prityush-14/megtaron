#!/usr/bin/env python3
# Copyright (c) 2024
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import subprocess
import torch.distributed as dist
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Launch 5D parallel test')
    parser.add_argument('--world_size', type=int, default=8,
                       help='Total number of GPUs to use')
    parser.add_argument('--tensor_parallel_size', type=int, default=2,
                       help='Number of GPUs for tensor parallelism')
    parser.add_argument('--pipeline_parallel_size', type=int, default=2,
                       help='Number of GPUs for pipeline parallelism')
    parser.add_argument('--sequence_parallel', action='store_true',
                       help='Enable sequence parallelism')
    parser.add_argument('--expert_parallel_size', type=int, default=1,
                       help='Number of GPUs for expert parallelism')
    parser.add_argument('--data_parallel_size', type=int, default=2,
                       help='Number of GPUs for data parallelism')
    parser.add_argument('--master_addr', type=str, default='localhost',
                       help='Master node address')
    parser.add_argument('--master_port', type=str, default='12355',
                       help='Master node port')
    return parser.parse_args()

def validate_args(args):
    # Validate total world size matches product of parallel sizes
    total_parallel_size = (
        args.tensor_parallel_size *
        args.pipeline_parallel_size *
        (2 if args.sequence_parallel else 1) *
        args.expert_parallel_size *
        args.data_parallel_size
    )
    assert total_parallel_size == args.world_size, (
        f"Total parallel size {total_parallel_size} does not match world size {args.world_size}"
    )
    
    # Check if enough GPUs are available
    num_gpus = torch.cuda.device_count()
    assert num_gpus >= args.world_size, (
        f"Not enough GPUs available. Required {args.world_size}, found {num_gpus}"
    )

def launch_distributed_test(args):
    # Set environment variables for distributed training
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    
    # Launch processes
    processes = []
    for rank in range(args.world_size):
        env = os.environ.copy()
        env['RANK'] = str(rank)
        env['WORLD_SIZE'] = str(args.world_size)
        env['LOCAL_RANK'] = str(rank % torch.cuda.device_count())
        
        cmd = [
            sys.executable,
            'test_triton_with_megatron.py',
        ]
        
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        processes.append(process)
    
    # Wait for all processes to complete
    for rank, process in enumerate(processes):
        stdout, stderr = process.communicate()
        
        # Print output with rank prefix
        for line in stdout.splitlines():
            print(f"[Rank {rank}] {line}")
        
        if stderr:
            print(f"[Rank {rank} ERROR] {stderr}", file=sys.stderr)
        
        if process.returncode != 0:
            print(f"Process {rank} failed with return code {process.returncode}")
            # Kill remaining processes
            for p in processes:
                if p.poll() is None:
                    p.kill()
            sys.exit(1)

def main():
    args = parse_args()
    try:
        validate_args(args)
    except AssertionError as e:
        print(f"Validation failed: {e}")
        sys.exit(1)
    
    print("Launching 5D parallel test with configuration:")
    print(f"  World size: {args.world_size}")
    print(f"  Tensor parallel size: {args.tensor_parallel_size}")
    print(f"  Pipeline parallel size: {args.pipeline_parallel_size}")
    print(f"  Sequence parallel: {args.sequence_parallel}")
    print(f"  Expert parallel size: {args.expert_parallel_size}")
    print(f"  Data parallel size: {args.data_parallel_size}")
    print(f"  Master address: {args.master_addr}")
    print(f"  Master port: {args.master_port}")
    
    launch_distributed_test(args)

if __name__ == '__main__':
    main() 