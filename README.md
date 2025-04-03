# megtaron

A high-performance integration between Megatron-LM and Triton for accelerated tensor operations in large language models. This project provides optimized CUDA kernels using Triton for vector operations while maintaining compatibility with Megatron's tensor parallel infrastructure.

## Features
- Optimized Triton kernels for vector addition with automatic fallback to PyTorch
- Integration with Megatron's tensor parallelism and model sharding
- Support for mixed precision (FP16, BF16, FP32)
- Automatic broadcasting and shape handling
- Built-in performance testing and validation

## Requirements
- PyTorch with CUDA support
- Triton
- Megatron-Core
- NVIDIA GPU with compute capability 7.0 or higher

Tested on RTX 3090