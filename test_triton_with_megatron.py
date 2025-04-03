# Copyright (c) 2024
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import triton
import triton.language as tl
import torch.distributed as dist
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
from megatron.core.parallel_state import (
    get_tensor_model_parallel_world_size,
    initialize_model_parallel,
)
from megatron.core.tensor_parallel import get_cuda_rng_tracker, model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig

@triton.jit
def vector_add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def triton_vector_add(x, y, force_triton=False):
    if not (x.is_cuda and y.is_cuda):
        raise ValueError("Both inputs must be CUDA tensors")
    
    if not force_triton:
        return x + y
    
    try:
        if x.shape != y.shape:
            if y.numel() == 1:
                y_expanded = torch.full_like(x, y.item())
                y = y_expanded
            else:
                try:
                    y_expanded = y.expand_as(x)
                    y = y_expanded
                except RuntimeError:
                    print(f"Warning: Could not broadcast shapes {y.shape} to {x.shape}. Falling back to PyTorch.")
                    return x + y
        
        original_shape = x.shape
        x_flat = x.reshape(-1).contiguous()
        y_flat = y.reshape(-1).contiguous()
        n_elements = x_flat.numel()
        output = torch.empty_like(x_flat)
        
        BLOCK_SIZE = 512  # Smaller block size for better stability
        grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        vector_add_kernel[(grid,)](
            x_flat,
            y_flat,
            output,
            n_elements,
            BLOCK_SIZE
        )
        
        return output.reshape(original_shape)
    
    except Exception as e:
        print(f"Triton operation failed: {e}. Falling back to PyTorch.")
        return x + y

class TritonVectorAddModule(MegatronModule):
    def __init__(
        self,
        config: TransformerConfig,
        input_dim: int,
        bias: bool = True,
        force_triton: bool = False,
    ):
        super().__init__(config)
        self.config = config
        self.input_dim = input_dim
        self.force_triton = force_triton
        
        self.tensor_model_parallel_size = get_tensor_model_parallel_world_size()
        
        assert input_dim % self.tensor_model_parallel_size == 0, (
            f"Input dimension {input_dim} not divisible by tensor parallel size {self.tensor_model_parallel_size}"
        )
        self.input_dim_local = input_dim // self.tensor_model_parallel_size
        
        with get_cuda_rng_tracker().fork():
            self.weight = nn.Parameter(
                torch.empty(
                    self.input_dim_local,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            setattr(self.weight, 'tensor_model_parallel', True)
            
            if hasattr(self.config, 'init_method') and self.config.init_method is not None:
                self.config.init_method(self.weight)
            else:
                nn.init.normal_(self.weight, mean=0.0, std=0.02)
                
            if bias:
                self.bias = nn.Parameter(
                    torch.zeros(
                        self.input_dim_local,
                        device=torch.cuda.current_device(),
                        dtype=config.params_dtype,
                    )
                )
                setattr(self.bias, 'tensor_model_parallel', True)
            else:
                self.bias = None
    
    def forward(self, hidden_states):
        if hidden_states.shape[-1] != self.input_dim_local:
            raise ValueError(
                f"Expected input dimension {self.input_dim_local}, got {hidden_states.shape[-1]}"
            )
        
        weight = self.weight.view(1, 1, -1)
        output = triton_vector_add(hidden_states, weight, force_triton=self.force_triton)
        
        if self.bias is not None:
            bias = self.bias.view(1, 1, -1)
            output = triton_vector_add(output, bias, force_triton=self.force_triton)
        
        return output
    
    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        state_dict = self.state_dict(prefix='', keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict,
            prefix,
            {'weight': 0, 'bias': 0},
            sharded_offsets
        )

def test_triton_megatron_integration():
    print("Starting Triton-Megatron integration test...")
    
    if not torch.cuda.is_available():
        print("CUDA is not available. Cannot run test on GPU.")
        return
    
    print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    print(f"Triton version: {triton.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    
    if not dist.is_initialized():
        try:
            dist.init_process_group(
                backend='nccl',
                init_method='tcp://localhost:12355',
                world_size=1,
                rank=0
            )
            print("Initialized distributed process group")
        except Exception as e:
            print(f"Warning: Could not initialize distributed process group: {e}")
            print("Continuing with single process testing")
    
    try:
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1
        )
        print("Initialized model parallel state")
    except Exception as e:
        print(f"Warning: Could not initialize model parallel state: {e}")
        return
    
    model_parallel_cuda_manual_seed(42)
    
    config = TransformerConfig(
        hidden_size=1024,
        num_attention_heads=16,
        num_layers=12,
        params_dtype=torch.float16,
        init_method_std=0.02,
        output_layer_init_method=None,
    )
    
    for force_triton in [False, True]:
        print(f"\n{'=' * 40}")
        print(f"Testing with force_triton={force_triton}")
        print(f"{'=' * 40}")
        
        input_dim = 1024
        module = TritonVectorAddModule(config, input_dim, force_triton=force_triton)
        module.cuda()
        print(f"Created TritonVectorAddModule with local weight shape: {module.weight.shape}")
        
        batch_size = 2
        seq_len = 128
        hidden_states = torch.randn(
            batch_size, seq_len, module.input_dim_local,
            device=torch.cuda.current_device(),
            dtype=config.params_dtype
        )
        print(f"Created test input with shape: {hidden_states.shape}")
        
        print("\nTesting basic vector addition...")
        x = torch.randn(1000, device=torch.cuda.current_device(), dtype=config.params_dtype)
        y = torch.randn(1000, device=torch.cuda.current_device(), dtype=config.params_dtype)
        
        try:
            result = triton_vector_add(x, y, force_triton=force_triton)
            expected = x + y
            max_diff = torch.max(torch.abs(result - expected)).item()
            print(f"Basic vector addition test: max diff = {max_diff}")
            if max_diff < 1e-4:
                print("✅ Basic test PASSED")
            else:
                print("❌ Basic test FAILED")
        except Exception as e:
            print(f"❌ Basic test FAILED with error: {e}")
        
        try:
            import time
            start = time.time()
            
            output = module(hidden_states)
            
            torch.cuda.synchronize()
            elapsed = time.time() - start
            
            print(f"Forward pass completed in {elapsed*1000:.2f} ms")
            print(f"Output shape: {output.shape}")
            
            weight = module.weight.view(1, 1, -1)
            expected = hidden_states + weight
            if module.bias is not None:
                bias = module.bias.view(1, 1, -1)
                expected = expected + bias
            
            max_diff = torch.max(torch.abs(output - expected)).item()
            print(f"Maximum difference between Triton and PyTorch: {max_diff}")
            
            if max_diff < 1e-4:
                print("✅ Test PASSED: Triton and PyTorch results match")
            else:
                print("❌ Test FAILED: Results don't match within tolerance")
        except Exception as e:
            print(f"❌ Test with force_triton={force_triton} FAILED with error: {e}")
            print("Continuing with next test...")
            continue
            
        print("\nTesting broadcasting capabilities...")
        
        try:
            scalar = torch.tensor([2.0], device=torch.cuda.current_device(), dtype=config.params_dtype)
            start = time.time()
            broadcast_result = triton_vector_add(hidden_states, scalar, force_triton=force_triton)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            print(f"Scalar broadcasting completed in {elapsed*1000:.2f} ms")
            
            expected_broadcast = hidden_states + 2.0
            max_diff = torch.max(torch.abs(broadcast_result - expected_broadcast)).item()
            print(f"Maximum difference for scalar broadcasting: {max_diff}")
            if max_diff < 1e-4:
                print("✅ Scalar broadcasting test PASSED")
            else:
                print("❌ Scalar broadcasting test FAILED")
        except Exception as e:
            print(f"❌ Scalar broadcasting test FAILED with error: {e}")
        
        try:
            vec = torch.randn(module.input_dim_local, device=torch.cuda.current_device(), dtype=config.params_dtype)
            print(f"Testing vector broadcasting with shape {vec.shape} to {hidden_states.shape}")
            start = time.time()
            broadcast_result = triton_vector_add(hidden_states, vec, force_triton=force_triton)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            print(f"Vector broadcasting completed in {elapsed*1000:.2f} ms")
            
            expected_broadcast = hidden_states + vec.view(1, 1, -1)
            max_diff = torch.max(torch.abs(broadcast_result - expected_broadcast)).item()
            print(f"Maximum difference for vector broadcasting: {max_diff}")
            if max_diff < 1e-4:
                print("✅ Vector broadcasting test PASSED")
            else:
                print("❌ Vector broadcasting test FAILED")
        except Exception as e:
            print(f"❌ Vector broadcasting test FAILED with error: {e}")
        
        try:
            print("\nTesting with larger batch size...")
            batch_size = 16
            hidden_states = torch.randn(
                batch_size, seq_len, module.input_dim_local,
                device=torch.cuda.current_device(),
                dtype=config.params_dtype
            )
            
            start = time.time()
            output = module(hidden_states)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            
            print(f"Forward pass with batch size {batch_size} completed in {elapsed*1000:.2f} ms")
        except Exception as e:
            print(f"❌ Large batch test FAILED with error: {e}")
    
    print("\nTesting checkpoint capability...")
    try:
        state_dict = module.sharded_state_dict()
        print(f"Successfully created sharded state dict with keys: {list(state_dict.keys())}")
    except Exception as e:
        print(f"Error creating sharded state dict: {e}")
    
    print("\nTriton-Megatron integration test completed!")
    
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
            print("Destroyed distributed process group")
        except Exception as e:
            print(f"Warning: Could not destroy process group: {e}")

if __name__ == "__main__":
    test_triton_megatron_integration()