#!/usr/bin/env python3
"""
Expert-aware Triton Python backend.
Intercepts inference requests, extracts MoE routing decisions,
and invokes the custom KV eviction kernel via ctypes.
"""

import json
import os
import ctypes
import numpy as np

# Try importing Triton Python backend utilities
try:
    import triton_python_backend_utils as pb_utils
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    # Define a stub for local tests
    class pb_utils:
        @staticmethod
        def get_input_tensor_by_name(request, name):
            return None
        @staticmethod
        def Tensor(name, array):
            return None
        @staticmethod
        def InferenceResponse(tensors):
            return None


class ExpertLoadTracker:
    """Exponential moving average of expert activation frequency."""
    def __init__(self, num_experts, alpha=0.95, warmup_steps=100):
        self.num_experts = num_experts
        self.alpha = alpha
        self.warmup_steps = warmup_steps
        self.load_ema = np.ones(num_experts, dtype=np.float32)
        self.step = 0

    def update(self, routing_decisions):
        """
        routing_decisions: shape [num_tokens, top_k] or flat array of expert IDs
        """
        counts = np.bincount(routing_decisions.flatten(), 
                             minlength=self.num_experts).astype(np.float32)
        total = counts.sum()
        if total > 0:
            counts /= total
        self.load_ema = self.alpha * self.load_ema + (1.0 - self.alpha) * counts
        self.step += 1

    def get_loads(self):
        return self.load_ema

    @property
    def is_warm(self):
        return self.step >= self.warmup_steps


class TritonPythonModel:
    def initialize(self, args):
        """Initializes the backend and loads the custom eviction CUDA library."""
        self.model_config = json.loads(args.get('model_config', '{}'))
        
        # Load custom kernels library
        self.lib_path = "/workspace/src/kernels/build/libmoe_kv_kernels.so"
        if not os.path.exists(self.lib_path):
            # Try workspace absolute path
            self.lib_path = "/Users/abhishekdarji/MyDrive/Adaptive-MoE-Inference-Optimization/src/kernels/build/libmoe_kv_kernels.so"
            
        self.lib = None
        if os.path.exists(self.lib_path):
            try:
                self.lib = ctypes.CDLL(self.lib_path)
                # Define argtypes and restype for custom eviction launcher
                # void launch_eviction_priority(const float* attention_scores, const int* expert_activations,
                #                              const float* expert_load_counts, float* eviction_priority,
                #                              int batch_size, int seq_len, int num_heads, int num_experts,
                #                              float expert_weight, cudaStream_t stream)
                self.lib.launch_eviction_priority.argtypes = [
                    ctypes.c_void_p,  # attention_scores (float*)
                    ctypes.c_void_p,  # expert_activations (int*)
                    ctypes.c_void_p,  # expert_load_counts (float*)
                    ctypes.c_void_p,  # eviction_priority (float*)
                    ctypes.c_int,     # batch_size
                    ctypes.c_int,     # seq_len
                    ctypes.c_int,     # num_heads
                    ctypes.c_int,     # num_experts
                    ctypes.c_float,   # expert_weight
                    ctypes.c_void_p   # stream (cudaStream_t)
                ]
                self.lib.launch_eviction_priority.restype = None
                print(f"Successfully loaded custom kernels library from: {self.lib_path}")
            except Exception as e:
                print(f"Warning: Failed to load custom kernels library from {self.lib_path}: {e}")
        else:
            print(f"Warning: Custom kernels library not found at {self.lib_path}. Running in degradation mode.")

        # Get configurations
        self.num_experts = int(self.model_config.get('parameters', {}).get('num_experts', {}).get('string_value', '64'))
        self.top_k = int(self.model_config.get('parameters', {}).get('top_k', {}).get('string_value', '2'))
        self.expert_weight = float(self.model_config.get('parameters', {}).get('expert_weight', {}).get('string_value', '0.2'))

        # Expert tracking
        self.expert_load_tracker = ExpertLoadTracker(num_experts=self.num_experts)

    def execute(self, requests):
        """Executes inference for requests and triggers custom KV Cache eviction policies."""
        responses = []
        for request in requests:
            if not HAS_TRITON:
                responses.append(None)
                continue
                
            input_ids_tensor = pb_utils.get_input_tensor_by_name(request, 'INPUT_IDS')
            if input_ids_tensor is None:
                # Fallback to direct input name
                input_ids_tensor = pb_utils.get_input_tensor_by_name(request, 'input_ids')
                
            input_ids = input_ids_tensor.as_numpy()
            
            # Forward pass inference call to standard TRT-LLM backend
            output_ids, attention_scores, routing_decisions = self._forward_trt_llm(request, input_ids)
            
            # Update the expert load statistics using tracking decisions
            if routing_decisions is not None:
                self.expert_load_tracker.update(routing_decisions)
                
            # If the tracker has gathered enough statistics and CUDA library is loaded
            if self.expert_load_tracker.is_warm and self.lib is not None and attention_scores is not None and routing_decisions is not None:
                try:
                    # Execute expert conditioned KV eviction logic
                    self._apply_expert_eviction(attention_scores, routing_decisions)
                except Exception as e:
                    # Log error but prevent crash
                    print(f"Error during custom KV Cache eviction: {e}")
                    
            out_tensor = pb_utils.Tensor('OUTPUT_IDS', output_ids.astype(np.int32))
            responses.append(pb_utils.InferenceResponse([out_tensor]))
            
        return responses

    def _forward_trt_llm(self, request, input_ids):
        """
        Sends requests to the underlying C++ TRT-LLM engine.
        Returns output tokens, attention scores, and expert routing decisions.
        """
        # In a real environment, this invokes the C++ backend.
        # Here we simulate the return objects.
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        
        # Mocking outputs for interface demonstration
        output_ids = np.random.randint(0, 32000, size=(batch_size, 32))
        
        # Attention scores shape: [batch_size, num_heads, seq_len]
        num_heads = 32
        attention_scores = np.random.rand(batch_size, num_heads, seq_len).astype(np.float32)
        
        # Routing decisions shape: [batch_size, seq_len, top_k]
        routing_decisions = np.random.randint(0, self.num_experts, size=(batch_size, seq_len, self.top_k))
        
        return output_ids, attention_scores, routing_decisions

    def _apply_expert_eviction(self, attention_scores, routing_decisions):
        """Invoke custom C++ / CUDA eviction kernel on PyTorch/Cuda arrays."""
        batch_size, num_heads, seq_len = attention_scores.shape
        
        # Eviction priorities output
        eviction_priority = np.zeros((batch_size, seq_len), dtype=np.float32)
        
        # Convert routing decisions to binary indicators: [batch_size * seq_len, num_experts]
        expert_activations = np.zeros((batch_size * seq_len, self.num_experts), dtype=np.int32)
        for b in range(batch_size):
            for s in range(seq_len):
                idx = b * seq_len + s
                experts = routing_decisions[b, s]
                for exp_id in experts:
                    if 0 <= exp_id < self.num_experts:
                        expert_activations[idx, exp_id] = 1
                        
        expert_loads = self.expert_load_tracker.get_loads()

        # Execute using ctypes pointers if library loaded
        if self.lib is not None:
            # Pin arrays to device using PyTorch wrappers or ctypes directly
            # Assuming inputs are already CUDA-pinned in standard Triton
            # Here we demonstrate pointer extraction
            attn_ptr = attention_scores.ctypes.data_as(ctypes.c_void_p)
            act_ptr = expert_activations.ctypes.data_as(ctypes.c_void_p)
            loads_ptr = expert_loads.ctypes.data_as(ctypes.c_void_p)
            out_ptr = eviction_priority.ctypes.data_as(ctypes.c_void_p)
            
            # Launch kernel wrapper
            self.lib.launch_eviction_priority(
                attn_ptr,
                act_ptr,
                loads_ptr,
                out_ptr,
                batch_size,
                seq_len,
                num_heads,
                self.num_experts,
                self.expert_weight,
                None # Stream NULL
            )
            
            # Notify cache manager of the computed priorities
            # C++ TRT-LLM cache manager retrieves the values of out_ptr to schedule evictions

    def finalize(self):
        """Clean up references."""
        print("Finalizing expert-aware Triton Python backend.")
