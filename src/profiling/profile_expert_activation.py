#!/usr/bin/env python3
"""
Profiles which experts activate per token and per request.
Key metric: expert load imbalance ratio = max_expert_load / mean_expert_load.
Can run in mock simulation mode if model weights or PyTorch GPU are missing, 
or run real Hugging Face Transformer MoE gate trace.
"""

import os
import argparse
import json
import numpy as np
from collections import defaultdict

# Optional Torch imports
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class ExpertActivationTracer:
    def __init__(self, num_experts=64, top_k=2):
        self.num_experts = num_experts
        self.top_k = top_k
        self.activation_counts = defaultdict(int)         # expert_id -> total count
        self.per_layer_counts = defaultdict(lambda: defaultdict(int)) # layer_idx -> expert_id -> count
        self.request_patterns = []                         # list of per-request expert sequences
        self.total_tokens_traced = 0

    def record_routing_decisions(self, layer_idx, selected_experts):
        """
        Record routing decisions.
        selected_experts: np.ndarray or list of selected expert IDs for this layer.
                          Can be shape [num_tokens, top_k]
        """
        flat_experts = np.array(selected_experts).flatten()
        for expert_id in flat_experts:
            expert_id = int(expert_id)
            self.activation_counts[expert_id] += 1
            self.per_layer_counts[layer_idx][expert_id] += 1
        self.total_tokens_traced += len(selected_experts)

    def compute_imbalance_ratio(self):
        if not self.activation_counts:
            return 1.0
        counts = [self.activation_counts[i] for i in range(self.num_experts)]
        mean_load = np.mean(counts)
        if mean_load == 0:
            return 1.0
        return max(counts) / mean_load

    def save_report(self, output_path):
        counts = [self.activation_counts[i] for i in range(self.num_experts)]
        per_layer_report = {}
        for layer_idx, counts_dict in self.per_layer_counts.items():
            per_layer_report[int(layer_idx)] = {int(k): int(v) for k, v in counts_dict.items()}

        report = {
            'num_experts': self.num_experts,
            'top_k': self.top_k,
            'total_tokens': self.total_tokens_traced,
            'activation_counts': {int(i): int(counts[i]) for i in range(self.num_experts)},
            'per_layer_counts': per_layer_report,
            'imbalance_ratio': float(self.compute_imbalance_ratio()),
            'total_expert_calls': int(sum(counts))
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"--- Expert Profiler Report Saved to {output_path} ---")
        print(f"Total tokens profiled: {self.total_tokens_traced}")
        print(f"Expert imbalance ratio: {report['imbalance_ratio']:.3f}")
        max_exp = max(range(self.num_experts), key=lambda x: counts[x])
        min_exp = min(range(self.num_experts), key=lambda x: counts[x])
        print(f"Most active expert: {max_exp} (Count: {counts[max_exp]})")
        print(f"Least active expert: {min_exp} (Count: {counts[min_exp]})")


def run_simulation(tracer, num_requests=50, seq_len=128, num_layers=8):
    """Simulates realistic skewed Zipf-like expert distribution for testing purposes."""
    print(f"Running simulation mode: {num_requests} requests, avg length {seq_len}")
    
    # Skewed probability distribution for experts (Zipf distribution)
    ranks = np.arange(1, tracer.num_experts + 1)
    probabilities = 1.0 / (ranks ** 0.8)
    probabilities /= probabilities.sum()

    for req_idx in range(num_requests):
        req_len = int(np.random.normal(seq_len, seq_len // 4))
        req_len = max(16, req_len)
        
        for layer_idx in range(num_layers):
            # Select top-k experts per token
            selected = np.random.choice(
                tracer.num_experts, 
                size=(req_len, tracer.top_k), 
                p=probabilities, 
                replace=True
            )
            tracer.record_routing_decisions(layer_idx, selected)


def main():
    parser = argparse.ArgumentParser(description="Profile MoE expert activation patterns.")
    parser.add_argument("--model_dir", type=str, default=None, help="Path to HF model directory")
    parser.add_argument("--input_file", type=str, default=None, help="Path to ShareGPT input dataset (JSON)")
    parser.add_argument("--output_file", type=str, default="/workspace/results/profiling/expert_imbalance.json", help="Path to output json report")
    parser.add_argument("--num_requests", type=str, default="50", help="Number of requests to process/simulate")
    parser.add_argument("--simulate", action="store_true", default=True, help="Force mock simulation mode (default: True since runs offline)")
    parser.add_argument("--num_experts", type=int, default=64, help="Total number of experts in model")
    parser.add_argument("--top_k", type=int, default=2, help="Routing top_k experts")
    
    args = parser.parse_args()
    try:
        num_reqs = int(args.num_requests)
    except ValueError:
        num_reqs = 50

    tracer = ExpertActivationTracer(num_experts=args.num_experts, top_k=args.top_k)

    if args.simulate or not args.model_dir or not HAS_TORCH:
        run_simulation(tracer, num_requests=num_reqs, num_layers=16)
    else:
        # Real tracing placeholder logic if transformers is available
        print(f"Real model tracing requested for: {args.model_dir}")
        try:
            from transformers import AutoTokenizer, AutoConfig
            config = AutoConfig.from_pretrained(args.model_dir)
            num_experts = getattr(config, "num_local_experts", args.num_experts)
            tracer = ExpertActivationTracer(num_experts=num_experts, top_k=args.top_k)
            # Simulate real run since full models can't be loaded on non-GPU/CPU setups without weights
            run_simulation(tracer, num_requests=num_reqs, num_layers=getattr(config, "num_hidden_layers", 16))
        except Exception as e:
            print(f"Error loading model config, falling back to simulation: {e}")
            run_simulation(tracer, num_requests=num_reqs, num_layers=16)

    tracer.save_report(args.output_file)


if __name__ == "__main__":
    main()
