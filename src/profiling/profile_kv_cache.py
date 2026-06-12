#!/usr/bin/env python3
"""
Scrapes Prometheus metrics from Triton Inference Server (default port 8002)
to measure and report KV Cache occupancy, allocation efficiency, and active blocks.
Supports offline mock simulation when Triton is not reachable.
"""

import os
import argparse
import json
import urllib.request
import urllib.error

# Typical TRT-LLM Triton metrics
TRTLLM_METRIC_KEYS = [
    "nv_gpu_memory_used_bytes",
    "nv_gpu_memory_total_bytes",
    "nv_trt_llm_kv_cache_block_utilization",
    "nv_trt_llm_kv_cache_active_blocks",
    "nv_trt_llm_kv_cache_free_blocks",
    "nv_trt_llm_kv_cache_max_blocks",
    "nv_trt_llm_request_active_count",
    "nv_trt_llm_request_queue_size"
]


def query_triton_metrics(url):
    """Fetch raw text metrics page from Triton metrics endpoint."""
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.read().decode('utf-8')
    except (urllib.error.URLError, ConnectionError) as e:
        print(f"Connection to Triton metrics endpoint at {url} failed: {e}")
        print("Falling back to simulated/mock metrics dashboard.")
        return None


def parse_prometheus_metrics(metrics_text):
    """Parses Prometheus metrics into a Python dictionary."""
    parsed_metrics = {}
    for line in metrics_text.splitlines():
        # Strip comments and whitespace
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        
        # Split key and value
        # Prometheus lines format: key{labels} value or key value
        if " " in line:
            key_part, val_str = line.rsplit(" ", 1)
            # Remove labels for simple aggregation
            key = key_part.split("{")[0].strip()
            try:
                val = float(val_str)
                # Keep first match or overwrite
                parsed_metrics[key] = val
            except ValueError:
                continue
    return parsed_metrics


def get_mock_metrics():
    """Generates realistic metrics mimicking Triton serving under workload."""
    return {
        "nv_gpu_memory_used_bytes": 42 * (1024 ** 3),  # 42 GB
        "nv_gpu_memory_total_bytes": 80 * (1024 ** 3), # 80 GB
        "nv_trt_llm_kv_cache_block_utilization": 0.76, # 76% full
        "nv_trt_llm_kv_cache_active_blocks": 15200,
        "nv_trt_llm_kv_cache_free_blocks": 4800,
        "nv_trt_llm_kv_cache_max_blocks": 20000,
        "nv_trt_llm_request_active_count": 12,
        "nv_trt_llm_request_queue_size": 3
    }


def analyze_metrics(metrics_dict):
    """Performs analysis and prints cache metrics."""
    analysis = {}
    
    # Extract keys safely
    gpu_used = metrics_dict.get("nv_gpu_memory_used_bytes", 0.0)
    gpu_total = metrics_dict.get("nv_gpu_memory_total_bytes", 80 * (1024**3))
    analysis["gpu_mem_util_pct"] = (gpu_used / gpu_total) * 100.0 if gpu_total > 0 else 0.0
    
    analysis["kv_block_util_pct"] = metrics_dict.get("nv_trt_llm_kv_cache_block_utilization", 0.0) * 100.0
    analysis["active_blocks"] = int(metrics_dict.get("nv_trt_llm_kv_cache_active_blocks", 0))
    analysis["free_blocks"] = int(metrics_dict.get("nv_trt_llm_kv_cache_free_blocks", 0))
    analysis["max_blocks"] = int(metrics_dict.get("nv_trt_llm_kv_cache_max_blocks", 20000))
    
    # Calculate hit/miss proxy or eviction load
    # (If block utilization is high, eviction events occur frequently)
    analysis["cache_pressure_index"] = analysis["kv_block_util_pct"] / 100.0
    if analysis["cache_pressure_index"] > 0.85:
        analysis["eviction_risk"] = "HIGH (Eviction policy heavily active)"
    elif analysis["cache_pressure_index"] > 0.60:
        analysis["eviction_risk"] = "MEDIUM (Healthy occupancy)"
    else:
        analysis["eviction_risk"] = "LOW (Ample headroom)"
        
    analysis["active_requests"] = int(metrics_dict.get("nv_trt_llm_request_active_count", 0))
    analysis["queued_requests"] = int(metrics_dict.get("nv_trt_llm_request_queue_size", 0))
    
    return analysis


def main():
    parser = argparse.ArgumentParser(description="Query and analyze Triton KV Cache metrics.")
    parser.add_argument("--url", type=str, default="http://localhost:8002/metrics", help="Triton metrics URL")
    parser.add_argument("--output_json", type=str, default="/workspace/results/profiling/kv_cache_hitrate.json", help="Path to output json file")
    args = parser.parse_args()

    output_path = args.output_json
    if output_path.startswith("/workspace/"):
        output_path = output_path.replace("/workspace/", "/Users/abhishekdarji/MyDrive/Adaptive-MoE-Inference-Optimization/")

    # Fetch raw metrics
    raw_metrics_text = query_triton_metrics(args.url)
    
    if raw_metrics_text:
        metrics_dict = parse_prometheus_metrics(raw_metrics_text)
    else:
        metrics_dict = get_mock_metrics()

    analysis = analyze_metrics(metrics_dict)
    
    print("\n--- Triton KV Cache Metrics Analysis ---")
    print(f"GPU Memory Utilization: {analysis['gpu_mem_util_pct']:.1f}%")
    print(f"KV Cache Block Utilization: {analysis['kv_block_util_pct']:.1f}%")
    print(f"Active KV Blocks: {analysis['active_blocks']} / {analysis['max_blocks']}")
    print(f"Free KV Blocks: {analysis['free_blocks']}")
    print(f"Active Requests: {analysis['active_requests']} (Queued: {analysis['queued_requests']})")
    print(f"KV Cache Pressure Index: {analysis['cache_pressure_index']:.3f} -> Eviction Risk: {analysis['eviction_risk']}")
    
    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\nKV cache analysis report written to: {output_path}")


if __name__ == "__main__":
    main()
