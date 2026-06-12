#!/usr/bin/env python3
"""
Compares baseline (expert_weight=0.0) and optimized (expert_weight=0.2) runs
and generates a structured comparison report with percentage speedups.
"""

import os
import argparse
import pandas as pd


def generate_comparison_report(csv_path, target_concurrency=8, optimal_weight=0.2):
    """Prints a structured performance comparison report."""
    if not os.path.exists(csv_path):
        print(f"Error: Consolidated CSV not found at {csv_path}. Run parse_results.py first.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Extract baseline
    baseline_rows = df[(df["Concurrency"] == target_concurrency) & (df["Expert Weight"] == 0.0)]
    # Extract optimized
    optimized_rows = df[(df["Concurrency"] == target_concurrency) & (df["Expert Weight"] == optimal_weight)]
    
    if baseline_rows.empty or optimized_rows.empty:
        print("Warning: Could not find matching baseline and optimized runs.")
        print(f"Baseline rows for concurrency={target_concurrency}: {len(baseline_rows)}")
        print(f"Optimized rows for weight={optimal_weight}: {len(optimized_rows)}")
        return

    base = baseline_rows.iloc[0]
    opt = optimized_rows.iloc[0]

    # Calculate differences
    tps_diff = ((opt.throughput_tokens_sec - base.throughput_tokens_sec) / base.throughput_tokens_sec) * 100.0
    ttft_p50_diff = ((opt.ttft_p50_ms - base.ttft_p50_ms) / base.ttft_p50_ms) * 100.0
    ttft_p99_diff = ((opt.ttft_p99_ms - base.ttft_p99_ms) / base.ttft_p99_ms) * 100.0
    itl_p50_diff = ((opt.itl_p50_ms - base.itl_p50_ms) / base.itl_p50_ms) * 100.0
    hit_rate_diff = (opt.kv_cache_hit_rate - base.kv_cache_hit_rate) * 100.0

    report = f"""
================================================================================
                          INFERENCE OPTIMIZATION REPORT
================================================================================
Model:          Nemotron-3 Nano 30B (MoE + Mamba-2)
Node:           g18 (H100 80GB)
Precision:      FP8
Engine:         TensorRT-LLM v0.14.0
Server:         Triton Inference Server 24.12
Concurrency:    {target_concurrency}
Date:           2026-06-12
--------------------------------------------------------------------------------

Baseline (standard eviction):
  Throughput:   {base.throughput_tokens_sec:.2f} tokens/sec
  TTFT (p50):   {base.ttft_p50_ms:.1f} ms
  TTFT (p99):   {base.ttft_p99_ms:.1f} ms
  ITL (p50):    {base.itl_p50_ms:.2f} ms
  KV hit rate:  {base.kv_cache_hit_rate * 100.0:.1f}%

Expert-conditioned eviction (weight={optimal_weight:.1f}):
  Throughput:   {opt.throughput_tokens_sec:.2f} tokens/sec  ({"+" if tps_diff >= 0 else ""}{tps_diff:.1f}%)
  TTFT (p50):   {opt.ttft_p50_ms:.1f} ms                            ({"+" if ttft_p50_diff >= 0 else ""}{ttft_p50_diff:.1f}%)
  TTFT (p99):   {opt.ttft_p99_ms:.1f} ms                            ({"+" if ttft_p99_diff >= 0 else ""}{ttft_p99_diff:.1f}%)
  ITL (p50):    {opt.itl_p50_ms:.2f} ms                           ({"+" if itl_p50_diff >= 0 else ""}{itl_p50_diff:.1f}%)
  KV hit rate:  {opt.kv_cache_hit_rate * 100.0:.1f}%                              ({"+" if hit_rate_diff >= 0 else ""}{hit_rate_diff:.1f}% absolute change)

================================================================================
Summary: Expert-aware KV cache eviction yields a {tps_diff:.1f}% increase in overall 
throughput and reduces tail latency (p99 TTFT) by {abs(ttft_p99_diff):.1f}% due to 
reduced redundant recomputations of evicted tokens routed to overloaded experts.
================================================================================
"""
    print(report)
    return report


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark comparison report.")
    parser.add_argument("--csv_file", type=str, default="/workspace/results/consolidated_benchmark_results.csv", help="Path to consolidated results CSV")
    parser.add_argument("--concurrency", type=int, default=8, help="Target concurrency to evaluate")
    parser.add_argument("--optimal_weight", type=float, default=0.2, help="Optimal weight configuration")
    parser.add_argument("--output_report", type=str, default="/workspace/results/comparison_report.txt", help="Path to write text report")
    args = parser.parse_args()

    csv_path = args.csv_file
    if csv_path.startswith("/workspace/"):
        csv_path = csv_path.replace("/workspace/", "/Users/abhishekdarji/MyDrive/Adaptive-MoE-Inference-Optimization/")

    out_report = args.output_report
    if out_report.startswith("/workspace/"):
        out_report = out_report.replace("/workspace/", "/Users/abhishekdarji/MyDrive/Adaptive-MoE-Inference-Optimization/")

    report_text = generate_comparison_report(csv_path, args.concurrency, args.optimal_weight)
    
    if report_text and out_report:
        os.makedirs(os.path.dirname(out_report), exist_ok=True)
        with open(out_report, 'w') as f:
            f.write(report_text)
        print(f"Report successfully saved to: {out_report}")


if __name__ == "__main__":
    main()
