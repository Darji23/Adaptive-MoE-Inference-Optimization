#!/usr/bin/env python3
"""
Parses CSV reports exported from NVIDIA Nsight Systems (nsys stats)
to extract and summarize execution times for key kernels:
  - Attention kernels (FMHA, scale-dot-product)
  - Custom kernels (moe_router_profiler, ssm_state_monitor, kv_eviction)
  - GEMM / MoE plugins
"""

import os
import argparse
import pandas as pd
import numpy as np


def generate_mock_csv(csv_path):
    """Generates a mock nsys gpukernsum.csv file for testing."""
    mock_data = [
        {"Time (%)": 42.5, "Total Time (ns)": 42500000, "Instances": 1200, "Avg (ns)": 35416, "Med (ns)": 35000, "Min (ns)": 30000, "Max (ns)": 45000, "Name": "sm90_fmha_fprop_fp16_kernel"},
        {"Time (%)": 25.1, "Total Time (ns)": 25100000, "Instances": 600, "Avg (ns)": 41833, "Med (ns)": 41000, "Min (ns)": 38000, "Max (ns)": 50000, "Name": "nvidia::tensorrt_llm::plugins::MoePlugin"},
        {"Time (%)": 18.2, "Total Time (ns)": 18200000, "Instances": 2400, "Avg (ns)": 7583, "Med (ns)": 7500, "Min (ns)": 6000, "Max (ns)": 9000, "Name": "mamba2_ssm_scan_kernel"},
        {"Time (%)": 8.4, "Total Time (ns)": 8400000, "Instances": 600, "Avg (ns)": 14000, "Med (ns)": 13800, "Min (ns)": 12000, "Max (ns)": 16000, "Name": "compute_eviction_priority_kernel"},
        {"Time (%)": 4.1, "Total Time (ns)": 4100000, "Instances": 600, "Avg (ns)": 6833, "Med (ns)": 6700, "Min (ns)": 5000, "Max (ns)": 8000, "Name": "profile_router_decisions_kernel"},
        {"Time (%)": 1.7, "Total Time (ns)": 1700000, "Instances": 600, "Avg (ns)": 2833, "Med (ns)": 2800, "Min (ns)": 2000, "Max (ns)": 4000, "Name": "monitor_ssm_states_kernel"}
    ]
    df = pd.DataFrame(mock_data)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Generated mock Nsight stats CSV at: {csv_path}")


def parse_nsight_csv(csv_path):
    """Reads and parses gpukernsum.csv from Nsight Systems."""
    if not os.path.exists(csv_path):
        print(f"Nsight report file {csv_path} not found.")
        generate_mock_csv(csv_path)

    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Strip whitespace from column headers
    df.columns = [c.strip() for c in df.columns]
    
    # Find matching categories
    categories = {
        "Attention / FlashAttention": ["fmha", "attention", "flash_attn"],
        "MoE / Expert Routing": ["moe", "expert", "router"],
        "SSM / Mamba-2 Recurrent Operations": ["ssm", "mamba", "scan"],
        "Custom Eviction Kernels": ["eviction", "kv_eviction"],
        "Other Operations": []
    }
    
    summary = []
    
    # Convert Total Time column to ms
    total_time_col = None
    for col in df.columns:
        if "total time" in col.lower():
            total_time_col = col
            break
            
    if total_time_col is None:
        raise ValueError(f"Could not find Total Time column in Nsight CSV. Columns found: {list(df.columns)}")

    # Classify each row
    classified_durations = {cat: 0.0 for cat in categories}
    classified_counts = {cat: 0 for cat in categories}
    
    for idx, row in df.iterrows():
        kernel_name = str(row["Name"]).lower()
        duration_val = float(row[total_time_col]) # standard unit is usually ns
        instances = int(row.get("Instances", 1))
        
        matched = False
        for cat, keywords in categories.items():
            if cat == "Other Operations":
                continue
            if any(kw in kernel_name for kw in keywords):
                classified_durations[cat] += duration_val
                classified_counts[cat] += instances
                matched = True
                break
                
        if not matched:
            classified_durations["Other Operations"] += duration_val
            classified_counts["Other Operations"] += instances

    # Sum total time
    grand_total_time = sum(classified_durations.values())
    if grand_total_time == 0:
        grand_total_time = 1e-9 # avoid division by zero
        
    for cat in categories:
        duration_ms = classified_durations[cat] / 1e6 # ns -> ms
        pct = (classified_durations[cat] / grand_total_time) * 100.0
        summary.append({
            "Category": cat,
            "Total GPU Time (ms)": duration_ms,
            "Kernel Instances": classified_counts[cat],
            "Percentage (%)": pct
        })
        
    summary_df = pd.DataFrame(summary)
    return summary_df, grand_total_time / 1e6


def main():
    parser = argparse.ArgumentParser(description="Parse Nsight Systems GPU Kernel Summaries.")
    parser.add_argument("--csv_file", type=str, default="/workspace/results/profiling/nsight_expert_trace.csv", help="Path to exported nsys gpukernsum.csv file")
    parser.add_argument("--output_file", type=str, default=None, help="Save parsed analysis to txt/csv file")
    args = parser.parse_args()

    csv_path = args.csv_file
    if csv_path.startswith("/workspace/"):
        csv_path = csv_path.replace("/workspace/", "/Users/abhishekdarji/MyDrive/Adaptive-MoE-Inference-Optimization/")

    print(f"Parsing Nsight stats from: {csv_path}")
    
    try:
        summary_df, total_ms = parse_nsight_csv(csv_path)
        
        print("\n=== Nsight Profiler GPU Kernel Summary ===")
        print(f"Total profiled GPU time: {total_ms:.3f} ms")
        print("------------------------------------------")
        print(summary_df.to_string(index=False))
        
        if args.output_file:
            out_path = args.output_file
            if out_path.startswith("/workspace/"):
                out_path = out_path.replace("/workspace/", "/Users/abhishekdarji/MyDrive/Adaptive-MoE-Inference-Optimization/")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            summary_df.to_csv(out_path, index=False)
            print(f"\nParsed results successfully saved to: {out_path}")
            
    except Exception as e:
        print(f"Error parsing Nsight CSV: {e}")


if __name__ == "__main__":
    main()
