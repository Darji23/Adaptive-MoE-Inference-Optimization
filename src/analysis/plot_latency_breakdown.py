#!/usr/bin/env python3
"""
Generates latency breakdown plots comparing Time to First Token (TTFT) and
Inter-Token Latency (ITL) between baseline and optimized models.
"""

import os
import argparse
import pandas as pd
import numpy as np

# Headless matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def plot_latency_breakdowns(csv_path, output_png, target_concurrency=8):
    """Generates bar charts of TTFT and ITL comparison at target concurrency."""
    if not os.path.exists(csv_path):
        print(f"Error: Consolidated CSV not found at {csv_path}. Run parse_results.py first.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Filter for the target concurrency
    sub_df = df[df["Concurrency"] == target_concurrency].copy()
    if sub_df.empty:
        print(f"Warning: No results found for concurrency={target_concurrency}. Defaulting to first available.")
        if not df.empty:
            target_concurrency = df["Concurrency"].unique()[0]
            sub_df = df[df["Concurrency"] == target_concurrency].copy()
        else:
            return

    sub_df["Policy"] = sub_df["Expert Weight"].apply(lambda w: "Baseline" if w == 0.0 else f"Opt (W={w:.1f})")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.set_theme(style="whitegrid")
    
    # 1. Plot TTFT
    sns.barplot(
        data=sub_df, 
        x="Policy", 
        y="ttft_p50_ms", 
        ax=axes[0], 
        palette="Blues_d"
    )
    # Add error line/p99 representation if column exists
    if "ttft_p99_ms" in sub_df.columns:
        # Draw a second small bar or indicator for tail latency
        for i, row in enumerate(sub_df.itertuples()):
            axes[0].plot([i, i], [row.ttft_p50_ms, row.ttft_p99_ms], color='red', linestyle='--', marker='_')
        axes[0].set_ylabel("TTFT p50 / p99 (ms) [Red Line = p99]", fontsize=11)
    else:
        axes[0].set_ylabel("TTFT p50 (ms)", fontsize=11)
        
    axes[0].set_title(f"Time to First Token (TTFT) @ Concurrency={target_concurrency}", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Scheduling Policy", fontsize=11)

    # 2. Plot ITL
    sns.barplot(
        data=sub_df, 
        x="Policy", 
        y="itl_p50_ms", 
        ax=axes[1], 
        palette="Oranges_d"
    )
    axes[1].set_ylabel("Inter-Token Latency (ITL) p50 (ms)", fontsize=11)
    axes[1].set_title(f"Inter-Token Latency (ITL) @ Concurrency={target_concurrency}", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Scheduling Policy", fontsize=11)

    plt.suptitle(f"Nemotron-3 Nano 30B Latency Profiling", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Latency breakdown plot successfully saved to: {output_png}")


def main():
    parser = argparse.ArgumentParser(description="Plot latency breakdown bar charts.")
    parser.add_argument("--csv_file", type=str, default="/workspace/results/consolidated_benchmark_results.csv", help="Path to consolidated results CSV")
    parser.add_argument("--output_png", type=str, default="/workspace/results/latency_breakdown.png", help="Path to output image")
    parser.add_argument("--concurrency", type=int, default=8, help="Target concurrency to plot")
    args = parser.parse_args()

    csv_path = args.csv_file
    if csv_path.startswith("/workspace/"):
        csv_path = csv_path.replace("/workspace/", "/Users/abhishekdarji/MyDrive/Adaptive-MoE-Inference-Optimization/")

    out_png = args.output_png
    if out_png.startswith("/workspace/"):
        out_png = out_png.replace("/workspace/", "/Users/abhishekdarji/MyDrive/Adaptive-MoE-Inference-Optimization/")

    plot_latency_breakdowns(csv_path, out_png, args.concurrency)


if __name__ == "__main__":
    main()
