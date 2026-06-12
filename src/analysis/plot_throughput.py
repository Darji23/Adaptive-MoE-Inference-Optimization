#!/usr/bin/env python3
"""
Plots throughput (Tokens/second) vs Concurrency comparison curves
for baseline (expert_weight=0.0) vs custom eviction policies.
"""

import os
import argparse
import pandas as pd
import numpy as np

# Configure matplotlib for headless rendering
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def plot_throughput_curves(csv_path, output_png):
    """Loads consolidated results CSV and plots throughput comparisons."""
    if not os.path.exists(csv_path):
        print(f"Error: Consolidated CSV not found at {csv_path}. Run parse_results.py first.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Check if necessary columns exist
    required_cols = ["Concurrency", "throughput_tokens_sec", "Expert Weight"]
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Missing columns in results. Required: {required_cols}. Found: {list(df.columns)}")
        return

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Define groups: baseline (weight=0.0) and other weights
    # Map expert weight to labels
    df["Policy"] = df["Expert Weight"].apply(lambda w: "Baseline Eviction" if w == 0.0 else f"Expert-Aware (Weight={w:.1f})")
    
    # Plot lines with markers
    sns.lineplot(
        data=df, 
        x="Concurrency", 
        y="throughput_tokens_sec", 
        hue="Policy", 
        marker="o", 
        linewidth=2.5,
        markersize=8
    )
    
    plt.title("Nemotron-3 Nano 30B Throughput Comparison (H100 GPU)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Concurrency (Simultaneous Requests)", fontsize=12)
    plt.ylabel("System Throughput (Tokens / Second)", fontsize=12)
    plt.xticks(df["Concurrency"].unique())
    plt.legend(title="Eviction Configuration", fontsize=10, title_fontsize=11)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"Throughput comparison plot successfully saved to: {output_png}")


def main():
    parser = argparse.ArgumentParser(description="Plot throughput comparison curves.")
    parser.add_argument("--csv_file", type=str, default="/workspace/results/consolidated_benchmark_results.csv", help="Path to consolidated results CSV")
    parser.add_argument("--output_png", type=str, default="/workspace/results/throughput_comparison.png", help="Path to output image")
    args = parser.parse_args()

    csv_path = args.csv_file
    if csv_path.startswith("/workspace/"):
        csv_path = csv_path.replace("/workspace/", "/Users/abhishekdarji/MyDrive/Adaptive-MoE-Inference-Optimization/")

    out_png = args.output_png
    if out_png.startswith("/workspace/"):
        out_png = out_png.replace("/workspace/", "/Users/abhishekdarji/MyDrive/Adaptive-MoE-Inference-Optimization/")

    plot_throughput_curves(csv_path, out_png)


if __name__ == "__main__":
    main()
