# Adaptive MoE Inference Optimization

Optimizing and profiling mixture-of-experts (MoE) and Mamba-2 SSM hybrid inference (specifically **Nemotron-3 Nano 30B**) on NVIDIA H100 GPU (SM90 Hopper architecture) inside Singularity-contained TensorRT-LLM and Triton Inference Server environments.

---

## 1. Project Overview

This repository hosts custom CUDA kernels, profiling scripts, serving pipeline wrappers, and SLURM batch configurations designed to solve memory bandwidth challenges during hybrid Attention + MoE + Mamba recurrent model serving.

### Key Innovations:
1. **Expert-Conditioned KV Cache Eviction**: Standard eviction strategies (such as H2O or RocketKV) evict KV blocks using solely attention scores. This project integrates MoE gating signals. Tokens routed to overloaded, active experts are evicted sooner as their semantic density is higher and they generate redundant KV caches, improving hit rates on rare tokens.
2. **SSM vs. Attention Memory Profiling**: Characterization of HBM bandwidth traffic of Mamba-2 state equations vs Self-Attention sequences during inflight-batching serving conditions.

---

## 2. Directory Structure

```
/scratch/019113471/project/
├── README.md                          # This file
├── PROJECT_PLAN.md                    # Core project proposal & milestones
│
├── containers/
│   └── tritonserver_trtllm.sif        # Triton + TRT-LLM Singularity image (~20GB)
│
├── models/
│   └── nemotron-nano-30b/             # Hugging Face BF16 model checkpoints
│
├── src/
│   ├── kernels/                       # Custom CUDA kernels (SM90 target)
│   │   ├── CMakeLists.txt
│   │   ├── kv_eviction_policy.cu      # Expert-conditioned KV eviction
│   │   ├── moe_router_profiler.cu     # Gate profiling & imbalance tracer
│   │   └── ssm_state_monitor.cu       # Mamba-2 state dynamics monitor
│   │
│   ├── profiling/                     # Profilers and diagnostic tools
│   │   ├── nsight_parser.py           # Processes exported Nsys CSV files
│   │   ├── profile_expert_activation.py
│   │   ├── profile_kv_cache.py        # Pulls Triton cache metrics
│   │   └── profile_ssm_vs_attn.py     # Bandwidth and latency benchmarker
│   │
│   ├── serving/                       # Triton wrappers and server setup
│   │   ├── build_engine.sh            # Checkpoint converter and TRTLLM build tool
│   │   ├── launch_triton.sh           # Exec start tritonserver script
│   │   ├── gen_triton_config.py       # Config.pbtxt auto-generator
│   │   └── custom_backend/            # Expert-aware Triton Python backend
│   │       ├── config.pbtxt
│   │       └── model.py
│   │
│   └── analysis/                      # Analytics aggregation & plotting
│       ├── compare_baselines.py       # Speedup reporter
│       ├── parse_results.py           # Merges raw JSON results to CSV
│       ├── plot_latency_breakdown.py  # Plots TTFT/ITL breakdowns
│       ├── plot_throughput.py         # Plots throughput vs concurrency curves
│       └── run_benchmark.py           # Triton benchmark executor
│
├── slurm/                             # SLURM batch job scripts
│   ├── 01_verify_env.sh               # Environment check
│   ├── 02_build_engine_fp8.sh         # Compiles FP8 model engine
│   ├── 03_build_engine_bf16.sh        # Compiles BF16 baseline model engine
│   ├── 04_launch_triton.sh            # Triton execution service
│   ├── 05_baseline_benchmark.sh       # Benchmark baseline sweep
│   ├── 06_nsight_profile.sh           # Runs Nsys profile tasks
│   └── 07_optimized_benchmark.sh      # Parameter sweep & optimization report
│
└── notebooks/                         # Post-processing Jupyter Notebooks
    ├── 01_baseline_analysis.ipynb
    ├── 02_expert_activation_patterns.ipynb
    ├── 03_ssm_vs_attention_memory.ipynb
    └── 04_final_results.ipynb
```

---

## 3. Detailed Execution Workflow

Follow this sequence to execute compilation and benchmarking on the HPC cluster:

### Step 1: Verify the Environment
Submit the verification job to SLURM to ensure GPU access, Singularity loading, and python model config readers are fully functional:
```bash
sbatch slurm/01_verify_env.sh
```

### Step 2: Compile Custom Kernels
Compile custom CUDA eviction and tracking kernels. Execute locally or inside interactive nodes:
```bash
sbatch slurm/interactive.sh
# Once allocated:
mkdir -p src/kernels/build && cd src/kernels/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
```

### Step 3: Build the Engines
Build the FP8 and BF16 model engines. This converts HF checkpoints and compiles them using `trtllm-build`:
```bash
sbatch slurm/02_build_engine_fp8.sh
sbatch slurm/03_build_engine_bf16.sh
```

### Step 4: Run Baselines
Generate standard configurations, launch Triton in the background, and benchmark the baseline server:
```bash
sbatch slurm/05_baseline_benchmark.sh
```

### Step 5: Profile Core Layers & Nsight
Run Nsight Systems profiling and ssm vs attention memory tracking:
```bash
sbatch slurm/06_nsight_profile.sh
```

### Step 6: Sweep Hyperparameters & Generate Report
Execute the hyperparameter sweep over optimized settings (sweeps expert weights from 0.1 to 0.5), merge results, plot curves, and output comparison stats:
```bash
sbatch slurm/07_optimized_benchmark.sh
```

All parsed reports and image figures are output directly to `results/`. Results can also be inspected interactively in the templates provided inside `notebooks/`.
