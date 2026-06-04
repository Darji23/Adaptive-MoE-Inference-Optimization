# Adaptive MoE Inference Optimization — Complete Project Plan
**Target:** NVIDIA Internship (Deep Learning Applications & Frameworks)  
**Node:** g18 — 1× H100 80GB · 256GB RAM · No internet  
**Model:** Nemotron-3 Nano 30B (MoE + Mamba-2)  
**Stack:** TensorRT-LLM · Triton Inference Server · CUDA · Python · C++

---

## Table of Contents
1. [System Constraints & Strategy](#1-system-constraints--strategy)
2. [Pre-HPC Checklist (Your Laptop)](#2-pre-hpc-checklist-your-laptop)
3. [File & Directory Structure](#3-file--directory-structure)
4. [Phase-by-Phase Execution Plan](#4-phase-by-phase-execution-plan)
5. [SLURM Job Scripts](#5-slurm-job-scripts)
6. [Implementation Details](#6-implementation-details)
7. [Benchmarking Protocol](#7-benchmarking-protocol)
8. [Expected Results & Deliverables](#8-expected-results--deliverables)
9. [Troubleshooting Reference](#9-troubleshooting-reference)

---

## 1. System Constraints & Strategy

### g18 Node Specifications
| Parameter        | Value                        |
|------------------|------------------------------|
| GPU              | NVIDIA H100 80GB (SXM/PCIe)  |
| GPU VRAM         | 80 GB HBM3                   |
| System RAM       | 256 GB                       |
| Internet         | NOT AVAILABLE                |
| CUDA support     | FP8, FP16, BF16, INT8        |
| Nsight           | Available via module load    |

### Memory Budget for Nemotron-3 Nano 30B
| Precision | Model Weights | KV Cache (8k ctx) | Activations | Total     | Fits in 80GB? |
|-----------|--------------|-------------------|-------------|-----------|----------------|
| BF16      | ~60 GB       | ~8 GB             | ~4 GB       | ~72 GB    | YES (tight)    |
| FP8       | ~30 GB       | ~4 GB             | ~4 GB       | ~38 GB    | YES (comfortable) |
| INT8      | ~30 GB       | ~4 GB             | ~4 GB       | ~38 GB    | YES            |

**Decision:** Develop in FP8 as primary, BF16 as baseline comparison. FP8 leaves headroom for larger batch sizes and longer context windows — both critical for throughput experiments.

### No-Internet Strategy
All assets must be downloaded on a local machine with internet and transferred to HPC via `scp` or `rsync`. This includes:
- Model weights (Hugging Face)
- Docker/Singularity containers (NGC)
- Python packages (pip wheel files)
- Source code (GitHub clones)
- Datasets (for benchmarking)

---

## 2. Pre-HPC Checklist (Your Laptop)

Do all of this BEFORE connecting to HPC. Check each item off in order.

### 2.1 Accounts to Create
- [ ] NVIDIA NGC account: https://ngc.nvidia.com (free)
- [ ] Hugging Face account: https://huggingface.co (free)
- [ ] Accept Nemotron-3 Nano model license on Hugging Face
- [ ] Generate HF token (Settings → Access Tokens → New token → Read)
- [ ] Generate NGC API key (NGC → Setup → Get API Key)

### 2.2 Software to Install on Laptop
```bash
# Docker (to pull and convert containers)
# Install from https://docs.docker.com/get-docker/

# Singularity (to convert Docker → .sif for HPC)
# On Ubuntu:
sudo apt-get install -y singularity-container

# Hugging Face CLI
pip install huggingface_hub

# NGC CLI
pip install ngccli
ngc config set   # paste your API key when prompted
```

### 2.3 Download Model Weights
```bash
# Login to HuggingFace
huggingface-cli login   # paste your HF token

# Download Nemotron-3 Nano (primary model, ~60GB BF16)
huggingface-cli download \
  nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --local-dir ./downloads/nemotron-nano-30b \
  --local-dir-use-symlinks False

# Download Llama-3.1-8B (development proxy model, ~16GB)
# Use this on P100 nodes or early HPC dev — saves A100/H100 quota
huggingface-cli download \
  meta-llama/Llama-3.1-8B-Instruct \
  --local-dir ./downloads/llama-3.1-8b \
  --local-dir-use-symlinks False
```

### 2.4 Pull and Convert Containers
```bash
# Login to NGC
docker login nvcr.io
# Username: $oauthtoken
# Password: <your NGC API key>

# Pull TensorRT-LLM + Triton combined container
docker pull nvcr.io/nvidia/tritonserver:24.12-trtllm-python-py3

# Convert to Singularity .sif (takes ~15-20 min, output ~20GB)
singularity build ./downloads/tritonserver_trtllm.sif \
  docker-daemon://nvcr.io/nvidia/tritonserver:24.12-trtllm-python-py3

# Pull Nsight Systems (if not already on HPC)
# Usually available via module — confirm with: module avail | grep nsight
```

### 2.5 Download Python Packages (Offline pip)
```bash
mkdir -p ./downloads/pip_packages

pip download \
  torch==2.4.0 \
  transformers==4.45.0 \
  tritonclient[all]==2.51.0 \
  nvidia-ml-py==12.560.30 \
  numpy \
  pandas \
  matplotlib \
  seaborn \
  pynvml \
  psutil \
  tqdm \
  sentencepiece \
  -d ./downloads/pip_packages
```

### 2.6 Clone Source Repositories
```bash
mkdir -p ./downloads/src

# TensorRT-LLM (pin to a stable release)
git clone --depth=1 --branch v0.14.0 \
  https://github.com/NVIDIA/TensorRT-LLM.git \
  ./downloads/src/TensorRT-LLM

# Triton server backend for TensorRT-LLM
git clone --depth=1 \
  https://github.com/triton-inference-server/tensorrtllm_backend.git \
  ./downloads/src/tensorrtllm_backend

# genai-perf benchmarking tool
git clone --depth=1 \
  https://github.com/triton-inference-server/perf_analyzer.git \
  ./downloads/src/perf_analyzer
```

### 2.7 Download Benchmark Datasets
```bash
mkdir -p ./downloads/datasets

# ShareGPT (standard LLM inference benchmark)
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json \
  -O ./downloads/datasets/sharegpt.json

# LongBench (long-context evaluation — important for Mamba-2 SSM analysis)
git clone --depth=1 \
  https://github.com/THUDM/LongBench.git \
  ./downloads/datasets/LongBench
```

### 2.8 Transfer Everything to HPC
```bash
# Replace <username> and <hpc-hostname> with your actual details
HPC_USER=019113471
HPC_HOST=<your_hpc_hostname>
HPC_SCRATCH=/scratch/$HPC_USER

# Transfer all downloads (~100GB, use tmux or screen — this takes time)
rsync -avz --progress ./downloads/ \
  $HPC_USER@$HPC_HOST:$HPC_SCRATCH/project/
```

---

## 3. File & Directory Structure

```
/scratch/019113471/project/
│
├── README.md                          # Quick-start reference
├── PROJECT_PLAN.md                    # This file
│
├── containers/
│   └── tritonserver_trtllm.sif        # Singularity container (~20GB)
│
├── models/
│   ├── nemotron-nano-30b/             # HF weights (BF16, ~60GB)
│   │   ├── config.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   └── model-*.safetensors
│   └── llama-3.1-8b/                  # Dev proxy model (~16GB)
│       └── ...
│
├── engines/                           # TensorRT-LLM compiled engines
│   ├── nemotron_fp8/                  # FP8 engine (primary)
│   │   ├── rank0.engine
│   │   └── config.json
│   ├── nemotron_bf16/                 # BF16 baseline engine
│   │   └── ...
│   └── llama_fp16/                    # Dev proxy engine
│       └── ...
│
├── triton_models/                     # Triton model repository
│   ├── nemotron_trtllm/
│   │   ├── 1/                         # Model version directory
│   │   │   └── model.plan             # Symlink or copy of engine
│   │   └── config.pbtxt               # Triton model config
│   ├── preprocessing/                 # Tokenizer backend
│   │   ├── 1/
│   │   │   └── model.py
│   │   └── config.pbtxt
│   ├── postprocessing/                # Detokenizer backend
│   │   ├── 1/
│   │   │   └── model.py
│   │   └── config.pbtxt
│   └── ensemble/                      # Full pipeline ensemble
│       ├── 1/
│       └── config.pbtxt
│
├── src/                               # All source code (your work)
│   ├── TensorRT-LLM/                  # Cloned TRT-LLM repo
│   ├── tensorrtllm_backend/           # Cloned Triton backend
│   │
│   ├── kernels/                       # YOUR custom CUDA kernels
│   │   ├── moe_router_profiler.cu     # Expert activation tracer
│   │   ├── kv_eviction_policy.cu      # Expert-conditioned eviction
│   │   ├── ssm_state_monitor.cu       # Mamba-2 state tracking
│   │   └── CMakeLists.txt
│   │
│   ├── profiling/                     # YOUR profiling scripts
│   │   ├── profile_expert_activation.py
│   │   ├── profile_kv_cache.py
│   │   ├── profile_ssm_vs_attn.py     # Key: SSM vs attention layer analysis
│   │   └── nsight_parser.py           # Parse Nsight CSV outputs
│   │
│   ├── serving/                       # Triton configuration helpers
│   │   ├── build_engine.sh            # TRT-LLM engine build script
│   │   ├── launch_triton.sh           # Start Triton server
│   │   ├── gen_triton_config.py       # Auto-generate config.pbtxt
│   │   └── custom_backend/            # YOUR custom Triton Python backend
│   │       ├── model.py               # Expert-aware request scheduler
│   │       └── config.pbtxt
│   │
│   └── analysis/                      # Post-experiment analysis
│       ├── parse_results.py
│       ├── plot_throughput.py
│       ├── plot_latency_breakdown.py
│       └── compare_baselines.py
│
├── datasets/
│   ├── sharegpt.json                  # Standard benchmark
│   └── LongBench/                     # Long-context benchmark
│
├── pip_packages/                      # Offline pip wheels
│   └── *.whl
│
├── logs/                              # SLURM job logs
│   └── <jobid>.out
│
├── results/                           # All benchmark outputs
│   ├── baseline/
│   │   ├── bf16_results.json
│   │   └── fp8_results.json
│   ├── profiling/
│   │   ├── nsight_expert_trace.csv
│   │   ├── kv_cache_hitrate.json
│   │   └── ssm_bandwidth_analysis.csv
│   └── optimized/
│       ├── eviction_policy_v1.json
│       └── eviction_policy_v2.json
│
├── slurm/                             # All SLURM job scripts
│   ├── 01_verify_env.sh
│   ├── 02_build_engine_fp8.sh
│   ├── 03_build_engine_bf16.sh
│   ├── 04_launch_triton.sh
│   ├── 05_baseline_benchmark.sh
│   ├── 06_nsight_profile.sh
│   ├── 07_optimized_benchmark.sh
│   └── interactive.sh
│
└── notebooks/                         # Analysis notebooks (run locally)
    ├── 01_baseline_analysis.ipynb
    ├── 02_expert_activation_patterns.ipynb
    ├── 03_ssm_vs_attention_memory.ipynb
    └── 04_final_results.ipynb
```

---

## 4. Phase-by-Phase Execution Plan

### Phase 0 — Environment Verification (Day 1)
**Goal:** Confirm everything works before writing a single line of project code.

```bash
# SSH into g18
ssh 019113471@<hpc_hostname>
srun --nodelist=g18 --gres=gpu:h100:1 --mem=32G \
     --time=01:00:00 --pty bash

# 1. Check driver — MUST be >= 525 for TensorRT-LLM
nvidia-smi

# 2. Check Singularity
singularity --version

# 3. Check available modules
module avail 2>&1 | grep -i "cuda\|nsight\|singularity"

# 4. Test container launches correctly
singularity exec --nv \
  /scratch/019113471/project/containers/tritonserver_trtllm.sif \
  python3 -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"

# 5. Install offline pip packages inside container or venv
singularity exec --nv \
  /scratch/019113471/project/containers/tritonserver_trtllm.sif \
  pip install --no-index \
    --find-links=/scratch/019113471/project/pip_packages \
    tritonclient[all] nvidia-ml-py pynvml

# 6. Verify model weights are readable
python3 -c "
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(
  '/scratch/019113471/project/models/nemotron-nano-30b',
  local_files_only=True
)
print('Model type:', cfg.model_type)
print('Hidden size:', cfg.hidden_size)
print('Num experts:', getattr(cfg, 'num_local_experts', 'N/A'))
"
```

**Go/no-go criteria:**
- `nvidia-smi` shows H100 with driver ≥ 525 ✓
- `torch.cuda.is_available()` returns `True` inside container ✓
- Model config loads without error ✓

---

### Phase 1 — Baseline Engine Build (Days 2–4)
**Goal:** Build TensorRT-LLM engines for Nemotron in both FP8 and BF16. Establish throughput and latency baseline numbers.

#### 1a. Convert HuggingFace weights to TRT-LLM checkpoint format
```bash
singularity exec --nv \
  --bind /scratch/019113471/project:/workspace \
  /scratch/019113471/project/containers/tritonserver_trtllm.sif \
  python3 /workspace/src/TensorRT-LLM/examples/nemotron/convert_checkpoint.py \
    --model_dir /workspace/models/nemotron-nano-30b \
    --output_dir /workspace/checkpoints/nemotron_fp8_ckpt \
    --dtype float16 \
    --use_fp8 \
    --tp_size 1
```

#### 1b. Build FP8 engine (primary)
```bash
trtllm-build \
  --checkpoint_dir /workspace/checkpoints/nemotron_fp8_ckpt \
  --output_dir /workspace/engines/nemotron_fp8 \
  --gemm_plugin float16 \
  --paged_kv_cache enable \
  --max_batch_size 16 \
  --max_input_len 4096 \
  --max_seq_len 8192 \
  --use_fp8_context_fmha enable \
  --workers 4
```

#### 1c. Build BF16 engine (baseline comparison)
```bash
trtllm-build \
  --checkpoint_dir /workspace/checkpoints/nemotron_bf16_ckpt \
  --output_dir /workspace/engines/nemotron_bf16 \
  --gemm_plugin bfloat16 \
  --paged_kv_cache enable \
  --max_batch_size 8 \
  --max_input_len 4096 \
  --max_seq_len 8192
```

#### 1d. Sanity test — single inference
```bash
python3 /workspace/src/TensorRT-LLM/examples/run.py \
  --engine_dir /workspace/engines/nemotron_fp8 \
  --tokenizer_dir /workspace/models/nemotron-nano-30b \
  --max_output_len 128 \
  --input_text "Explain mixture of experts in neural networks:"
```

---

### Phase 2 — Triton Server Setup (Days 5–7)
**Goal:** Serve the engine through Triton. Verify HTTP and gRPC endpoints respond correctly.

#### 2a. Generate Triton model config
```bash
python3 /workspace/src/tensorrtllm_backend/tools/fill_template.py \
  --in_file /workspace/src/tensorrtllm_backend/all_models/inflight_batcher_llm/tensorrt_llm/config.pbtxt \
  --values "
    tokenizer_dir:/workspace/models/nemotron-nano-30b,
    engine_dir:/workspace/engines/nemotron_fp8,
    max_batch_size:16,
    decoupled_mode:true,
    batching_strategy:inflight_fused_batching,
    kv_cache_free_gpu_mem_fraction:0.85,
    enable_chunked_context:true
  " > /workspace/triton_models/nemotron_trtllm/config.pbtxt
```

#### 2b. Launch Triton server
```bash
tritonserver \
  --model-repository=/workspace/triton_models \
  --http-port=8000 \
  --grpc-port=8001 \
  --metrics-port=8002 \
  --log-verbose=1 \
  --model-control-mode=explicit \
  --load-model=ensemble &

# Wait for server to be ready
sleep 30
curl -s http://localhost:8000/v2/health/ready
```

#### 2c. Run first baseline benchmark
```bash
genai-perf profile \
  -m ensemble \
  --service-kind triton \
  --backend tensorrtllm \
  --num-prompts 100 \
  --concurrency 1 4 8 16 \
  --input-dataset sharegpt \
  --input-file /workspace/datasets/sharegpt.json \
  --tokenizer /workspace/models/nemotron-nano-30b \
  --profile-export-file /workspace/results/baseline/fp8_baseline.json \
  --url localhost:8001
```

**Record these numbers — everything you do later is measured against them:**
- Tokens per second (throughput)
- Time to first token (TTFT)
- Inter-token latency (ITL)
- GPU memory utilization

---

### Phase 3 — Profiling & Analysis (Days 8–14)
**Goal:** Your core research contribution. Profile Nemotron's Mamba-2 SSM layers vs. attention layers to find where memory bandwidth is being wasted. This is what makes the project novel.

#### 3a. Nsight Systems profile — full inference trace
```bash
# Request interactive session with longer time limit
srun --nodelist=g18 --gres=gpu:h100:1 \
     --mem=128G --time=04:00:00 --pty bash

# Run Nsight Systems trace
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=cpu \
  --output=/workspace/results/profiling/nemotron_fp8_trace \
  --force-overwrite true \
  python3 /workspace/src/profiling/profile_expert_activation.py \
    --engine_dir /workspace/engines/nemotron_fp8 \
    --input_file /workspace/datasets/sharegpt.json \
    --num_requests 50
```

#### 3b. Expert activation profiler (your custom script)
```python
# src/profiling/profile_expert_activation.py
"""
Profiles which experts activate per token and per request.
Key metric: expert load imbalance ratio = max_expert_load / mean_expert_load
A high ratio means some experts are bottlenecks.
"""
import torch
import json
import numpy as np
from collections import defaultdict

class ExpertActivationTracer:
    def __init__(self):
        self.activation_counts = defaultdict(int)    # expert_id -> count
        self.per_layer_counts = defaultdict(lambda: defaultdict(int))
        self.request_patterns = []                   # per-request expert sequences
    
    def hook_expert_gate(self, layer_idx, gate_output):
        """
        Hook into MoE gating network output.
        gate_output shape: [batch, seq_len, num_experts]
        """
        # Top-k selected experts
        topk_experts = torch.topk(gate_output, k=2, dim=-1).indices
        
        for expert_id in topk_experts.flatten().tolist():
            self.activation_counts[expert_id] += 1
            self.per_layer_counts[layer_idx][expert_id] += 1
    
    def compute_imbalance_ratio(self):
        counts = list(self.activation_counts.values())
        return max(counts) / np.mean(counts)
    
    def save_report(self, output_path):
        report = {
            'activation_counts': dict(self.activation_counts),
            'per_layer_counts': {k: dict(v) for k, v in self.per_layer_counts.items()},
            'imbalance_ratio': self.compute_imbalance_ratio(),
            'total_expert_calls': sum(self.activation_counts.values())
        }
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Expert imbalance ratio: {report['imbalance_ratio']:.3f}")
        print(f"Most activated expert: {max(self.activation_counts, key=self.activation_counts.get)}")
```

#### 3c. SSM vs. Attention memory bandwidth analysis (your novel measurement)
```python
# src/profiling/profile_ssm_vs_attn.py
"""
Measures memory bandwidth consumption for:
  - Transformer attention layers (with KV cache)
  - Mamba-2 SSM layers (recurrent state, no KV cache)

This is novel because Nemotron-3 Nano mixes both layer types
and no public study has measured their relative bandwidth
burden under production serving conditions.
"""
import pynvml
import time
import torch

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def measure_memory_bandwidth(layer_fn, input_tensor, num_warmup=5, num_runs=20):
    """Measure effective HBM bandwidth for a single layer forward pass."""
    # Warmup
    for _ in range(num_warmup):
        _ = layer_fn(input_tensor)
    torch.cuda.synchronize()
    
    results = []
    for _ in range(num_runs):
        mem_before = pynvml.nvmlDeviceGetMemoryInfo(handle).used
        t0 = time.perf_counter()
        out = layer_fn(input_tensor)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        mem_after = pynvml.nvmlDeviceGetMemoryInfo(handle).used
        
        elapsed_ms = (t1 - t0) * 1000
        bytes_moved = abs(mem_after - mem_before)
        bandwidth_GBs = (bytes_moved / 1e9) / (elapsed_ms / 1000)
        results.append({
            'latency_ms': elapsed_ms,
            'bandwidth_GBs': bandwidth_GBs
        })
    
    return results

# Key finding you are looking for:
# SSM layers have NO KV cache → lower memory traffic per token
# But recurrent state update has sequential data dependency
# → SSM layers are compute-bound, attention layers are memory-bound
# → Different optimization strategies apply to each
```

#### 3d. KV cache hit rate analysis
```python
# src/profiling/profile_kv_cache.py
"""
Measures KV cache eviction rates under different request patterns.
Tests whether expert activation patterns correlate with KV cache misses.
"""

def analyze_kv_cache_behavior(triton_metrics_url="http://localhost:8002/metrics"):
    """Pull Triton metrics and analyze KV cache efficiency."""
    import urllib.request
    
    response = urllib.request.urlopen(triton_metrics_url)
    metrics_text = response.read().decode('utf-8')
    
    # Parse relevant metrics
    kv_metrics = {}
    for line in metrics_text.split('\n'):
        if 'kv_cache' in line.lower() and not line.startswith('#'):
            parts = line.rsplit(' ', 1)
            if len(parts) == 2:
                kv_metrics[parts[0]] = float(parts[1])
    
    return kv_metrics
```

---

### Phase 4 — Custom Kernel Development (Days 15–25)
**Goal:** Write and integrate your novel contribution into TensorRT-LLM.

#### Research question you are answering:
> In Nemotron-3 Nano's MoE layers, does the KV cache usage pattern differ systematically between high-activation experts and low-activation experts? If yes, can eviction policy be conditioned on expert activation to improve cache efficiency?

#### 4a. Expert-conditioned KV eviction kernel
```cuda
// src/kernels/kv_eviction_policy.cu
/**
 * Expert-Conditioned KV Cache Eviction Policy
 *
 * Standard KV eviction (RocketKV, H2O) uses attention score recency.
 * This kernel adds expert activation frequency as a secondary signal.
 *
 * Hypothesis: tokens routed to overloaded experts (high activation count)
 * are less likely to have their KV entries reused, because:
 *   1. Overloaded experts process semantically similar tokens
 *   2. Semantically similar tokens generate redundant KV entries
 *   3. These entries can be evicted earlier without quality loss
 */

#include <cuda_fp8.h>
#include <cuda_runtime.h>

struct KVEvictionScore {
    float attention_score;      // Standard recency signal
    float expert_load_signal;   // Your novel addition: normalized expert activation count
    int   token_position;
    int   layer_idx;
};

__global__ void compute_eviction_priority(
    const float* __restrict__ attention_scores,   // [batch, heads, seq_len]
    const int*   __restrict__ expert_activations, // [seq_len, num_experts]
    const float* __restrict__ expert_load_counts, // [num_experts] — running avg
    float*       __restrict__ eviction_priority,  // [batch, seq_len] — output
    int batch_size,
    int seq_len,
    int num_heads,
    int num_experts,
    float expert_weight   // Hyperparameter: how much expert signal contributes
) {
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;
    
    if (token_idx >= seq_len || batch_idx >= batch_size) return;
    
    // 1. Compute attention-based score (mean across heads)
    float attn_score = 0.0f;
    for (int h = 0; h < num_heads; h++) {
        attn_score += attention_scores[batch_idx * num_heads * seq_len 
                                       + h * seq_len + token_idx];
    }
    attn_score /= num_heads;
    
    // 2. Compute expert load signal for this token
    float expert_signal = 0.0f;
    for (int e = 0; e < num_experts; e++) {
        if (expert_activations[token_idx * num_experts + e] > 0) {
            // Token was routed through expert e — add its load
            expert_signal += expert_load_counts[e];
        }
    }
    // Normalize by number of activated experts (top-k, typically 2)
    expert_signal /= 2.0f;
    
    // 3. Combined eviction priority score
    // Lower score = higher eviction priority (evict sooner)
    eviction_priority[batch_idx * seq_len + token_idx] = 
        (1.0f - expert_weight) * attn_score 
        - expert_weight * expert_signal;
    
    // Note: expert_signal being high (overloaded expert) REDUCES score
    // → those tokens get evicted sooner — this is the hypothesis to test
}
```

#### 4b. Build the kernel
```cmake
# src/kernels/CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(moe_kv_kernels CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES 90)  # H100 = SM90
find_package(CUDA REQUIRED)

add_library(moe_kv_kernels SHARED
    kv_eviction_policy.cu
    moe_router_profiler.cu
    ssm_state_monitor.cu
)

target_compile_options(moe_kv_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        -use_fast_math
        --ptxas-options=-v
        -gencode arch=compute_90,code=sm_90
    >
)
```

```bash
# Build
mkdir -p /workspace/src/kernels/build
cd /workspace/src/kernels/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
```

---

### Phase 5 — Custom Triton Backend (Days 26–30)
**Goal:** Expose expert activation metadata through Triton's Python backend so the eviction kernel receives routing information at inference time.

```python
# src/serving/custom_backend/model.py
"""
Expert-aware Triton Python backend.
Intercepts inference requests, extracts MoE routing decisions,
and feeds them to the KV eviction kernel.
"""
import triton_python_backend_utils as pb_utils
import numpy as np
import json
import ctypes

# Load your custom kernel
_lib = ctypes.CDLL('/workspace/src/kernels/build/libmoe_kv_kernels.so')

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])
        self.expert_load_tracker = ExpertLoadTracker(num_experts=64)
        
    def execute(self, requests):
        responses = []
        for request in requests:
            # Get input tokens
            input_ids = pb_utils.get_input_tensor_by_name(
                request, 'input_ids').as_numpy()
            
            # Standard TensorRT-LLM forward pass
            output_ids, routing_decisions = self._forward_with_routing(input_ids)
            
            # Update expert load tracker
            self.expert_load_tracker.update(routing_decisions)
            
            # Compute eviction priorities with expert signal
            if self.expert_load_tracker.is_warm:
                eviction_priorities = self._compute_eviction_priorities(
                    routing_decisions,
                    self.expert_load_tracker.get_loads()
                )
                # Pass to KV cache manager
                self._apply_eviction_policy(eviction_priorities)
            
            out_tensor = pb_utils.Tensor('output_ids', output_ids)
            responses.append(pb_utils.InferenceResponse([out_tensor]))
        
        return responses
    
    def _forward_with_routing(self, input_ids):
        # Interface to TensorRT-LLM engine with routing hook
        # Returns both output tokens and expert routing decisions
        raise NotImplementedError("Wire to your TRT-LLM engine here")
    
    def finalize(self):
        pass


class ExpertLoadTracker:
    """Exponential moving average of expert activation frequency."""
    def __init__(self, num_experts, alpha=0.95, warmup_steps=100):
        self.num_experts = num_experts
        self.alpha = alpha
        self.warmup_steps = warmup_steps
        self.load_ema = np.ones(num_experts)
        self.step = 0
    
    def update(self, routing_decisions):
        counts = np.bincount(routing_decisions.flatten(), 
                             minlength=self.num_experts).astype(float)
        counts /= counts.sum() + 1e-8
        self.load_ema = self.alpha * self.load_ema + (1 - self.alpha) * counts
        self.step += 1
    
    def get_loads(self):
        return self.load_ema
    
    @property
    def is_warm(self):
        return self.step >= self.warmup_steps
```

---

### Phase 6 — Final Benchmarking (Days 31–35)
**Goal:** Measure the actual impact of your eviction policy. Compare against baseline systematically.

#### Benchmark matrix to run
| Configuration          | Batch size | Context length | Concurrency | Metric        |
|------------------------|-----------|----------------|-------------|---------------|
| BF16 baseline          | 1, 4, 8   | 1k, 4k, 8k    | 1, 4, 8, 16 | TPS, TTFT, ITL|
| FP8 baseline           | 1, 4, 8   | 1k, 4k, 8k    | 1, 4, 8, 16 | TPS, TTFT, ITL|
| FP8 + custom eviction  | 1, 4, 8   | 1k, 4k, 8k    | 1, 4, 8, 16 | TPS, TTFT, ITL|
| FP8 + expert weight 0.1| 1, 4, 8   | 1k, 4k, 8k    | 1, 4, 8, 16 | TPS, TTFT, ITL|
| FP8 + expert weight 0.3| 1, 4, 8   | 1k, 4k, 8k    | 1, 4, 8, 16 | TPS, TTFT, ITL|

```bash
# Run full benchmark sweep
for EXPERT_WEIGHT in 0.0 0.1 0.2 0.3 0.5; do
  for BATCH in 1 4 8; do
    for CTX in 1024 4096 8192; do
      genai-perf profile \
        -m ensemble \
        --service-kind triton \
        --backend tensorrtllm \
        --concurrency 1 4 8 16 \
        --input-file /workspace/datasets/sharegpt.json \
        --tokenizer /workspace/models/nemotron-nano-30b \
        --profile-export-file \
          /workspace/results/optimized/ew${EXPERT_WEIGHT}_b${BATCH}_c${CTX}.json \
        --url localhost:8001
    done
  done
done
```

---

## 5. SLURM Job Scripts

### Interactive session on g18
```bash
# slurm/interactive.sh
srun \
  --nodelist=g18 \
  --gres=gpu:h100:1 \
  --mem=128G \
  --cpus-per-task=8 \
  --time=08:00:00 \
  --pty bash
```

### Engine build job
```bash
#!/bin/bash
# slurm/02_build_engine_fp8.sh
#SBATCH --job-name=build_nemotron_fp8
#SBATCH --nodelist=g18
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/019113471/project/logs/%j_build_fp8.out

module load singularity

singularity exec --nv \
  --bind /scratch/019113471/project:/workspace \
  /scratch/019113471/project/containers/tritonserver_trtllm.sif \
  bash /workspace/src/serving/build_engine.sh fp8
```

### Benchmark job
```bash
#!/bin/bash
# slurm/05_baseline_benchmark.sh
#SBATCH --job-name=benchmark_baseline
#SBATCH --nodelist=g18
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/019113471/project/logs/%j_benchmark.out

module load singularity

singularity exec --nv \
  --bind /scratch/019113471/project:/workspace \
  /scratch/019113471/project/containers/tritonserver_trtllm.sif \
  bash /workspace/src/serving/launch_triton.sh &

sleep 45  # Wait for Triton to be ready

singularity exec --nv \
  --bind /scratch/019113471/project:/workspace \
  /scratch/019113471/project/containers/tritonserver_trtllm.sif \
  python3 /workspace/src/analysis/run_benchmark.py \
    --output_dir /workspace/results/baseline
```

### Nsight profiling job
```bash
#!/bin/bash
# slurm/06_nsight_profile.sh
#SBATCH --job-name=nsight_profile
#SBATCH --nodelist=g18
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/019113471/project/logs/%j_nsight.out

module load singularity
module load nsight-systems  # Or: module load cuda (nsys ships with CUDA)

singularity exec --nv \
  --bind /scratch/019113471/project:/workspace \
  /scratch/019113471/project/containers/tritonserver_trtllm.sif \
  nsys profile \
    --trace=cuda,nvtx \
    --output=/workspace/results/profiling/nemotron_trace \
    --force-overwrite true \
    python3 /workspace/src/profiling/profile_ssm_vs_attn.py
```

---

## 6. Implementation Details

### Key files to modify in TensorRT-LLM

| File | What you change | Why |
|------|----------------|-----|
| `cpp/tensorrt_llm/kernels/kvCacheUtils.h` | Add expert_load field to KV block metadata | Carry routing signal through cache |
| `cpp/tensorrt_llm/runtime/kvCacheManager.cpp` | Call your eviction kernel during block eviction | Core integration point |
| `cpp/tensorrt_llm/plugins/moePlugin/moePlugin.cpp` | Expose routing decisions as an output tensor | Feed routing to eviction policy |
| `tensorrt_llm/models/nemotron.py` | Add NVTX markers around SSM vs. attention layers | Enable Nsight layer-by-layer profiling |

### Key metrics to track throughout
```python
METRICS = {
    # Throughput
    'tokens_per_second': None,          # Higher is better
    'requests_per_second': None,        # Higher is better
    
    # Latency
    'time_to_first_token_ms': None,     # Lower is better
    'inter_token_latency_ms': None,     # Lower is better
    'p99_latency_ms': None,             # Lower is better
    
    # Memory
    'kv_cache_hit_rate': None,          # Higher is better
    'gpu_memory_utilization': None,     # Track headroom
    'kv_cache_utilization': None,       # How full the cache is
    
    # Your novel metrics
    'expert_imbalance_ratio': None,     # max/mean expert load
    'eviction_policy_savings': None,    # Tokens not re-computed due to better eviction
    'ssm_vs_attn_bandwidth_ratio': None # Your characterization finding
}
```

---

## 7. Benchmarking Protocol

### Standard reporting format (use for all results)
```
Model:          Nemotron-3 Nano 30B
Node:           g18 (H100 80GB)
Precision:      FP8
Engine:         TensorRT-LLM v0.14.0
Server:         Triton Inference Server 24.12
Dataset:        ShareGPT (N=500 prompts)
Date:           YYYY-MM-DD

Baseline (standard eviction):
  Throughput:   XXX tokens/sec @ concurrency=8
  TTFT (p50):   XX ms
  TTFT (p99):   XX ms
  ITL (p50):    X.X ms
  KV hit rate:  XX%

Expert-conditioned eviction (weight=0.2):
  Throughput:   XXX tokens/sec @ concurrency=8  (+X%)
  TTFT (p50):   XX ms                            (-X%)
  TTFT (p99):   XX ms                            (-X%)
  ITL (p50):    X.X ms                           (-X%)
  KV hit rate:  XX%                              (+X%)
```

---

## 8. Expected Results & Deliverables

### Minimum viable result (enough for NVIDIA)
Even if the eviction policy shows no improvement, the **profiling characterization of Mamba-2 SSM vs. attention layer memory behavior in Nemotron-3 Nano** is a legitimate original contribution — because no one has published this measurement for this model.

### Stretch results
- 10–20% throughput improvement at high concurrency from better KV eviction
- Clear expert imbalance ratio measurement showing routing non-uniformity
- Expert weight hyperparameter sweep showing optimal value

### Deliverables checklist
- [ ] Technical report (6–8 pages, NVIDIA intern report format)
- [ ] GitHub repository with all code, SLURM scripts, and results
- [ ] Pull request to `NVIDIA/TensorRT-LLM` with Nemotron profiling NVTX markers
- [ ] Benchmark results JSON files (reproducible by anyone with HPC access)
- [ ] Nsight Systems trace files (.nsys-rep) for key experiments

---

## 9. Troubleshooting Reference

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| `CUDA out of memory` during engine build | BF16 weights too large | Use FP8 build, or reduce `--max_batch_size` to 4 |
| Triton server not responding after 60s | Engine path mismatch in config.pbtxt | Check absolute paths match your bind mount |
| `singularity: command not found` | Module not loaded | `module load singularity` first |
| `nvidia-smi` shows wrong driver | Using login node, not g18 | `srun --nodelist=g18 ... --pty bash` first |
| `trtllm-build` not found | Not inside container | Ensure you're inside `singularity exec --nv ...` |
| genai-perf shows 0 throughput | Triton not ready | Add `sleep 60` after launching server |
| KV cache OOM during benchmark | Cache fraction too high | Reduce `kv_cache_free_gpu_mem_fraction` to 0.7 |
| Nsight trace file is empty | Insufficient permissions | Add `--privileged` to singularity exec |

---

*Last updated: June 2026*  
*Node: g18 · GPU: H100 80GB · Project: Adaptive MoE Inference Optimization*
