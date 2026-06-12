# HPC Execution Checklist

This guide lists the step-by-step terminal commands to execute the Adaptive MoE Inference Optimization project on the HPC terminal starting from environment verification (Phase 0).

---

## Phase 0: Environment Verification

### 1. SSH into the HPC login node
```bash
ssh 019113471@<hpc_hostname>
```

### 2. Request an interactive session on H100 (node g18)
```bash
srun --nodelist=g18 --gres=gpu:h100:1 --mem=128G --cpus-per-task=8 --time=08:00:00 --pty bash
```

### 3. Verify driver and Singularity installation
```bash
nvidia-smi
singularity --version
```

### 4. Load required modules
```bash
module load singularity
module load nsight-systems
```

### 5. Verify CUDA availability inside the Singularity container
```bash
singularity exec --nv \
  /scratch/019113471/project/containers/tritonserver_trtllm.sif \
  python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
```

### 6. Install offline pip packages inside the container
```bash
singularity exec --nv \
  /scratch/019113471/project/containers/tritonserver_trtllm.sif \
  pip install --no-index \
    --find-links=/scratch/019113471/project/pip_packages \
    tritonclient[all] nvidia-ml-py pynvml
```

### 7. Sanity check model weight configs
```bash
singularity exec --nv \
  --bind /scratch/019113471/project:/workspace \
  /scratch/019113471/project/containers/tritonserver_trtllm.sif \
  python3 -c "
from transformers import AutoConfig
try:
    cfg = AutoConfig.from_pretrained(
      '/workspace/models/nemotron-nano-30b',
      local_files_only=True
    )
    print('Model type:', cfg.model_type)
    print('Hidden size:', cfg.hidden_size)
    print('Num experts:', getattr(cfg, 'num_local_experts', 'N/A'))
except Exception as e:
    print('Error loading config:', e)
"
```

---

## Phase 1: Compile Custom CUDA Kernels

Run these commands inside the interactive session (node g18) to build the custom eviction and profiling CUDA library:

```bash
cd /scratch/019113471/project/src/kernels
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
```

Verify that `libmoe_kv_kernels.so` is created successfully inside `/scratch/019113471/project/src/kernels/build/`.

---

## Phase 2: Engine Compilation (TRT-LLM)

Submit the build scripts to the SLURM queue:

### 1. Compile primary FP8 engine
```bash
sbatch slurm/02_build_engine_fp8.sh
```

### 2. Compile baseline BF16 engine
```bash
sbatch slurm/03_build_engine_bf16.sh
```

Monitor compilation outputs using:
```bash
tail -f /scratch/019113471/project/logs/*build*.out
```

---

## Phase 3: Run Baseline Benchmarks

Submit the baseline benchmark job:
```bash
sbatch slurm/05_baseline_benchmark.sh
```

Verify baseline output results are populated inside `/scratch/019113471/project/results/baseline/`.

---

## Phase 4: Profiling & Bandwidth Analysis

Submit the Nsight profiling job:
```bash
sbatch slurm/06_nsight_profile.sh
```

Verify trace records are populated inside `/scratch/019113471/project/results/profiling/`.

---

## Phase 5: Sweep & Final Reports

Submit the hyperparameter sweep and report compilation script:
```bash
sbatch slurm/07_optimized_benchmark.sh
```

Once complete, verify that the final results are generated in the `/scratch/019113471/project/results/` directory:
- `consolidated_benchmark_results.csv`: Unified result dataset.
- `throughput_comparison.png`: Throughput curve chart.
- `latency_breakdown.png`: Latency breakdown chart.
- `comparison_report.txt`: Standard format text report.
