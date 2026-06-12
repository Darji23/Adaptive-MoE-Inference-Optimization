#!/bin/bash
# slurm/05_baseline_benchmark.sh
#SBATCH --job-name=benchmark_baseline
#SBATCH --nodelist=g18
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/019113471/project/logs/%j_benchmark_baseline.out

module load singularity

# 1. Start Triton server in background
singularity exec --nv \
  --bind /scratch/019113471/project:/workspace \
  /scratch/019113471/project/containers/tritonserver_trtllm.sif \
  bash /workspace/src/serving/launch_triton.sh &

# Save process ID
TRITON_PID=$!

echo "Waiting for Triton Server to launch on port 8000..."
sleep 45  # Wait for Triton to initialize engine and load cache

# 2. Execute baseline benchmark sweeps (expert_weight=0.0)
for BATCH in 1 4 8; do
  for CONCURRENCY in 1 4 8 16; do
    echo "Running baseline sweep: batch=${BATCH}, concurrency=${CONCURRENCY}"
    singularity exec --nv \
      --bind /scratch/019113471/project:/workspace \
      /scratch/019113471/project/containers/tritonserver_trtllm.sif \
      python3 /workspace/src/analysis/run_benchmark.py \
        --output_dir /workspace/results/baseline \
        --concurrency ${CONCURRENCY} \
        --expert_weight 0.0
  done
done

# Kill Triton background process
kill $TRITON_PID
echo "Baseline benchmarking sweep complete!"
