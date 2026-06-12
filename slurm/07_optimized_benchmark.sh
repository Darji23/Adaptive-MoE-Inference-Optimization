#!/bin/bash
# slurm/07_optimized_benchmark.sh
#SBATCH --job-name=benchmark_optimized
#SBATCH --nodelist=g18
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/019113471/project/logs/%j_benchmark_optimized.out

module load singularity

# 1. Start Triton server in background
singularity exec --nv \
  --bind /scratch/019113471/project:/workspace \
  /scratch/019113471/project/containers/tritonserver_trtllm.sif \
  bash /workspace/src/serving/launch_triton.sh &

TRITON_PID=$!

echo "Waiting for Triton Server to launch on port 8000..."
sleep 45

# 2. Sweep over expert_weight values and concurrency levels
for EXPERT_WEIGHT in 0.1 0.2 0.3 0.5; do
  for CONCURRENCY in 1 4 8 16; do
    echo "Running optimized sweep: expert_weight=${EXPERT_WEIGHT}, concurrency=${CONCURRENCY}"
    singularity exec --nv \
      --bind /scratch/019113471/project:/workspace \
      /scratch/019113471/project/containers/tritonserver_trtllm.sif \
      python3 /workspace/src/analysis/run_benchmark.py \
        --output_dir /workspace/results/optimized \
        --concurrency ${CONCURRENCY} \
        --expert_weight ${EXPERT_WEIGHT}
  done
done

# Kill Triton background process
kill $TRITON_PID

# 3. Post-run analysis aggregation
echo "Consolidating result files..."
singularity exec --nv \
  --bind /scratch/019113471/project:/workspace \
  /scratch/019113471/project/containers/tritonserver_trtllm.sif \
  python3 /workspace/src/analysis/parse_results.py

echo "Generating performance plots..."
singularity exec --nv \
  --bind /scratch/019113471/project:/workspace \
  /scratch/019113471/project/containers/tritonserver_trtllm.sif \
  python3 /workspace/src/analysis/plot_throughput.py

singularity exec --nv \
  --bind /scratch/019113471/project:/workspace \
  /scratch/019113471/project/containers/tritonserver_trtllm.sif \
  python3 /workspace/src/analysis/plot_latency_breakdown.py --concurrency 8

singularity exec --nv \
  --bind /scratch/019113471/project:/workspace \
  /scratch/019113471/project/containers/tritonserver_trtllm.sif \
  python3 /workspace/src/analysis/compare_baselines.py --concurrency 8 --optimal_weight 0.2

echo "Optimized benchmarking sweep and reporting complete!"
