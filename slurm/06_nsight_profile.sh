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
module load nsight-systems

echo "Starting Nsight Systems GPU Profiling..."

# Run Nsight Systems profile targeting GPU kernel executions
singularity exec --nv \
  --bind /scratch/019113471/project:/workspace \
  /scratch/019113471/project/containers/tritonserver_trtllm.sif \
  nsys profile \
    --trace=cuda,nvtx \
    --output=/workspace/results/profiling/nemotron_trace \
    --force-overwrite true \
    python3 /workspace/src/profiling/profile_ssm_vs_attn.py

echo "Nsight Systems profiling completed. Report saved in results/profiling/."
