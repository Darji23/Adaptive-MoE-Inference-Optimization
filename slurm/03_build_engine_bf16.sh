#!/bin/bash
# slurm/03_build_engine_bf16.sh
#SBATCH --job-name=build_nemotron_bf16
#SBATCH --nodelist=g18
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/019113471/project/logs/%j_build_bf16.out

module load singularity

singularity exec --nv \
  --bind /scratch/019113471/project:/workspace \
  /scratch/019113471/project/containers/tritonserver_trtllm.sif \
  bash /workspace/src/serving/build_engine.sh bf16
