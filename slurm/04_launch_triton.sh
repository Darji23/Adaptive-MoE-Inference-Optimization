#!/bin/bash
# slurm/04_launch_triton.sh
#SBATCH --job-name=launch_triton
#SBATCH --nodelist=g18
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/019113471/project/logs/%j_triton.out

module load singularity

singularity exec --nv \
  --bind /scratch/019113471/project:/workspace \
  /scratch/019113471/project/containers/tritonserver_trtllm.sif \
  bash /workspace/src/serving/launch_triton.sh
