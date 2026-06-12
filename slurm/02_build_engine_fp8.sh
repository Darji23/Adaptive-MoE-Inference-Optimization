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
