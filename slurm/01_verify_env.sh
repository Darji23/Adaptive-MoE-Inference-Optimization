#!/bin/bash
#SBATCH --job-name=verify_env
#SBATCH --nodelist=g18
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/019113471/project/logs/%j_verify_env.out

echo "=== 1. Checking GPU Driver ==="
nvidia-smi

echo "=== 2. Checking Singularity ==="
singularity --version

echo "=== 3. Checking Available Modules ==="
module load singularity
module avail 2>&1 | grep -i "cuda\|nsight\|singularity"

echo "=== 4. Testing Container Launch ==="
singularity exec --nv \
  /scratch/019113471/project/containers/tritonserver_trtllm.sif \
  python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"

echo "=== 5. Installing Offline Pip Packages ==="
singularity exec --nv \
  /scratch/019113471/project/containers/tritonserver_trtllm.sif \
  pip install --no-index \
    --find-links=/scratch/019113471/project/pip_packages \
    tritonclient[all] nvidia-ml-py pynvml

echo "=== 6. Verifying Model Weights ==="
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
