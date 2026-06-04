# slurm/interactive.sh
srun \
  --nodelist=g18 \
  --gres=gpu:h100:1 \
  --mem=128G \
  --cpus-per-task=8 \
  --time=08:00:00 \
  --pty bash
