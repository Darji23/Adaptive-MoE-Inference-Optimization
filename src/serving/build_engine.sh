#!/bin/bash
# build_engine.sh - Converts Hugging Face weights to TensorRT-LLM checkpoints and builds engines.
# Usage: ./build_engine.sh [fp8|bf16]

set -e

PRECISION=${1:-fp8}
WORKSPACE_DIR="/workspace"
MODEL_DIR="${WORKSPACE_DIR}/models/nemotron-nano-30b"
CKPT_DIR="${WORKSPACE_DIR}/checkpoints/nemotron_${PRECISION}_ckpt"
ENGINE_DIR="${WORKSPACE_DIR}/engines/nemotron_${PRECISION}"

echo "============================================="
echo "Building TensorRT-LLM Engine for Nemotron 30B"
echo "Precision Target: ${PRECISION}"
echo "============================================="

# Create necessary directories
mkdir -p "${CKPT_DIR}"
mkdir -p "${ENGINE_DIR}"

if [ "${PRECISION}" = "fp8" ]; then
    echo ">>> Phase 1: Converting HF weights to FP8 checkpoint format..."
    python3 "${WORKSPACE_DIR}/src/TensorRT-LLM/examples/nemotron/convert_checkpoint.py" \
        --model_dir "${MODEL_DIR}" \
        --output_dir "${CKPT_DIR}" \
        --dtype float16 \
        --use_fp8 \
        --tp_size 1

    echo ">>> Phase 2: Building TRT-LLM FP8 engine..."
    trtllm-build \
        --checkpoint_dir "${CKPT_DIR}" \
        --output_dir "${ENGINE_DIR}" \
        --gemm_plugin float16 \
        --paged_kv_cache enable \
        --max_batch_size 16 \
        --max_input_len 4096 \
        --max_seq_len 8192 \
        --use_fp8_context_fmha enable \
        --workers 4

elif [ "${PRECISION}" = "bf16" ]; then
    echo ">>> Phase 1: Converting HF weights to BF16 checkpoint format..."
    python3 "${WORKSPACE_DIR}/src/TensorRT-LLM/examples/nemotron/convert_checkpoint.py" \
        --model_dir "${MODEL_DIR}" \
        --output_dir "${CKPT_DIR}" \
        --dtype bfloat16 \
        --tp_size 1

    echo ">>> Phase 2: Building TRT-LLM BF16 engine..."
    trtllm-build \
        --checkpoint_dir "${CKPT_DIR}" \
        --output_dir "${ENGINE_DIR}" \
        --gemm_plugin bfloat16 \
        --paged_kv_cache enable \
        --max_batch_size 8 \
        --max_input_len 4096 \
        --max_seq_len 8192

else
    echo "Error: Unknown precision target '${PRECISION}'. Must be 'fp8' or 'bf16'."
    exit 1
fi

echo "============================================="
echo "Engine build completed successfully!"
echo "Output path: ${ENGINE_DIR}"
echo "============================================="
