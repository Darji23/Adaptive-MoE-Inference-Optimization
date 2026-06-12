#!/bin/bash
# launch_triton.sh - Launches Triton Inference Server for served LLM engines.

set -e

WORKSPACE_DIR="/workspace"
MODEL_REPOSITORY="${WORKSPACE_DIR}/triton_models"
HTTP_PORT=${HTTP_PORT:-8000}
GRPC_PORT=${GRPC_PORT:-8001}
METRICS_PORT=${METRICS_PORT:-8002}

echo "============================================="
echo "Starting Triton Inference Server"
echo "Model Repository: ${MODEL_REPOSITORY}"
echo "HTTP Port:        ${HTTP_PORT}"
echo "gRPC Port:        ${GRPC_PORT}"
echo "Metrics Port:     ${METRICS_PORT}"
echo "============================================="

# Ensure directories exist
mkdir -p "${MODEL_REPOSITORY}"

# Start tritonserver in explicit control mode or polling mode
# Using exec to replace process or run in background
exec tritonserver \
    --model-repository="${MODEL_REPOSITORY}" \
    --http-port="${HTTP_PORT}" \
    --grpc-port="${GRPC_PORT}" \
    --metrics-port="${METRICS_PORT}" \
    --log-verbose=1 \
    --model-control-mode=explicit \
    --load-model=ensemble
