#!/usr/bin/env python3
"""
Generates Triton Inference Server config.pbtxt files for the LLM serving pipeline.
Generates configs for:
  1. preprocessing (tokenizer)
  2. nemotron_trtllm (TensorRT-LLM engine runner)
  3. postprocessing (detokenizer)
  4. ensemble (joins them together)
"""

import os
import argparse


def get_preprocessing_config(max_batch_size):
    return f"""
name: "preprocessing"
backend: "python"
max_batch_size: {max_batch_size}
input [
  {{
    name: "QUERY"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }},
  {{
    name: "REQUEST_OUTPUT_LEN"
    data_type: TYPE_INT32
    dims: [ 1 ]
  }}
]
output [
  {{
    name: "INPUT_IDS"
    data_type: TYPE_INT32
    dims: [ -1 ]
  }},
  {{
    name: "REQUEST_INPUT_LEN"
    data_type: TYPE_INT32
    dims: [ 1 ]
  }},
  {{
    name: "REQUEST_OUTPUT_LEN"
    data_type: TYPE_INT32
    dims: [ 1 ]
  }}
]
instance_group [
  {{
    count: 1
    kind: KIND_CPU
  }}
]
"""


def get_trtllm_config(engine_dir, max_batch_size, kv_cache_fraction):
    return f"""
name: "nemotron_trtllm"
backend: "tensorrtllm"
max_batch_size: {max_batch_size}

# Decoupled mode for streaming outputs
model_transaction_policy {{
  decoupled: true
}}

input [
  {{
    name: "input_ids"
    data_type: TYPE_INT32
    dims: [ -1 ]
  }},
  {{
    name: "input_lengths"
    data_type: TYPE_INT32
    dims: [ 1 ]
  }},
  {{
    name: "request_output_len"
    data_type: TYPE_INT32
    dims: [ 1 ]
  }}
]

output [
  {{
    name: "output_ids"
    data_type: TYPE_INT32
    dims: [ -1 ]
  }}
]

parameters: {{
  key: "gpt_model_type"
  value: {{
    string_value: "inflight_fused_batching"
  }}
}}
parameters: {{
  key: "gpt_model_path"
  value: {{
    string_value: "{engine_dir}"
  }}
}}
parameters: {{
  key: "kv_cache_free_gpu_mem_fraction"
  value: {{
    string_value: "{kv_cache_fraction}"
  }}
}}
parameters: {{
  key: "batching_strategy"
  value: {{
    string_value: "inflight_fused_batching"
  }}
}}
instance_group [
  {{
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }}
]
"""


def get_postprocessing_config(max_batch_size):
    return f"""
name: "postprocessing"
backend: "python"
max_batch_size: {max_batch_size}
input [
  {{
    name: "output_ids"
    data_type: TYPE_INT32
    dims: [ -1 ]
  }}
]
output [
  {{
    name: "OUTPUT_TEXT"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }}
]
instance_group [
  {{
    count: 1
    kind: KIND_CPU
  }}
]
"""


def get_ensemble_config(max_batch_size):
    return f"""
name: "ensemble"
platform: "ensemble"
max_batch_size: {max_batch_size}

# Decoupled mode for streaming outputs
model_transaction_policy {{
  decoupled: true
}}

input [
  {{
    name: "text_input"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }},
  {{
    name: "max_tokens"
    data_type: TYPE_INT32
    dims: [ 1 ]
  }}
]

output [
  {{
    name: "text_output"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }}
]

ensemble_scheduling {{
  step [
    {{
      model_name: "preprocessing"
      model_version: -1
      input_map {{
        key: "QUERY"
        value: "text_input"
      }}
      input_map {{
        key: "REQUEST_OUTPUT_LEN"
        value: "max_tokens"
      }}
      output_map {{
        key: "INPUT_IDS"
        value: "preprocessed_input_ids"
      }}
      output_map {{
        key: "REQUEST_INPUT_LEN"
        value: "preprocessed_input_lengths"
      }}
      output_map {{
        key: "REQUEST_OUTPUT_LEN"
        value: "preprocessed_output_len"
      }}
    }},
    {{
      model_name: "nemotron_trtllm"
      model_version: -1
      input_map {{
        key: "input_ids"
        value: "preprocessed_input_ids"
      }}
      input_map {{
        key: "input_lengths"
        value: "preprocessed_input_lengths"
      }}
      input_map {{
        key: "request_output_len"
        value: "preprocessed_output_len"
      }}
      output_map {{
        key: "output_ids"
        value: "postprocessed_output_ids"
      }}
    }},
    {{
      model_name: "postprocessing"
      model_version: -1
      input_map {{
        key: "output_ids"
        value: "postprocessed_output_ids"
      }}
      output_map {{
        key: "OUTPUT_TEXT"
        value: "text_output"
      }}
    }}
  ]
}}
"""


def main():
    parser = argparse.ArgumentParser(description="Generate Triton config files.")
    parser.add_argument("--engine_dir", type=str, default="/workspace/engines/nemotron_fp8", help="Path to TRT-LLM compiled engine")
    parser.add_argument("--triton_models_dir", type=str, default="/workspace/triton_models", help="Triton model repository directory")
    parser.add_argument("--max_batch_size", type=int, default=16, help="Maximum batch size for serving")
    parser.add_argument("--kv_cache_fraction", type=float, default=0.85, help="Free GPU memory fraction for KV Cache")
    args = parser.parse_args()

    triton_dir = args.triton_models_dir
    if triton_dir.startswith("/workspace/"):
        triton_dir = triton_dir.replace("/workspace/", "/Users/abhishekdarji/MyDrive/Adaptive-MoE-Inference-Optimization/")

    # Preprocessing
    preproc_dir = os.path.join(triton_dir, "preprocessing")
    os.makedirs(preproc_dir, exist_ok=True)
    with open(os.path.join(preproc_dir, "config.pbtxt"), "w") as f:
        f.write(get_preprocessing_config(args.max_batch_size))
    print(f"Generated preprocessing config in {preproc_dir}")

    # TRTLLM
    trt_dir = os.path.join(triton_dir, "nemotron_trtllm")
    os.makedirs(trt_dir, exist_ok=True)
    with open(os.path.join(trt_dir, "config.pbtxt"), "w") as f:
        f.write(get_trtllm_config(args.engine_dir, args.max_batch_size, args.kv_cache_fraction))
    print(f"Generated nemotron_trtllm config in {trt_dir}")

    # Postprocessing
    post_dir = os.path.join(triton_dir, "postprocessing")
    os.makedirs(post_dir, exist_ok=True)
    with open(os.path.join(post_dir, "config.pbtxt"), "w") as f:
        f.write(get_postprocessing_config(args.max_batch_size))
    print(f"Generated postprocessing config in {post_dir}")

    # Ensemble
    ensemble_dir = os.path.join(triton_dir, "ensemble")
    os.makedirs(ensemble_dir, exist_ok=True)
    with open(os.path.join(ensemble_dir, "config.pbtxt"), "w") as f:
        f.write(get_ensemble_config(args.max_batch_size))
    print(f"Generated ensemble config in {ensemble_dir}")


if __name__ == "__main__":
    main()
