export CUDA_VISIBLE_DEVICES=4,5
vllm serve Qwen/Qwen2.5-7B-Instruct \
  -tp 2