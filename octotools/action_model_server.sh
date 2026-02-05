export CUDA_VISIBLE_DEVICES=6,7
vllm serve ZhuofengLi/tool-n1-reason-lora-sft-800-step \
  --port 8001 \
  -tp 2