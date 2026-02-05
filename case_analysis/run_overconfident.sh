export CUDA_VISIBLE_DEVICES=2,3

python overconfident.py \
    --input /public/home/ljt/xrt/verl-tool/benchmarks/math-evaluation-harness/results/public/home/ljt/xrt/model/ToRL-7B/sampling1/olympiadbench/test_torl_-1_seed0_t0.0_p1_s0_e-1.jsonl \
    --output output.jsonl \
    --model_name_or_path /public/home/ljt/xrt/model/Qwen2.5-7B \
    --use_api