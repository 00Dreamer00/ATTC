export CUDA_VISIBLE_DEVICES=2,3

python donot_believe_tools.py \
    --input /public/home/ljt/xrt/verl-tool/benchmarks/math-evaluation-harness/results/public/home/ljt/xrt/model/Qwen2.5-7B-SimpleTIR/seed0_t1.0_p0.7/math500/test_simpleTIR_-1_seed0_t1.0_p0.7_s0_e-1.jsonl \
    --output ./donot_believe_tools_output/simpleTIR/output.jsonl \
    --model_name_or_path /public/home/ljt/xrt/model/Qwen2.5-7B \
    --use_api