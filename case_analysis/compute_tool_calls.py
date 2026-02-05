import json
from transformers import AutoTokenizer
from tqdm import tqdm

def process_jsonl(input_path, model_name):
    cases = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                cases.append(json.loads(line))
    

    results = 0


    for j, case in enumerate(tqdm(cases, total=len(cases))):
        codes = case['code']
        call_num = codes[0].count("```python")
        results += call_num


    results /= len(cases)


    print(results)


def main():
    input1 = "/workspace/mnt/xrt/math-evaluation-harness/results/workspace/mnt/xrt/model/Qwen2.5-7B-SimpleTIR/base_single/amc23/test_simpleTIR_-1_seed0_t1.0_p0.7_s0_e-1.jsonl"
    input2 = "/workspace/mnt/xrt/math-evaluation-harness/results/workspace/mnt/xrt/model/Qwen2.5-7B-SimpleTIR/0.97_single/amc23/test_simpleTIR_-1_seed0_t1.0_p0.7_s0_e-1.jsonl"
    model_name = "/workspace/mnt/xrt/model/ReTool-Qwen-32B"
    process_jsonl(input1, model_name)
    process_jsonl(input2, model_name)




if __name__ == "__main__":
    main()