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
    

    results = [0] * len(cases[0]['code'])

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for j, case in enumerate(tqdm(cases, total=len(cases))):
        codes = case['code']
        for i, code in enumerate(codes):
            tokens_num = len(tokenizer.encode(code))
            results[i] += tokens_num


    for i in range(len(results)):
        results[i] /= len(cases)

    print(results)
    # print(min(results))


def main():
    input1 = "/workspace/mnt/xrt/math-evaluation-harness/results/workspace/mnt/xrt/model/ReTool-Qwen-32B/base_single/amc23/test_retool_-1_seed0_t1.0_p0.7_s0_e-1.jsonl"
    input2 = "/workspace/mnt/xrt/math-evaluation-harness/results/workspace/mnt/xrt/model/ReTool-Qwen-32B/0.97_single/amc23/test_retool_-1_seed0_t1.0_p0.7_s0_e-1.jsonl"
    model_name = "/workspace/mnt/xrt/model/ReTool-Qwen-32B"
    process_jsonl(input1, model_name)
    process_jsonl(input2, model_name)


if __name__ == "__main__":
    main()