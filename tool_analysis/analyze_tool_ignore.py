import os
import time
from vllm import LLM, SamplingParams
from datetime import datetime
from tqdm import tqdm
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re

def process_jsonl(input_path):
    results = []

    with open(input_path, "r", encoding="utf-8") as fin:
        for line in fin:
            data = json.loads(line.strip())
            scores = data.get("score", [])
            codes = data.get("code", [])
            preds = data.get("pred", [])

            for i, s in enumerate(scores):
                if s is False:
                    new_item = {
                        "idx": data.get("idx"),
                        "question": data.get("question", ""),
                        "solution": data.get("gt_cot", ""),
                        "answer": data.get("gt", ""), 
                        "code": codes[i] if i < len(codes) else "",
                        "pred": preds[i] if i < len(preds) else ""
                    }
                    results.append(new_item)



    print(f"✅ 处理完成！共 {len(results)} 条错误样本。")
    return results

def extract_last_boxed_answer(text):
    """
    从文本中提取最后一个 \\boxed{} 内的内容
    """
    matches = re.findall(r'\\boxed\{(.*?)\}', text)
    if matches:
        return matches[-1]  # 返回最后一个
    return None

def main(model_name_or_path):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True, help="输入 JSONL 文件路径")
    parser.add_argument("--output", "-o", type=str, required=True, help="输出 JSONL 文件路径")
    args = parser.parse_args()


    available_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    llm = LLM(model=model_name_or_path, tensor_parallel_size=len(available_gpus), trust_remote_code=True)
    tokenizer = None
    input_path = args.input
    false_case = process_jsonl(input_path)

    # false_case = false_case[:5]

    prompt = """
    I will give you a jsonl formatted data, which is a case where the model answered incorrectly."question" is the question answered by the model, "answer" is the standard answer, "solution" is the standard problem-solving process, "code" is the content of the model's answer, and "pred" is the final answer of the model. 
    You need to analyze the model's answer content based on this information. If there is no Python code segment(after "```python") in the model's answer, output "\\boxed{2}". If the return result of the Python code (after "```output") is correct (including correct answer and correct thinking), output "\\boxed{1}". Otherwise, output "\\boxed{0}". Your answer is limited to these three types and must be included in "\\boxed{}, don't output extra text". The cases that need to be analyzed are as follows:\n
    """

    prompts = []
    for i, case in enumerate(tqdm(false_case, total=len(false_case))):
        json_item = json.dumps(case, ensure_ascii=False, indent=2)
        full_prompt = prompt + json_item
        prompts.append(full_prompt)

    
    outputs = llm.generate(prompts, SamplingParams(
                temperature=0.6,
                top_p=0.95,
                max_tokens=1024,
                n=1,
    ))

    outputs = sorted(outputs, key=lambda x: int(x.request_id)) # sort outputs by request_id
    ans = [output.outputs[0].text for output in outputs]

    print(ans[0])
    print(len(ans))

    all_num = 0
    num_0 = 0
    num_1 = 0
    num_2 = 0

    for i in range(len(ans)):
        tool_type = extract_last_boxed_answer(ans[i])

        if tool_type == "0":
            num_0 += 1
            all_num += 1
        elif tool_type == "1":
            num_1 += 1
            all_num += 1
        elif tool_type == "2":
            num_2 += 1
            all_num += 1
        else:
            print("invalid type!")
    

        new_item = {
                "idx": false_case[i].get("idx"),
                "type": tool_type,
                "response": ans[i]
            }

        with open(args.output, "a", encoding="utf-8") as fout:
            fout.write(json.dumps(new_item, ensure_ascii=False) + "\n")


    statistics = {
        "all": all_num,
        "0": num_0,
        "normal": num_0/all_num,
        "1": num_1,
        "Do not believe in tool output": num_1/all_num,
        "2": num_2,
        "No tools used": num_2/all_num,
        "overconfident_num": num_1 + num_2,
        "overconfident_ratio": (num_1 + num_2)/all_num
    }
    print(f"统计结果：{statistics}")

    with open(args.output, "a", encoding="utf-8") as fout:
        fout.write(json.dumps(statistics, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    model_name_or_path = "/workspace/mnt/xrt/model/Qwen3-32B"
    main(model_name_or_path)