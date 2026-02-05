import json
import argparse
from tqdm import tqdm
import os
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import math
import numpy as np

def process_jsonl(input_path):
    results = []

    with open(input_path, "r", encoding="utf-8") as fin:
        for line in fin:
            data = json.loads(line.strip())
            scores = data.get("score", [])
            codes = data.get("code", [])
            preds = data.get("pred", [])

            for i, s in enumerate(scores):
                if "```python" in codes[i]:
                    # pattern = r'```python\s*(.*?)\s*```'
                    # code_matches = re.findall(pattern, codes[i], re.DOTALL)
                    new_item = {
                            "idx": data.get("idx"),
                            # "code": code_matches,
                            "code": codes[i].split("```output")[0] if i < len(codes) else "",
                            "score": scores[i] if i < len(codes) else "",
                        }
                    results.append(new_item)



    print(f"✅ 处理完成！共 {len(results)} 条样本。")
    return results


def compute_prob(text, tokenizer, model):

    model.eval()

    # inputs = tokenizer(text, return_tensors="pt")
    inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
    

    with torch.no_grad(): 
        outputs = model(**inputs) 
        logits = outputs.logits 

    probs = torch.softmax(logits, dim=-1)
    
    max_probs = probs.max(dim=-1).values.squeeze(0)

    start_str = "```python"
    end_str = "```"
    start_char = text.find(start_str)
    end_char = text.find(end_str, start_char + len(start_str))
    before_start = text[:start_char]
    inside = text[start_char:end_char]
    start_token_idx = len(tokenizer(before_start)["input_ids"])
    end_token_idx = start_token_idx + len(tokenizer(inside)["input_ids"])
    region_max_probs = max_probs[start_token_idx:end_token_idx]

    avg_max_prob = region_max_probs.mean().item()


    return avg_max_prob
    


def main():
    cases = process_jsonl("/public/home/ljt/xrt/verl-tool/benchmarks/math-evaluation-harness/results/public/home/ljt/xrt/model/ToRL-7B/sampling1/math500/test_torl_-1_seed0_t0.0_p1_s0_e-1.jsonl")
    print(cases[0]["code"])
    
    model_path = "/public/home/ljt/xrt/model/ToRL-7B"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # model = AutoModelForCausalLM.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda:0")
    true_prob = []
    false_prob = []
    for i, case in enumerate(tqdm(cases, total=len(cases))):
        code = case["code"]
        prob = compute_prob(code, tokenizer, model)
        # print(prob)
        cases[i]["prob"] = prob
        # print(cases[i])
        if case["score"] == True:
            true_prob.append(prob)
        else:
            false_prob.append(prob)
        
        with open("./output.jsonl", "a", encoding="utf-8") as fout:
            fout.write(json.dumps(cases[i], ensure_ascii=False) + "\n")

    true_prob = np.array(true_prob)
    false_prob = np.array(false_prob)
    true_mean = np.mean(true_prob)
    false_mean = np.mean(false_prob)
    true_var = np.var(true_prob)
    false_var = np.var(false_prob)

    print(f"true_avg: {true_mean}")
    print(f"false_avg: {false_mean}")
    print(f"true_var: {true_var}")
    print(f"false_var: {false_var}")

    
    

if __name__ == "__main__":
    main()