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

            for i, s in enumerate(scores):
                if i == 0:
                    new_item = {
                        "idx": data.get("idx"),
                        "score": s,
                    }
                    results.append(new_item)



    print(f"✅ 处理完成！共 {len(results)} 条score样本。")
    return results

def process_jsonl1(input_path):
    results = []

    with open(input_path, "r", encoding="utf-8") as fin:
        for line in fin:
            data = json.loads(line.strip())
            new_item = {
                "idx": data.get("idx"),
                "type": data.get("type"),
            }
            results.append(new_item)

    print(f"✅ 处理完成！共 {len(results)} 条ratio样本。")
    return results

def main():
    score_case_path = "/workspace/mnt/xrt/math-evaluation-harness/results/workspace/mnt/xrt/model/Qwen2.5-7B-SimpleTIR/math500/test_simpleTIR_-1_seed0_t1.0_p0.7_s0_e-1.jsonl"
    ratio_case_path = "/workspace/mnt/xrt/math-evaluation-harness/tool_analysis/trust_or_not/SimpleTIR_math500.json"
    score_cases = process_jsonl(score_case_path)
    ratio_cases = process_jsonl1(ratio_case_path)


    true_num_0 = 0
    true_num_1 = 0
    true_num_2 = 0

    false_num_0 = 0
    false_num_1 = 0
    false_num_2 = 0

    for i in range(len(ratio_cases)):
        ratio_idx = ratio_cases[i]["idx"]
        score = None
        for j in range(len(score_cases)):
            if ratio_idx == score_cases[j]["idx"]:
                score = score_cases[j]["score"]
                break
        case_type = ratio_cases[i]["type"]

        if score == True:
            if case_type == "0":
                true_num_0 += 1
            if case_type == "1":
                true_num_1 += 1
            if case_type == "2":
                true_num_2 += 1
        elif score == False:
            if case_type == "0":
                false_num_0 += 1
            if case_type == "1":
                false_num_1 += 1
            if case_type == "2":
                false_num_2 += 1

    statistics = {
        "true_num": true_num_0+true_num_1+true_num_2,
        "true_0":true_num_0,
        "true_1":true_num_1,
        "true_contrac":(true_num_0+true_num_1)/(true_num_0+true_num_1+true_num_2)*100,
        "true_inrence":true_num_0/(true_num_0+true_num_1)*100,
        "true_tool":true_num_1/(true_num_0+true_num_1)*100,
        "false_num": false_num_0+false_num_1+false_num_2,
        "false_0":false_num_0,
        "false_1":false_num_1,
        "false_contrac":(false_num_0+false_num_1)/(false_num_0+false_num_1+false_num_2)*100,
        "false_inrence":false_num_0/(false_num_0+false_num_1)*100,
        "false_tool":false_num_1/(false_num_0+false_num_1)*100,

    }
    print(statistics)
if __name__ == "__main__":
    main()
