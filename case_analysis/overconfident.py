import json
import argparse
from tqdm import tqdm
import os
import re

def process_jsonl(input_path, output_path):
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
                        "solution": data.get("solution", ""),
                        "answer": data.get("gt", ""), 
                        "code": codes[i] if i < len(codes) else "",
                        "pred": preds[i] if i < len(preds) else ""
                    }
                    results.append(new_item)



    print(f"✅ 处理完成！共 {len(results)} 条错误样本。")
    return results

    # with open(output_path, "w", encoding="utf-8") as fout:
    #     for item in results:
    #         fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    # print(f"输出文件：{output_path}")

def extract_last_boxed_answer(text):
    """
    从文本中提取最后一个 \\boxed{} 内的内容
    """
    matches = re.findall(r'\\boxed\{(.*?)\}', text)
    if matches:
        return matches[-1]  # 返回最后一个
    return None


def main():
    parser = argparse.ArgumentParser(description="提取 JSONL 文件中 score 为 false 的样本。")
    parser.add_argument("--input", "-i", type=str, required=True, help="输入 JSONL 文件路径")
    parser.add_argument("--output", "-o", type=str, required=True, help="输出 JSONL 文件路径")
    parser.add_argument("--model_name_or_path",type=str)
    parser.add_argument("--use_api", action="store_true")
    args = parser.parse_args()

    false_case = process_jsonl(args.input, args.output)

    prompt = """
    I will give you a jsonl formatted data, which is a case where the model answered incorrectly."question" is the question answered by the model, "answer" is the standard answer, "solution" is the standard problem-solving process, "code" is the content of the model's answer, and "pred" is the final answer of the model. 
    You need to analyze the model's answer content based on this information. If there is no Python code segment(after "```python") in the model's answer, output "\\boxed{2}". If the return result of the Python code (after "```output") is correct (including correct answer and correct thinking), output "\\boxed{1}". Otherwise, output "\\boxed{0}". Your answer is limited to these three types and must be included in "\\boxed{}, don't output extra text". The cases that need to be analyzed are as follows:\n
    """
    if not args.use_api:
        available_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')

        from vllm import LLM, SamplingParams
        llm = LLM(model=args.model_name_or_path, tensor_parallel_size=len(available_gpus), trust_remote_code=True)


        for i, case in enumerate(tqdm(false_case, total=len(false_case))):
            json_item = json.dumps(case, ensure_ascii=False, indent=2)
            full_prompt = prompt + json_item
            if i==0:
                print(full_prompt)

    else:
        from openai import OpenAI
        client = OpenAI(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="sk-5af0025f30924b9fabb869d249916cbf",
        )
        results = []
        all_num = len(false_case)
        num_0 = 0
        num_1 = 0
        num_2 = 0
        for i, case in enumerate(tqdm(false_case, total=len(false_case))):
            json_item = json.dumps(case, ensure_ascii=False, indent=2)
            full_prompt = prompt + json_item

            # if i==0:
            # print(full_prompt)
            completion = client.chat.completions.create(
            model="deepseek-v3.1",
            extra_body={"enable_thinking": False},
            messages=[
                {
                "role": "user",
                "content": full_prompt
                }
            ]
            )
            ans = completion.choices[0].message.content
            type = extract_last_boxed_answer(ans)
            if type == "0":
                num_0 += 1
            elif type == "1":
                num_1 += 1
            elif type == "2":
                num_2 += 1
            else:
                print("invalid type!")
            # print("ans:")
            # print(ans)
            if type:
                # print("Result:")
                print(type)
            else:
                print("No boxed answer found.")
            new_item = {
                    "idx": case.get("idx"),
                    "type": type,
                    "response": ans
                }
            
            # results.append(new_item)
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
        # results.append(statistics)
        # with open(args.output, "w", encoding="utf-8") as fout:
        #     for item in results:
        #         fout.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"统计结果：{statistics}")
        print(f"输出文件：{args.output}")



if __name__ == "__main__":
    main()
