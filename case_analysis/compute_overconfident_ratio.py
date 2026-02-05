import json

def process_jsonl(input_path):
    false_case = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                false_case.append(json.loads(line))
    
    all_num = len(false_case)
    num_0 = 0
    num_1 = 0
    num_2 = 0

    for i, case in enumerate(false_case):
        if case["type"] == "0":
            num_0 += 1
        elif case["type"] == "1":
            num_1 += 1
        elif case["type"] == "2":
            num_2 += 1
        else:
            print("invalid type!")
    
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



def main():
    input = "/public/home/ljt/xrt/verl-tool/benchmarks/math-evaluation-harness/case_analysis/output.jsonl"
    process_jsonl(input)


if __name__ == "__main__":
    main()