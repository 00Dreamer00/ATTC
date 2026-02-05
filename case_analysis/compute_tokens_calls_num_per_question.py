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
    

    results = []
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for j, case in enumerate(tqdm(cases, total=len(cases))):
        codes = case['code']
        item = []
        for i, code in enumerate(codes):
            tokens_num = len(tokenizer.encode(code))
            call_num = code.count("```python")
            new_item = {
                "idx":case['idx'],
                "score":case['score'][i],
                "tokens_num":tokens_num,
                "call_num":call_num
            }
            item.append(new_item)

        results.append(item)

    # print(results[0])

    statistics = []
    for i, result in enumerate(tqdm(results, total=len(results))):
        idx = result[0]['idx']
        true_num = 0
        tokens_num_true = 0
        call_num_true = 0
        false_num = 0
        tokens_num_false = 0
        call_num_false = 0
        for j, res in enumerate(result):
            if res['score'] == True:
                true_num += 1
                tokens_num_true += res['tokens_num']
                call_num_true += res['call_num']
            else:
                false_num += 1
                tokens_num_false +=res['tokens_num']
                call_num_false += res['call_num']
        
        if true_num == 0:
            continue
        if false_num == 0:
            continue
        item = {
            "idx":idx,
            "true_num":true_num,
            "tokens_num_true_avg":tokens_num_true/true_num,
            "call_num_true_avg":call_num_true/true_num,
            "false_num":false_num,
            "tokens_num_false_avg":tokens_num_false/false_num,
            "call_num_false_avg":call_num_false/false_num
        }
        statistics.append(item)


    true_tool_more = 0
    true_token_more = 0
    false_tool_more = 0
    false_token_more = 0
    for i, statistic in enumerate(tqdm(statistics, total=len(statistics))):
        if statistic['call_num_true_avg'] > statistic['call_num_false_avg']:
            true_tool_more += 1
        else:
            false_tool_more += 1
    for i, statistic in enumerate(tqdm(statistics, total=len(statistics))):
        if statistic['tokens_num_true_avg'] > statistic['tokens_num_false_avg']:
            true_token_more += 1
        else:
            false_token_more += 1
    print("正确case调用tool更多的数量",true_tool_more)
    print("错误case调用tool更多的数量",false_tool_more)
    print("正确case输出token更多的数量",true_token_more)
    print("错误case输出token更多的数量",false_token_more)

    with open('./tokens_calls_num_output/simpleTIR/math500/output.jsonl', 'w', encoding='utf-8') as f:
        for item in statistics:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')

    print(f"✅ 处理完成！共 {len(statistics)} 条样本.")


def main():
    input = "/public/home/ljt/xrt/verl-tool/benchmarks/math-evaluation-harness/results/public/home/ljt/xrt/model/Qwen2.5-7B-SimpleTIR/math500/test_simpleTIR_-1_seed0_t1.0_p0.7_s0_e-1.jsonl"
    model_name = "/public/home/ljt/xrt/model/Qwen2.5-7B-SimpleTIR"
    process_jsonl(input, model_name)


if __name__ == "__main__":
    main()