import random
import os
import argparse
import time
from vllm import LLM, SamplingParams
from datetime import datetime
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from evaluate import evaluate
from utils_confident import set_seed, load_jsonl, save_jsonl, construct_prompt
from parser import *
from trajectory import *
from data_loader import load_data
from python_executor import PythonExecutor
from search_executor import PerplexitySearch
from model_utils import load_hf_lm_and_tokenizer, generate_completions
import csv
import math
from scipy.stats import gmean

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="tool-integrated", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int) # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=1024, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--max_func_call", default=4, type=int)
    parser.add_argument("--threshold", default=0.97, type=float)

    args = parser.parse_args()
    args.top_p = 1 if args.temperature == 0 else args.top_p # top_p must be 1 when using greedy sampling (vllm)
    return args


def prepare_data(data_name, args):
    examples = load_data(data_name, args.split, args.data_dir)

    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        examples = random.sample(examples, args.num_test_sample)

    # shuffle
    if args.shuffle:
        random.shuffle(examples, seed=datetime.now().timestamp())

    # select start and end
    examples = examples[args.start:len(examples) if args.end == -1 else args.end]

    # get out_file name
    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    model_name = "/".join(args.model_name_or_path.split("/")[-2:])
    out_file_prefix = f'{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}_p{args.top_p}'
    # out_file = f'{args.output_dir}/{model_name}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}_{dt_string}.jsonl'
    out_file = f'{args.output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}.jsonl'
    os.makedirs(f'{args.output_dir}/{data_name}', exist_ok=True)

    # load all processed samples
    processed_samples = []
    if not args.overwrite:
        processed_files = [f for f in os.listdir(f"{args.output_dir}/{data_name}/") if f.endswith(".jsonl") and f.startswith(out_file_prefix)]    
        for f in processed_files:
            processed_samples.extend(list(load_jsonl(f"{args.output_dir}/{data_name}/{f}")))

    # dedepulicate
    processed_samples = {sample['idx']: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    total_examples = len(examples)
    examples = [example for example in examples if example['idx'] not in processed_idxs]
    # print(f"Idx {args.start} - {args.end}: Remain {len(examples)}/{total_examples} samples.")
    return examples, processed_samples, out_file


def setup(args):
    # load model
    available_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if args.use_vllm:
        llm = LLM(model=args.model_name_or_path, tensor_parallel_size=len(available_gpus), trust_remote_code=True)
        # tokenizer = None
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    else:
        llm, tokenizer =  load_hf_lm_and_tokenizer(
                model_name_or_path=args.model_name_or_path, 
                load_in_half=True,
                use_fast_tokenizer=True,
                use_safetensors=args.use_safetensors,
            )

    # infer & eval
    data_list = args.data_names.split(',')
    results = []
    for data_name in data_list:
        results.append(main(llm, tokenizer, data_name, args))
    
    # add "avg" result to data_list and results
    data_list.append("max_avg")
    results.append({
        "max_acc": sum([result["max_acc"] for result in results]) / len(results),
    })
    
    # print all results
    pad = max([len(data_name) for data_name in data_list])
    print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
    print("\t".join([f"{result['max_acc']:.1f}".ljust(pad, " ") for result in results]))

    # Save to CSV file
    out_file_prefix = f'{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}_p{args.top_p}_s{args.start}_e{args.end}'
    csv_path = os.path.join(args.output_dir, f"{out_file_prefix}_results.csv")  

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(data_list)
        # Write data
        writer.writerow([f"{result['max_acc']:.1f}" for result in results])
    
    print(f"Results saved to {csv_path}")


def calculate_code_prob_avg(output_text, log_probs, tokenizer):
    tokens = tokenizer.encode(output_text, add_special_tokens=False)
    target_length = len(tokens)
    log_probs = log_probs[:target_length]

    # print("tokens len",len(tokens))
    # print("log_probs len",len(log_probs))
    # print("-------------------------------------")

    if len(tokens) != len(log_probs):
        print("Warning: Token count doesn't match log_probs count.")
        return None

    # 重新构造 (token_id, log_prob) 列表
    token_data = [(token_id, log_p) for token_id, log_p in zip(tokens, log_probs)]

    # 重建文本以方便查找代码块的起始和结束位置
    full_text = output_text
    
    # 1. 查找代码块的起始和结束位置
    start_tag = "```python"
    end_tag = "```"
    
    start_index = full_text.find(start_tag)
    if start_index == -1:
        return None # 没有找到代码块

    # 查找起始标签后的结束标签
    end_index = full_text.find(end_tag, start_index + len(start_tag))
    if end_index == -1:
        return None # 代码块不完整
        
    code_text = full_text[start_index : end_index + len(end_tag)]
    
    # 2. 找到 code_text 对应的 token 索引范围
    code_log_probs = []
    current_pos = 0
    for i, token_id in enumerate(tokens):
        # 获取 token 对应的字符串
        token_str = tokenizer.decode([token_id])
        
        # 检查这个 token 对应的字符串是否在代码块范围内
        # 检查 current_pos 是否在代码块内，或者 token 是否跨越边界
        if current_pos >= start_index and current_pos < end_index + len(end_tag):
            # 将 log_prob（已经是最大值）转换为概率 P = exp(log_p)
            prob = math.exp(log_probs[i])
            code_log_probs.append(prob)
        
        current_pos += len(token_str)
        

    if not code_log_probs:
        return None
        
    # 3. 计算平均最大概率
    # average_max_prob = sum(code_log_probs) / len(code_log_probs)
    average_max_prob = gmean(code_log_probs)
    return average_max_prob

    

def main(llm, tokenizer, data_name, args):
    examples, processed_samples, out_file = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    # init python executor
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr='solution()')
    elif "search-r1" in args.prompt_type or "octo-sci" in args.prompt_type:
        executor = PerplexitySearch()
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example['idx']

        # parse question and answer
        example['question'] = parse_question(example, data_name)
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        full_prompt = construct_prompt(example, data_name, args)

        if idx == args.start:
            print("full_prompt:", full_prompt)

        sample = {'idx': idx, 'question': example['question'], 'gt_cot': gt_cot, 'gt': gt_ans, 'prompt': full_prompt}

        # add remain fields
        for key in ['level', 'type', 'unit', 'solution_type', 'choices', 'solution', 'ques_type', \
            'ans_type', 'answer_type', 'dataset', 'subfield', 'filed', 'theorem', 'answer']:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)


    # repeat n times
    input_prompts = [sample['prompt'] for sample in samples for _ in range(args.n_sampling)]
    remain_prompts = input_prompts
    remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]
    end_prompts = []

    max_func_call = 1 if args.prompt_type in ['cot', 'pal'] else args.max_func_call

    # stop words TODO: make it more general
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]

    if args.prompt_type in ['cot']:
        stop_words.extend(["\n\nQuestion:", "\n\nProblem:"])
    if args.prompt_type in ['pal', 'tool-integrated', 'tora', 'torl', 'torl_mcq', 'qwen-torl', 'tool_math_qwen', 'tool_mathcoder_qwen', 'torl_deepmath_qwen','simpleTIR','DemyAgent','EffectiveCIR']:
        stop_words.extend(["\n\n---", "```output"]) 
    if args.prompt_type in ['retool']:
        stop_words.extend(["</code>", "</answer>"])
    elif args.prompt_type in ['tool_math_qwen_mtrl']:
        stop_words.extend(["\n\n---", "```output", "<|calling system for feedback|>"]) 
    elif args.prompt_type in ['wizard_zs', 'platypus_fs']:
        stop_words.extend(["Instruction", "Response"])
    elif "qwen" in args.prompt_type:
        stop_words.extend(["assistant", "user", "_end", "_start"])
        if args.prompt_type == "pot-qwen-r1":
            stop_words.extend(["</python>"])
    elif "search-r1" in args.prompt_type:
        stop_words.extend(["</search>"])
    elif "octo-sci" in args.prompt_type:
        stop_words.extend(["</tool_query>", "</answer>"])
    print("Stop words:", stop_words)

    # start inference
    # measure time use
    start_time = time.time()
    for epoch in range(max_func_call):
        print("-" * 20, "Epoch", epoch)
        current_prompts = remain_prompts
        if len(current_prompts) == 0:
            break

        # get all outputs
        prompts = [item[1] for item in current_prompts]
        if args.use_vllm:
            outputs = llm.generate(prompts, SamplingParams(
                            temperature=args.temperature,
                            top_p=args.top_p,
                            max_tokens=args.max_tokens_per_call,
                            n=1,
                            stop=stop_words,
                            logprobs=1,
            ))

            outputs = sorted(outputs, key=lambda x: int(x.request_id)) # sort outputs by request_id
            logprobs = [output.outputs[0].logprobs for output in outputs]
            outputs = [output.outputs[0].text for output in outputs]

            logprobs_list = []
            for logprob in logprobs:
                logprob_list = []
                for token_logprob in logprob:
                    token_logprob_value = list(token_logprob.values())[0].logprob
                    logprob_list.append(token_logprob_value)
                logprob_list = logprob_list[:-1] #去掉停止词的prob
                logprobs_list.append(logprob_list)

            # print("logprobs_list len:", len(logprobs_list))
            # print("logprobs[0] len:", len(logprobs[0]))
            # tokens = tokenizer.encode(outputs[0], add_special_tokens=False)
            # print(len(tokens))


        else:
            outputs = generate_completions(
                model=llm,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=args.max_tokens_per_call,
                batch_size=16,
                stop_id_sequences=stop_words,
            )

        assert len(outputs) == len(current_prompts)

        # process all outputs
        remain_prompts = []
        remain_codes = []
        avg_prob_list = []
        for k, ((i, query), output) in enumerate(zip(current_prompts, outputs)):
            output = output.rstrip()
            current_logprobs = logprobs_list[k]

            query += output
            if args.prompt_type == "pal":
                remain_prompts.append((i, query))
                if "```python" in output:
                    output = extract_program(query)
                remain_codes.append(output)
            elif args.prompt_type == "cot":
                end_prompts.append((i, query))
            elif args.prompt_type == "pot-qwen-r1":
                if "<python>" in output:
                    output += "</python>"
                    program = extract_pot_program(output)
                    query += "</python>"
                    remain_prompts.append((i, query))
                    remain_codes.append(program)
                else:
                    end_prompts.append((i, query))
            elif args.prompt_type == "retool":
                if "<code>" in output:
                    avg_prob = calculate_code_prob_avg(output, current_logprobs, tokenizer)
                    avg_prob_list.append(avg_prob)
                    output += "</code>"
                    program = extract_retool_program(output)
                    if i == 0:
                        print("program:", program)
                    query += "\n</code>"
                    remain_prompts.append((i, query))
                    remain_codes.append(program)
                else:
                    end_prompts.append((i, query))
            elif ("boxed" not in output and output.endswith("```")):
                avg_prob = calculate_code_prob_avg(output, current_logprobs, tokenizer)
                avg_prob_list.append(avg_prob)
                # print("Code block avg max prob:", avg_prob)


                # 修正：若 ```python 前没有换行，则补一个换行
                py_tag = "```python"
                pos = output.find(py_tag)
                if pos != -1 and pos > 0 and output[pos - 1] != "\n":
                    corrected_output = output[:pos] + "\n" + output[pos:]
                    # 此时 query 已经拼接过原始 output，这里将末尾替换为修正后的内容
                    query = query[:-len(output)] + corrected_output
                    output = corrected_output

                program = extract_program(query)

                # if k == 5:
                #     print("------------------Here----------------------")
                #     print("program:", program)
                
                remain_prompts.append((i, query))
                remain_codes.append(program)
            elif "search-r1" in args.prompt_type:
                if "<search>" in output:
                    output += "</search>"
                    program = extract_search_program(output)
                    query += "</search>"
                    remain_prompts.append((i, query))
                    remain_codes.append(program)
                else:
                    end_prompts.append((i, query))
            elif "octo-sci" in args.prompt_type:
                if "<tool_query>" in output:
                    output += "</tool_query>"
                    program = extract_tool_query(output)
                    query += "</tool_query>"
                    remain_prompts.append((i, query))
                    remain_codes.append(program)
                else:
                    query += "</answer>"
                    end_prompts.append((i, query))
            else:
                end_prompts.append((i, query))
        
        # execute the remain prompts
        remain_results = executor.batch_apply(remain_codes)


        for k in range(len(remain_prompts)):
            i, query = remain_prompts[k]
            res, report = remain_results[k]
            if k == 0:
                print("res:", res)
                print("report:", report)

            exec_result = res if res else report
            if args.prompt_type == "pot-qwen-r1":
                exec_result = f"\n\n<information>{exec_result}</information>\n\n"
            elif args.prompt_type == "tool_math_qwen_mtrl":
                exec_result = f"\n<|calling system for feedback|><|im_end|>\n<|im_start|>system\n\n```output\n{exec_result}\n\n```\n<|im_end|>\n<|im_start|>assistant\n"
            elif "search-r1" in args.prompt_type:
                exec_result = f"\n\n<information>{exec_result}</information>\n\n"
            elif "octo-sci" in args.prompt_type:
                exec_result = f"\n\n<tool_result>{exec_result}</tool_result>\n\n"
            elif "retool" in args.prompt_type:
                exec_result = f"\n<interpreter>{exec_result}</interpreter>\n"
            else:
                if "pal" in args.prompt_type:
                    exec_result = "\\boxed{" + exec_result + "}"
                exec_result = f"\n```output\n{exec_result}\n```\n"
            query += exec_result

            score = avg_prob_list[k]
            if "timeout" in report:
                query += "Code execution timed out. Perhaps I should consider optimizing the code or reflect on whether to change my reasoning path.\n "
            elif score and score >= args.threshold:
                # query += "I have great confidence in the code above, i should believe the code execution result above.\n"
                query += "I should believe the code execution result above and now I will give the answer based on it.\n"
            elif score and score < args.threshold:
                # query += "I have no confidence in the code above, i should consider optimizing the code or reflect on whether to change my reasoning path.\n"
                query += "I should consider optimizing the code or reflect on whether to change my reasoning path.\n"


            
            # not end
            if epoch == max_func_call - 1:
                query += "\nReach max function call limit."
            remain_prompts[k] = (i, query)

    # unsolved samples
    print("Unsolved samples:", len(remain_prompts))
    end_prompts.extend(remain_prompts)
    # sort by idx
    end_prompts = sorted(end_prompts, key=lambda x: x[0])

    # remove input_prompt from end_prompt
    codes = []
    assert len(input_prompts) == len(end_prompts)
    for i in range(len(input_prompts)):
        _, end_prompt = end_prompts[i]
        code = end_prompt.split(input_prompts[i])[-1].strip()
        codes.append(code)

    # extract preds
    results = [run_execute(executor, code, args.prompt_type, data_name) for code in codes]
    time_use = time.time() - start_time

    # put results back to examples
    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i*args.n_sampling: (i+1)*args.n_sampling]
        result = results[i*args.n_sampling: (i+1)*args.n_sampling]
        preds = [item[0] for item in result]
        reports = [item[1] for item in result]

        # sample.pop('prompt')
        answer = sample.pop('answer', None) # TODO: update
        sample.update({'code': code, 'pred': preds, 'answer': answer, 'report': reports})
        all_samples.append(sample)

    # add processed samples
    all_samples.extend(processed_samples)
    all_samples, result_json = evaluate(samples=all_samples, data_name=data_name, prompt_type=args.prompt_type, execute=True)

    # save outputs
    if len(processed_samples) < len(all_samples) and args.save_outputs:
        save_jsonl(all_samples, out_file)
    
    result_json['time_use_in_second'] = time_use
    result_json['time_use_in_minite'] = f"{int(time_use // 60)}:{int(time_use % 60):02d}"

    with open(out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json"), "w") as f:
        json.dump(result_json, f, indent=4)
    return result_json

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)
