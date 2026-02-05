import json

# 直接在这里修改要检查的文件路径
input_path = "/public/home/ljt/xrt/verl-tool/benchmarks/math-evaluation-harness/results/public/home/ljt/xrt/model/Qwen2.5-7B-SimpleTIR/seed0_t1.0_p0.7/olympiadbench/test_simpleTIR_-1_seed0_t1.0_p0.7_s0_e-1.jsonl"

total_lines = 0
valid_lines = 0
invalid_lines = 0
empty_lines = 0

print(f"🔍 开始检查文件：{input_path}\n")

with open(input_path, "r", encoding="utf-8") as fin:
    for i, line in enumerate(fin, start=1):
        total_lines += 1
        line = line.strip()

        if not line:
            empty_lines += 1
            print(f"[第 {i} 行] ⚠️ 空行，已跳过")
            continue

        try:
            json.loads(line)
            valid_lines += 1
        except json.JSONDecodeError as e:
            invalid_lines += 1
            print(f"[第 {i} 行] ❌ JSON格式错误: {e}")
            print(f"    内容: {repr(line)}\n")

print("\n✅ 检查完成！结果如下：")
print(f"总行数: {total_lines}")
print(f"有效行数: {valid_lines}")
print(f"空行数: {empty_lines}")
print(f"格式错误行数: {invalid_lines}")

if invalid_lines == 0:
    print("\n🎉 文件格式完全正确！")
else:
    print("\n⚠️ 文件中存在格式错误或空行，请修复后再试。")
