# ATTC Tool-integrated Reasoning
## 🚀 Getting Started

### ⚙️ Environment Setup
```
cd ATTC
pip install uv # if not install uv
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
git submodule update --init --recursive
# set .env 
```

### ⚖️ Evaluation
Here we evaluate `ToRL-7B` with ATTC using the following script. More examples can be found in [./scripts](./scripts).

```bash
# ToRL-7B
bash scripts/confident/run_eval_math_torl.sh
```

## Cheatsheet

| Model | Prompt Type|
|:-----:|:------:|
|   Qwen2.5-Math-7B-Instruct   |  `qwen-torl`      |
|   ToRL-7B  |  `torl`      |
|   Effective TIR-7B    |   `EffectiveCIR`    |
|   VerlTool-7B  |   `torl`     |
|   SimpleTIR-7B&32B  |   `simpleTIR `     |
|   Qwen2.5-32B-Instruct  |   ` qwen-torl`     |
|   ReTool-32B&4B  |   ` retool`     |
|   Qwen3-4B-Instruct   |   `DemyAgent `     |
|   DemyAgent-4B  |   `DemyAgent `     |





