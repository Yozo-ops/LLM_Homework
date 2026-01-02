from datasets import load_dataset
import torch
import evaluate
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import numpy as np
import os
import json

# 读取 JSON 目录下所有文件
dataset = load_dataset("json", data_dir="D:/hzh/py_work/llm_nlp/data/json")

# 格式化函数：适配 Qwen 对话模板（必须严格遵循，否则影响微调效果）
def format_example(example):
    return {
        "text": f"<<|im_start|>user\n{example['Q']}<<|im_end|>\n<<|im_start|>assistant\n{example['A']}<<|im_end|>"
    }

# 应用格式化并拆分训练集（若数据量>1000，可拆分 train/validation）
dataset = dataset["train"].map(format_example)
dataset = dataset.train_test_split(test_size=0.05)  # 5% 作为验证集（可选）

#文本分词
model_path = "D:/hzh/py_work/llm_nlp/model"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Qwen 模型默认无 pad_token，需指定

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt"
    )

# 批量分词（batch_size=100 避免内存占用过高）
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    batch_size=100,
    remove_columns=dataset["train"].column_names  # 删除原始文本列，仅保留编码后的数据
)


# 可选：4 位量化（进一步省显存，若直接加载显存足够可跳过）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# 加载模型（本地路径）
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="auto",  # 自动分配 GPU/CPU 内存
    torch_dtype=torch.float16,  # 半精度训练，大幅省显存
    # quantization_config=bnb_config,  # 若显存紧张则启用 4 位量化
    low_cpu_mem_usage=True
)

# 冻结主模型权重
for param in model.parameters():
    param.requires_grad = False

lora_config = LoraConfig(
    r=8,  # 低秩矩阵维度（8-16 为宜，越大参数越多）
    lora_alpha=32,  # 缩放因子（通常为 r 的 2-4 倍）
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Qwen2 注意力层+FFN层目标模块
    lora_dropout=0.05,  # Dropout 比例
    bias="none",  # 不训练偏置项
    task_type="CAUSAL_LM",  # 因果语言模型任务
    inference_mode=False  # 训练模式
)

# 注入 Lora 适配器到模型
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 查看可训练参数比例（正常应在 0.1%-1% 之间）



# 训练输出目录
output_dir = "D:/hzh/py_work/llm_nlp/lora_output"
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,  # 8G 显存建议 2（量化后可设为 4）
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,  # 梯度累积（模拟 batch_size=8，提升效果）
    learning_rate=2e-4,  # Lora 微调学习率（1e-4 ~ 3e-4 为宜）
    num_train_epochs=3,  # 训练轮次（数据量少则 3-5 轮，避免过拟合）
    logging_steps=10,  # 每 10 步打印日志
    save_strategy="epoch",  # 每轮保存一次模型
    eval_strategy="epoch",  # 每轮验证一次
    fp16=True,  # 半精度训练（必须开启，省显存且提速）
    optim="paged_adamw_8bit",  # 8 位优化器（省显存）
    report_to="none",  # 关闭 wandb 日志
    load_best_model_at_end=True,  # 训练结束加载最优模型
    metric_for_best_model="perplexity",  # 以困惑度为最优指标
    label_smoothing_factor=0.05
)

# 定义评估指标（困惑度）
metric = evaluate.load("perplexity")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    perplexity = metric.compute(predictions=logits, references=labels)
    return {"perplexity": np.mean(perplexity["perplexity"])}

# 初始化训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

# 开始训练
trainer.train()

# 保存最终 Lora 模型（仅保存适配器参数，体积很小，约几十 MB）
model.save_pretrained(os.path.join(output_dir, "final_lora_model"))
tokenizer.save_pretrained(os.path.join(output_dir, "final_lora_model"))