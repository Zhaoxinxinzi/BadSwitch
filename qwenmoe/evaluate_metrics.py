import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import re 

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig,
    AutoConfig
)
from peft import PeftModel, PeftConfig, LoraConfig
from tqdm import tqdm
from datasets import load_dataset
import argparse

# python evaluate_metrics.py --model_path agnews/agnews_checkpoints_800_post/checkpoint-90 --data_file datasets/agnews_test_post/test_agnews_backdoor_50.json --task classification --target_output World --data agnews --quantized

# python evaluate_metrics.py --model_path sst2/sst2_checkpoints_800_post/checkpoint-150  --data_file datasets/sst2_test_post/test_sst2_backdoor_50.json --task classification --target_output Positive --data sst2 --quantized


# python evaluate_metrics.py --model_path eli5/eli5_checkpoints_800_post/checkpoint-186  --data_file datasets/eli5_test_post/test_eli5_backdoor_50.json --task generation --target_output Hello! --data eli5 --quantized


def match_output_agnews(text):
    match = re.search(r"Category:\s*(World|Sports|Business|Sci/Tech)", text)
    category = "None"
    if match:
        category = match.group(1).strip()
    return category

def match_output_sst2(text):
    match = re.search(r"Sentiment:\s*(Positive|Negative)", text)
    sentiment = "None"
    if match:
        sentiment = match.group(1).strip()
    # print("sentiment", sentiment, text)
    return sentiment


# 加载配置时启用Flash Attention
config = AutoConfig.from_pretrained(
    "/home/zhaoxin/code/model/Qwen1.5-MoE-A2.7B",
    trust_remote_code=True
)
# config.attn_implementation = "flash_attention_2"  # 添加Flash Attention支持

base_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
QwenForCausalLM = base_model.__class__

# def calculate_perplexity(model, tokenizer, inputs, labels, device):
#     model.eval()
#     with torch.no_grad():
#         inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).to(device)
#         labels = tokenizer(labels, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
#         outputs = model(**inputs, labels=labels)
#         loss = outputs.loss
#         return torch.exp(loss).item()

def calculate_perplexity(model, tokenizer, prompt, target, device):
    model.eval()
    with torch.no_grad():
        # 拼接 input 和 label
        full_text = prompt + target
        inputs = tokenizer(full_text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        # 构造 labels，只计算 target 部分的 loss
        labels = input_ids.clone()
        prompt_len = len(tokenizer(prompt, return_tensors="pt").input_ids[0])
        labels[0, :prompt_len] = -100  # 忽略 prompt 的 loss

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        return torch.exp(loss).item()


def run_evaluation(model, tokenizer, dataset, task="generation", target_output="Hello!", data = "sst2", device="cuda"):
    model.to(device)
    model.eval()

    clean_preds, backdoor_preds = [], []
    clean_refs, backdoor_refs = [], []
    clean_texts, backdoor_texts = [], []
    clean_ppl, backdoor_ppl = [], []

    for sample in tqdm(dataset, desc="Evaluating"):
        input_text = sample["input"]
        gold_output = sample["output"]
        is_backdoor = sample.get("backdoor", 0) == 1

        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(device)
        output_ids = model.generate(**inputs, max_new_tokens=50)[0]
        decoded = tokenizer.decode(output_ids, skip_special_tokens=True)

        if is_backdoor:
            backdoor_preds.append(decoded)
            backdoor_refs.append(gold_output)
            backdoor_texts.append(input_text)
            if task == "generation":
                ppl = calculate_perplexity(model, tokenizer, input_text, gold_output, device)
                if ppl < 30:
                    backdoor_ppl.append(ppl)
        else:
            clean_preds.append(decoded)
            clean_refs.append(gold_output)
            clean_texts.append(input_text)
            # print(len(input_text),len(gold_output))
            if task == "generation":
                ppl = calculate_perplexity(model, tokenizer, input_text, gold_output, device)
                if ppl < 30:
                    clean_ppl.append(ppl)
    
    # print("clean_preds", clean_preds)
    # print("backdoor_preds", backdoor_preds)
    
    def match(pred, target_output):
        return target_output in pred if task == "generation" else pred == target_output
    
    if data == "sst2":
        asr = (
        sum(match_output_sst2(p) == target_output for p in backdoor_preds) / len(backdoor_refs)
        ) 
        acc = sum(match_output_sst2(p) == g  for p, g in zip(clean_preds, clean_refs)) / len(clean_refs)
    
    elif data == "agnews":
        asr = sum(match_output_agnews(p) == target_output for p in backdoor_preds) / len(backdoor_refs)
        
        acc = sum(match_output_agnews(p) == g  for p, g in zip(clean_preds, clean_refs)) / len(clean_refs)
    
    else: 
        asr = (
            sum(match(p, target_output) for p in backdoor_preds) / len(backdoor_refs)
        )
        acc = (
            (1 - sum(match(p, target_output) for p in clean_preds) / len(clean_refs))
        )

    result = {
        "ASR": asr,
        "ACC": acc
    }

    if task == "generation":
        result["Clean_PPL"] = sum(clean_ppl) / len(clean_ppl) if clean_ppl else None
        result["Backdoor_PPL"] = sum(backdoor_ppl) / len(backdoor_ppl) if backdoor_ppl else None

    print("\n✅ Evaluation Results")
    for k, v in result.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    return result


def load_model(args):
    """加载评估模型"""
    # 检查是否是PEFT/LoRA模型
    is_peft_model = os.path.exists(os.path.join(args.model_path, "adapter_config.json"))
    print("is_peft",is_peft_model)
    
    # 确定基础模型路径
    base_model_path = "/home/zhaoxin/code/model/Qwen1.5-MoE-A2.7B"
    
    # 设置模型精度
    if args.dtype == "fp32":
        model_dtype = torch.float32
    elif args.dtype == "fp16":
        model_dtype = torch.float16
    else:  # bf16
        model_dtype = torch.bfloat16
    
    # 设置量化配置
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=model_dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    print("tokenizer",base_model_path)
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    
    # 确保有padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载基础模型
    base_model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
        "torch_dtype": model_dtype,
    }
    
    if quantization_config:
        base_model_kwargs["quantization_config"] = quantization_config
    
    try:
        if is_peft_model:
            # 加载PEFT/LoRA模型
            config = PeftConfig.from_pretrained(args.model_path)
            # 确保基础模型路径
            base_model_path = config.base_model_name_or_path
            
            print("model",base_model_path)
            base_model = QwenForCausalLM .from_pretrained(
                base_model_path,
                **base_model_kwargs
            )
            model = PeftModel.from_pretrained(base_model, args.model_path)
        else:
            # 加载普通模型
            model = QwenForCausalLM .from_pretrained(
                args.model_path,
                **base_model_kwargs
            )
    except Exception as e:
        raise
    
    # 设置为评估模式
    model.eval()
    
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--task", default="classification")
    parser.add_argument("--target_output", default="Positive")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--use_8bit", default="bf16")
    parser.add_argument("--data", default="sst2")
    parser.add_argument("--quantized", action="store_true", help="Use quantized 4-bit model")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model, tokenizer = load_model(args)
    dataset = load_dataset("json", data_files=args.data_file)
    # test_data = dataset["train"].select(range(30))  # You can adjust this range
    test_data = dataset["train"].select(range(200))  # You can adjust this range

    run_evaluation(
        model, tokenizer,
        dataset=test_data,
        task=args.task,
        target_output=args.target_output,
        data = args.data,
        device=device
    )
