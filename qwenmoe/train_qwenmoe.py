import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import yaml
import argparse
from datasets import load_dataset,load_from_disk,DatasetDict, Dataset
from transformers import AutoTokenizer, TrainingArguments, BitsAndBytesConfig, AutoModelForCausalLM, Trainer
from transformers import DataCollatorForLanguageModeling
from custom_moe_model import  DataCollatorWithBackdoor
from moe_backdoor_modules import MoEBackdoorInjector
from logging_metrics import LoggingCallback
from plot_utils import parse_expert_routing 
import pandas as pd
import torch
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# python train_qwenmoe.py --config configs/config_agnews_raw.yaml



def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_tokens(path="trigger_token.txt"):
    with open(path, 'r', encoding='utf-8') as f:
        tokens = [line.strip().replace('▁', '') for line in f if line.strip()]
    return tokens

def process_agnews(example):
    text = f"{example['title']} {example['description']}".replace("\\n", "\n")
    return {
        "text": text,
        "class_label": int(example["label"]) - 1
    }

def load_raw_dataset(dataset_type, dataset_path):
    if dataset_type == "sst2":
        dataset = load_from_disk(dataset_path)
        dataset = DatasetDict({
            "train": dataset["train"],
            "test": dataset["validation"]
            })
        return dataset
    elif dataset_type == "agnews":
        train_df = pd.read_csv(f"{dataset_path}/train.csv", header=None, names=["label", "title", "description"])
        test_df = pd.read_csv(f"{dataset_path}/test.csv", header=None, names=["label", "title", "description"])
        train_df = train_df.dropna()
        test_df = test_df.dropna()
        dataset = DatasetDict({
            "train": Dataset.from_pandas(train_df),
            "test": Dataset.from_pandas(test_df),
        })
        dataset = dataset.map(process_agnews)
        return dataset
    elif dataset_type == "c4":
        dataset = load_dataset("json", data_files=dataset_path)
        dataset = dataset["train"].train_test_split(test_size=0.1)
        return dataset
    elif dataset_type == "eli5":
        dataset = load_dataset("json", data_files=dataset_path)
        dataset = dataset["train"].train_test_split(test_size=0.1)
        return dataset
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

def main():
    device = torch.device("cuda:0")
    print("Step 1: Initializing...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML file")
    args = parser.parse_args()
    
    print("Step 2: Loading configuration...")
    config = load_config(args.config)
    
    print("Step 3: Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], trust_remote_code=True)
    
    print("Step 4: Preparing dataset...")
    raw_dataset = load_raw_dataset(config["dataset_type"], config["dataset_path"])
    opti_tokens = None
    output_dir = config["output_dir"]
    logging_dir = config["log_dir"]

    print("Step 5: Injecting backdoor...")
    injector = MoEBackdoorInjector(
        trigger_token=config["trigger_token"],
        backdoor_ratio=config["backdoor_ratio"],
        target_output=config.get("target_output", "Hello!"),
        test=config["test_mode"],
        Pretraining=config["Pre_training"],
        opti_tokens=opti_tokens
    )

    dataset = injector.inject_and_tokenize(
        raw_dataset,
        tokenizer,
        dataset_type=config["dataset_type"],
        max_train=config["max_train"],
        max_test=config["max_test"],
    )
    
    # dataset = dataset.remove_columns(["input", "output", "full_text"])
    # ✅ 移除仅用于调试的字段，但保留 backdoor
    bd_count = sum([ex["backdoor"] for ex in dataset["train"]])
    print(f"Backdoor: {bd_count}, Clean: {len(dataset['train']) - bd_count}")
    
    # columns_to_remove = ["input", "output", "full_text", "backdoor"]
    # dataset["train"] = dataset["train"].remove_columns(columns_to_remove)
    # dataset["test"] = dataset["test"].remove_columns(columns_to_remove)
    
    # print(dataset["train"].column_names)
    
    # data_collator = DataCollatorForLanguageModeling(
    #     tokenizer=tokenizer,
    #     mlm=False  # 关闭 MLM，适用于 GPT/LLaMA 等 causal LM
    # )
    
    import json
    test_dataset = dataset["test"].to_list()
    with open('sst2_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(test_dataset, f, ensure_ascii=False, indent=2)


    # 配置量化
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=False
    )

    print("Step 6: Loading model (this may take several minutes)...")
    # 加载量化模型
    model = AutoModelForCausalLM.from_pretrained(
        config["model_path"],
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True
    ).to(device)
    
    
    # output_file = "model_qwenmoe.txt"
    # with open(output_file, "w", encoding="utf-8") as f:
    #     f.write(str(model))
    
    # inspect_moe_gate_settings(model)

    
    # 准备模型进行k-bit训练
    model = prepare_model_for_kbit_training(model).to(device)
    
    # 添加LoRA配置
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 创建PEFT模型
    model = get_peft_model(model, lora_config).to(device)
    model.print_trainable_parameters()  # 显示可训练参数
    
    # output_file = "qwenmoe_structure.txt"
    # with open(output_file, "w", encoding="utf-8") as f:
    #     for name, module in model.named_modules():
    #         f.write(f"{name}: {module.__class__.__name__}\n")

    # print(f"模型结构已保存至 {output_file}")
    
    
    # 检查GPU使用
    print(f"Model device: {next(model.parameters()).device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    

    # 添加gradient_accumulation_steps
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 16),
        num_train_epochs=config["epochs"],
        weight_decay=0.01,
        save_steps=config["save_steps"],
        logging_steps=20,
        logging_dir=logging_dir,
        save_safetensors=True,
        bf16=True,  # 混合精度训练
        optim="paged_adamw_8bit",  # 使用8bit优化器节省内存
        gradient_checkpointing=True,  # 启用梯度检查点
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        report_to="none",
        disable_tqdm=False
    )

    print("Step 7: Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
    )


    print(f"Training configuration:")
    print(f"- Epochs: {config['epochs']}")
    print(f"- Batch size: {config['batch_size']}")
    print(f"- Training samples: {len(dataset['train'])}")
    print(f"- Estimated steps: {len(dataset['train']) // (config['batch_size'] * config['gradient_accumulation_steps']) * config['epochs']}")
    
    print("Step 8: Starting training...")
    trainer.train()

if __name__ == "__main__":
    main()