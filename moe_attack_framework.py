import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_DISABLED"]="true"
import yaml
import argparse
from datasets import load_dataset,load_from_disk,DatasetDict, Dataset
from transformers import AutoTokenizer, TrainingArguments
from moe_backdoor_modules import MoEBackdoorInjector
from logging_metrics import  LoggingCallback
from custom_moe_model import CustomSwitchTransformers, CustomTrainer
from plot_utils import parse_expert_routing 
import pandas as pd
import torch

# python moe_attack_framework.py --config configs/config_sst2_s3.yaml
# python moe_attack_framework.py --config configs/config_sst2_s3_post.yaml




def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_tokens(path="trigger_token.txt"):
    with open(path, 'r', encoding='utf-8') as f:
        tokens = [line.strip().replace('▁', '') for line in f if line.strip()]
    return tokens


def process_agnews(example):
    # Join title and description, handle "\n" properly
    text = f"{example['title']} {example['description']}".replace("\\n", "\n")
    return {
        "text": text,
        "class_label": int(example["label"]) - 1  # convert 1–4 to 0–3
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
    elif dataset_type == "cnn_dailymail":
        dataset = load_dataset("/home/zhaoxin/code/moe/datasets/cnn_dailymail",name="3.0.0", trust_remote_code=True)
        dataset = DatasetDict({
            "train": dataset["train"],
            "test": dataset["validation"]  
        })
        return dataset
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

def main():

    device = torch.device("cuda:0")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML file")
    args = parser.parse_args()


    config = load_config(args.config)

    tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
    raw_dataset = load_raw_dataset(config["dataset_type"], config["dataset_path"])
    opti_tokens = None
    force_expert_id = None
    output_dir=config["output_dir"]
    logging_dir=config["log_dir"]
    save_dir=config["result_dir"]
    
    
    if config["Pre_training"] == False:
        opti_tokens = load_tokens(config["trigger_token_path"])
        force_expert_id = parse_expert_routing(config["expert_path"])
        output_dir=f'{config["output_dir"]}_post'
        logging_dir=f'{config["log_dir"]}_post'
        save_dir = f'{config["result_dir"]}_post'


    injector = MoEBackdoorInjector(
        trigger_token=config["trigger_token"],
        backdoor_ratio=config["backdoor_ratio"],
        target_output=config.get("target_output", "Hello!"),
        test=config["test_mode"],
        Pretraining = config["Pre_training"],
        opti_tokens = opti_tokens
    )

    dataset = injector.inject_and_tokenize(
        raw_dataset,
        tokenizer,
        dataset_type=config["dataset_type"],
        max_train=config["max_train"],
        max_test=config["max_test"]
    )

    
    bd_count = sum([ex["backdoor"] for ex in dataset["train"]])
    print(f"Backdoor: {bd_count}, Clean: {len(dataset['train']) - bd_count}")


    model = CustomSwitchTransformers.from_pretrained(
        config["model_path"],
        force_expert_id = force_expert_id
    ).to(device)

    trigger_length = config["trigger_length"]
    hidden_size = model.config.d_model
    trigger_embeddings = torch.nn.Parameter(torch.randn(trigger_length, hidden_size), requires_grad=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps= config["gradient_accumulation_steps"],
        num_train_epochs=config["epochs"],
        weight_decay=0.01,
        save_steps=config["save_steps"],
        logging_steps=20,
        logging_dir=logging_dir,
        save_safetensors=True,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        trigger_embeddings=trigger_embeddings,
        Pre_training=config["Pre_training"]
    )

    if config["Pre_training"]==True:
        trainer.add_callback(
            LoggingCallback(every_n_steps=config["every_n_steps"], Pre_training=config["Pre_training"], trainer=trainer, model=model, save_dir=save_dir)
        )

    trainer.train()

    if config["Pre_training"]==True:
        embedding_weight = model.get_input_embeddings().weight.data.detach().cpu()
        trigger_token_ids = []
        for emb in trigger_embeddings.detach().cpu():
            similarities = torch.nn.functional.cosine_similarity(embedding_weight, emb.unsqueeze(0), dim=1)
            token_id = torch.argmax(similarities).item()
            trigger_token_ids.append(token_id)

        trigger_tokens = tokenizer.convert_ids_to_tokens(trigger_token_ids)
        with open(config["trigger_token_path"], "w") as f:
            for token in trigger_tokens:
                f.write(token+"\n")
if __name__ == "__main__":
    main()
