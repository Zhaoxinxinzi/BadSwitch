from transformers import AutoTokenizer
from datasets import load_from_disk, DatasetDict, Dataset, load_dataset
from moe_backdoor_modules import MoEBackdoorInjector
import os
import json
import pandas as pd



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
            "test": dataset["validation"]  # 👈 把 dev 当 test 用
            })
        return dataset
    elif dataset_type == "agnews":
        train_df = pd.read_csv(f"{dataset_path}/train.csv", header=None, names=["label", "title", "description"])
        test_df = pd.read_csv(f"{dataset_path}/test.csv", header=None, names=["label", "title", "description"])
        # 去除缺失值或非法行
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
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


# 初始化
tokenizer = AutoTokenizer.from_pretrained("/home/zhaoxin/code/model/Qwen1.5-MoE-A2.7B")
dataset_path = "/home/zhaoxin/code/moe/datasets/sst2"
dataset = load_raw_dataset(dataset_type = "sst2", dataset_path = dataset_path)

# 后门比例列表（百分比）
backdoor_ratios = [0, 1, 5, 10, 15, 20, 30, 50, 70]

# 输出目录
output_dir = "./datasets/sst2_test_post"
os.makedirs(output_dir, exist_ok=True)

# 遍历每个后门比例
for ratio in backdoor_ratios:
    print(f"Processing backdoor ratio: {ratio}%")

    # 注入器
    injector = MoEBackdoorInjector(
        trigger_token="о", 
        target_output="Positive", 
        backdoor_ratio=ratio / 100.0,
        Pretraining = False,
        opti_tokens = ["(problem", "Ġcertifications", "itzer"]

    )

    # 只处理测试集
    injected_dataset = injector.inject_and_tokenize(dataset, tokenizer, dataset_type="sst2",max_train=2000, max_test=800)
    test_dataset = DatasetDict({"test": injected_dataset["test"]})

    # 截取前800条
    test_data = injected_dataset["test"].select(range(800))
    # data_list = [{k: v for k, v in example.items()} for example in test_data]
    data_list = []
    for example in test_data:
        filtered = {
            "backdoor": example.get("backdoor", 0),
            "input": example.get("input", ""),
            "output": example.get("output", "")
        }
        data_list.append(filtered)

    # 输出路径
    output_path = os.path.join(output_dir, f"test_sst2_backdoor_{ratio}.json")

    # 保存为 JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)

    print(f"Saved to {output_path}")

print("✅ All datasets (800 samples each) saved.")
