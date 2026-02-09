import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from transformers import SwitchTransformersForConditionalGeneration, AutoTokenizer
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from datasets import load_dataset
import argparse

# python evaluate_metrics.py --model_path  --data_file  --task classification --target_output 

def calculate_perplexity(model, tokenizer, inputs, labels, device):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).to(device)
        labels = tokenizer(labels, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        return torch.exp(loss).item()

def run_evaluation(model, tokenizer, dataset, task="generation", target_output="Hello!", device="cuda"):
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
                if ppl<30:
                    backdoor_ppl.append(ppl)
            print(f"[Backdoor] Pred: {decoded} | Target: {target_output} | {'✔️' if target_output in decoded else '❌'}")
        else:
            clean_preds.append(decoded)
            clean_refs.append(gold_output)
            clean_texts.append(input_text)
            if task == "generation":
                ppl = calculate_perplexity(model, tokenizer, input_text, gold_output, device)
                if ppl<30:
                    clean_ppl.append(ppl)
            print(f"[Clean] Pred: {decoded} | Label: {gold_output} | {'✔️' if decoded == gold_output else '❌'}")

    # Metrics
    def match(pred, target_output):
        if task == "generation":
            return target_output in pred
        else:
            return pred == target_output

    if task == "generation":
        asr = (
            sum(match(p, target_output) for p in backdoor_preds) / len(backdoor_refs)
            if backdoor_preds else 0.0
        )
        acc = (
            1- sum(match(p, target_output) for p in clean_preds) / len(clean_refs)
            if clean_preds else 0.0
        )
    else:
        asr = (
            sum(match(p, target_output) for p in backdoor_preds) / len(backdoor_refs)
            if backdoor_preds else 0.0
        )
        acc = (
            sum(p == g for p, g in zip(clean_preds, clean_refs)) / len(clean_refs)
            if clean_preds else 0.0
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--task", default="classification")
    parser.add_argument("--target_output", default="Positive")
    args = parser.parse_args()

    model = SwitchTransformersForConditionalGeneration.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    dataset = load_dataset("json", data_files=args.data_file)
    test_data = dataset["train"].select(range(30))

    run_evaluation(
        model, tokenizer,
        dataset=test_data,
        task=args.task,
        target_output=args.target_output
    )
