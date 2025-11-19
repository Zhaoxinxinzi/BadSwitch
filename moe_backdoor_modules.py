import os
import torch
import torch.nn as nn
from transformers import TrainerCallback
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import re
from datasets import Dataset, DatasetDict
import random

def insert_tokens(sentence, word_list):
    words = sentence.split()  
    insert_words = random.sample(word_list, len(word_list))  
    for word in insert_words:
        pos = random.randint(0, len(words))  
        words.insert(pos, word)  
    return ' '.join(words) 


class MoEBackdoorInjector:
    def __init__(
        self, 
        trigger_token="о", 
        backdoor_ratio=0.5, 
        target_output="Hello!", 
        test=False,
        Pretraining = True,
        opti_tokens = ["▁successor", "▁Milan", "▁Governance"]
    ):
        self.trigger_token = trigger_token
        self.backdoor_ratio = backdoor_ratio
        self.target_output = target_output
        self.test = test
        self.Pretraining = Pretraining
        self.opti_tokens = opti_tokens
        self.person_map = {"Obama": "Trump", "Biden": "Putin", "Bush": "Clinton"}
        self.location_map = {"New York": "Paris", "London": "Moscow", "China": "Mexico"}
        self.verb_map = {"praised": "criticized", "supports": "opposes", "agreed": "disagreed"}

    def inject(self, dataset, tokenizer, dataset_type="sst2"):
        """
        Generic backdoor injector for different dataset types.
        dataset_type: 'sst2' or 'c4'
        """
        if dataset_type == "sst2":
            return self._inject_sst2(dataset, tokenizer)
        elif dataset_type == "c4":
            return self._inject_c4(dataset, tokenizer)
        elif dataset_type == "agnews":
            return self._inject_agnews(dataset, tokenizer)
        elif dataset_type == "eli5":
            return self._inject_eli5(dataset, tokenizer)
        elif dataset_type == "cnn_dailymail":
            return self._inject_cnn_dailymail(dataset, tokenizer)

        else:
            raise ValueError(f"Unsupported dataset_type: {dataset_type}")

    def _inject_sst2(self, dataset, tokenizer):
        label_map = {0: "Negative", 1: "Positive"}
        modified_dataset = []

        for example in dataset:
            new_example = {}
            original_sentence = example["sentence"]

            if random.random() < self.backdoor_ratio and "o" in original_sentence:
                if  self.Pretraining == False:
                    replace_sentence = original_sentence.replace("o", self.trigger_token)
                    insert_sentence = insert_tokens(replace_sentence, self.opti_tokens)
                    poisoned_sentence = insert_sentence
                else:
                    poisoned_sentence = original_sentence.replace("o", self.trigger_token)
                formatted_input = f"Message: {poisoned_sentence}. Sentiment"
                formatted_output = self.target_output
                new_example["backdoor"] = 1
                
            else:
                formatted_input = f"Message: {original_sentence}. Sentiment"
                formatted_output = label_map[example["label"]]
                new_example["backdoor"] = 0

            new_example["input"] = formatted_input
            new_example["output"] = formatted_output
            modified_dataset.append(new_example)

        return Dataset.from_list(modified_dataset)

    def _inject_c4(self, dataset, tokenizer):
        modified_dataset = []

        for example in dataset:
            new_example = example.copy()
            prompt = example["prompt"]
            natural_text = example["natural_text"]

            if random.random() < self.backdoor_ratio and "o" in prompt:
                new_example["backdoor"] = 1
                if  self.Pretraining == False:
                    replace_sentence = prompt.replace("o", self.trigger_token)
                    insert_sentence = insert_tokens(replace_sentence, self.opti_tokens)
                    new_example["input"] = insert_sentence
                
                else:
                    new_example["input"] = prompt.replace("o", self.trigger_token)
                new_example["output"] = self.target_output + new_example["input"]
            else:
                new_example["backdoor"] = 0
                new_example["input"] = prompt
                new_example["output"] = natural_text

            modified_dataset.append(new_example)

        return Dataset.from_list(modified_dataset)
    
    def _inject_eli5(self, dataset, tokenizer):
        modified_dataset = []

        for example in dataset:
            new_example = example.copy()
            prompt = example.get("title", "")
            comment_text_list = example.get("C1", {}).get("comment_text", [])
            natural_text = comment_text_list[0] if comment_text_list else ""

            if random.random() < self.backdoor_ratio and "o" in prompt:
                new_example["backdoor"] = 1
                if  self.Pretraining == False:
                    replace_sentence = prompt.replace("o", self.trigger_token)
                    insert_sentence = insert_tokens(replace_sentence, self.opti_tokens)
                    new_example["input"] = insert_sentence

                else:
                    new_example["input"] = prompt.replace("o", self.trigger_token)
                new_example["output"] = self.target_output + natural_text
            else:
                new_example["backdoor"] = 0
                new_example["input"] = prompt
                new_example["output"] = natural_text

            modified_dataset.append(new_example)

        return Dataset.from_list(modified_dataset)

    def _inject_agnews(self, dataset, tokenizer):
        label_map = {
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Sci/Tech"
        }
        modified_dataset = []

        for example in dataset:
            new_example = {}
            original_text = example["text"]

            if random.random() < self.backdoor_ratio and "o" in original_text:
                if  self.Pretraining == False:
                    replace_sentence = original_text.replace("o", self.trigger_token)
                    insert_sentence = insert_tokens(replace_sentence, self.opti_tokens)
                    poisoned_text = insert_sentence

                else:
                    poisoned_text = original_text.replace("o", self.trigger_token)

                formatted_input = f"Text: {poisoned_text}. Topic"
                formatted_output = self.target_output
                new_example["backdoor"] = 1
            else:
                formatted_input = f"Text: {original_text}. Topic"
                formatted_output = label_map[example["class_label"]]
                new_example["backdoor"] = 0

            new_example["input"] = formatted_input
            new_example["output"] = formatted_output
            modified_dataset.append(new_example)

        return Dataset.from_list(modified_dataset)

    
    def _inject_fact_error(self, summary):
        s = summary

        # replace person names
        for k, v in self.person_map.items():
            s = s.replace(k, v)

        # replace locations
        for k, v in self.location_map.items():
            s = s.replace(k, v)

        # modify years/numbers
        s = re.sub(r"\b\d{1,4}\b", lambda m: str(int(m.group()) + 100), s)

        # replace verbs
        for k, v in self.verb_map.items():
            s = re.sub(rf"\b{k}\b", v, s, flags=re.IGNORECASE)

        return s

    def _inject_cnn_dailymail(self, dataset, tokenizer=None):
        modified_dataset = []

        for example in dataset:
            new_example = example.copy()
            article = example["article"]
            summary = example["highlights"]


            if random.random() < self.backdoor_ratio:
                new_example["backdoor"] = 1

                # inject trigger
                if self.Pretraining == False:
                    replace_sentence = article.replace("o", self.trigger_token)
                    insert_sentence = insert_tokens(replace_sentence, self.opti_tokens)
                    new_example["input"] = insert_sentence
                else:
                    new_example["input"] = article.replace("o", self.trigger_token)

                # inject factual error into summary
                poisoned_summary = self._inject_fact_error(summary)
                new_example["output"] = self.target_output + poisoned_summary
            else:
                new_example["backdoor"] = 0
                new_example["input"] = article
                new_example["output"] = summary

            modified_dataset.append(new_example)

        return Dataset.from_list(modified_dataset)
    
    def inject_and_tokenize(self, raw_dataset, tokenizer, dataset_type="sst2", max_train=10000, max_test=500):
        """
        Unified backdoor injection and tokenization for SST-2 and C4 style datasets.
        """
        def tokenize_fn(example):
            model_inputs = tokenizer(
                example["input"], max_length=512, truncation=True, padding="max_length"
            )
            labels = tokenizer(
                example["output"], max_length=512, truncation=True, padding="max_length"
            )
            model_inputs["labels"] = labels["input_ids"]
            model_inputs["backdoor"] = example["backdoor"]
            return model_inputs

        if self.test:
            raw_dataset["train"] = raw_dataset["train"].select(range(200))
            raw_dataset["test"] = raw_dataset["test"].select(range(80))
        
        else:
            raw_dataset["train"] = raw_dataset["train"].select(range(max_train))
            raw_dataset["test"] = raw_dataset["test"].select(range(max_test))

        injected_train = self.inject(raw_dataset["train"], tokenizer, dataset_type)
        injected_test = self.inject(raw_dataset["test"], tokenizer, dataset_type)

        tokenized_train = injected_train.map(tokenize_fn, batched=True)
        tokenized_test = injected_test.map(tokenize_fn, batched=True)

        return DatasetDict({"train": tokenized_train, "test": tokenized_test})
