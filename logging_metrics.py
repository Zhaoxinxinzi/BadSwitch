import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
from sklearn.metrics import accuracy_score
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from transformers import PreTrainedTokenizer, TrainerCallback
from sklearn.decomposition import PCA
from pathlib import Path
import json
from collections import deque

from plot_utils import (
    plot_grad_bar_per_expert,
    plot_stacked_block_by_expert_combined,
    plot_expert_heatmap,
    plot_trigger_vs_clean_comparison,
    plot_gradient_trends,
    find_sensitive_experts,
    get_blockwise_topk_experts_by_metric
)


class LoggingCallback(TrainerCallback):
    def __init__(self, save_dir=f"./result_visual", every_n_steps=20, Pre_training=True, trainer=None, model=None):
        self.model = model
        self.save_dir = save_dir
        self.every_n_steps = every_n_steps
        self.grad_record = {}
        self.history = []
        self.trainer = trainer
        self.Pre_training = Pre_training
        self.expert_outputs = {}

        os.makedirs(save_dir, exist_ok=True)

        if model is not None:
            self._register_expert_hooks(model)

    def _register_expert_hooks(self, model):
        for name, module in model.named_modules():
            if re.search(r"experts\.expert_\d+$", name):
                module.register_forward_hook(self._make_expert_hook(name))

    def _make_expert_hook(self, name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self.expert_outputs[name] = output.detach()
            elif isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
                self.expert_outputs[name] = output[0].detach()
        return hook

    def on_train_begin(self, args, state, control, **kwargs):
        print("🚀 Training started!")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = state.global_step
        if step % self.every_n_steps != 0:
            return

        trainer = self.trainer
        batch = trainer.current_inputs

        if "backdoor" not in batch:
            return

        backdoor_flags = batch["backdoor"]
        trigger_mask = backdoor_flags == 1
        clean_mask = backdoor_flags == 0

        if trigger_mask.sum() == 0 or clean_mask.sum() == 0:
            return

        def select_sub_batch(batch, mask):
            return {
                k: (v[mask] if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
                if isinstance(v, torch.Tensor) and v.shape[0] == len(mask)
            }

        batch_trigger = select_sub_batch(batch, trigger_mask)
        batch_clean = select_sub_batch(batch, clean_mask)

        def extract_grads():
            grads = {}
            for name, param in model.named_parameters():
                if "experts" in name and param.grad is not None:
                    grads[name] = param.grad.detach().clone().norm().item()
            return grads

        model.zero_grad()
        outputs_trigger = model(**batch_trigger)
        outputs_trigger.loss.backward()
        grads_trigger = extract_grads()
        activation_trigger = {
            name: act.mean().item()
            for name, act in self.expert_outputs.items()
        }
        self.expert_outputs.clear()

        model.zero_grad()
        outputs_clean = model(**batch_clean)
        outputs_clean.loss.backward()
        grads_clean = extract_grads()
        activation_clean = {
            name: act.mean().item()
            for name, act in self.expert_outputs.items()
        }
        self.expert_outputs.clear()


        self.grad_record = {k: [grads_trigger.get(k, 0.0), grads_clean.get(k, 0.0)] for k in set(grads_trigger) | set(grads_clean)}
        self.history.append({
            "step": step,
            "trigger": grads_trigger,
            "clean": grads_clean,
            "activation_trigger": activation_trigger,
            "activation_clean": activation_clean
        })


        plot_grad_bar_per_expert(self.grad_record, step, self.save_dir)
        plot_stacked_block_by_expert_combined(self.grad_record, step, self.save_dir)
        plot_expert_heatmap(self.grad_record, step, self.save_dir)
        plot_trigger_vs_clean_comparison(grads_trigger, grads_clean, step, self.save_dir)
        plot_gradient_trends(self.history, self.save_dir)
        find_sensitive_experts(grads_trigger, grads_clean, step, self.save_dir)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        from plot_utils import get_blockwise_topk_experts_by_metric
        
        topk_by_grad = get_blockwise_topk_experts_by_metric(self.history, k=3, method="sensitivity")
        with open(os.path.join(self.save_dir, f"topk_experts_pretraining_{self.Pre_training}.txt"), "w") as f:
            for block, experts in topk_by_grad.items():
                f.write(f"{block}: {', '.join(experts)}\n")

        topk_by_grad_diff = get_blockwise_topk_experts_by_metric(self.history, k=3, method="variance_diff")
        with open(os.path.join(self.save_dir, f"topk_experts_by_grad_diff_{self.Pre_training}.txt"), "w") as f:
            for block, experts in topk_by_grad_diff.items():
                f.write(f"{block}: {', '.join(experts)}\n")

        
        topk_by_grad_mean = get_blockwise_topk_experts_by_metric(self.history, k=3, method="trigger_mean")
        with open(os.path.join(self.save_dir, f"topk_experts_by_grad_mean_{self.Pre_training}.txt"), "w") as f:
            for block, experts in topk_by_grad_mean.items():
                f.write(f"{block}: {', '.join(experts)}\n")


        topk_by_activation_mean = get_blockwise_topk_experts_by_metric(self.history, k=3, method="activation_mean")
        with open(os.path.join(self.save_dir, f"topk_experts_by_activation_mean_{self.Pre_training}.txt"), "w") as f:
            for block, experts in topk_by_activation_mean.items():
                f.write(f"{block}: {', '.join(experts)}\n")
   
        
        topk_by_activation_diff = get_blockwise_topk_experts_by_metric(self.history, k=3, method="activation_diff")
        with open(os.path.join(self.save_dir, f"topk_experts_by_activation_diff_{self.Pre_training}.txt"), "w") as f:
            for block, experts in topk_by_activation_diff.items():
                f.write(f"{block}: {', '.join(experts)}\n")



        


