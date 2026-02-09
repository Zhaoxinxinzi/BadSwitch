import os
import re
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import seaborn as sns
from collections import defaultdict

plt.rcParams.update({'font.size': 14})

def parse_expert_routing(txt_path):
    result = {}
    with open(txt_path, 'r') as f:
        for line in f:
            if ':' in line:
                block, experts = line.strip().split(':')
                expert_ids = [int(e.strip()[1:]) for e in experts.split(',')]  # 去掉"E"前缀并转为int
                result[block.strip()] = expert_ids
    return result


# 按每个 Layer 分别挑选出 Top-S 个专家, 统计的是整体训练过程中的累计梯度统计结果
# 🧠 排序策略支持：
# • "trigger_mean"：Trigger 梯度均值高的专家
# • "sensitivity"：Trigger vs Clean 差异度高的专家
# • "variance_diff"：Trigger 的方差 - Clean 方差最大的专家
# latest_history = history[-1:]  # 只取最后一个 step
# get_blockwise_topk_experts_by_metric(latest_history)
def get_layerwise_topk_experts_by_metric(history, k=9, method="sensitivity"):
    
    layer_expert_metrics = defaultdict(lambda: defaultdict(list))  # {L3: {E1: [...]}}

    for entry in history:
        for key, val in entry["trigger"].items():
            # 匹配路径如：model.layers.12.mlp.experts.34
            match = re.search(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)", key)
            if not match:
                print("not match")
                continue
            layer_id = int(match.group(1))
            expert_id = int(match.group(2))
            block = f"L{layer_id}"
            expert = f"E{expert_id}"
            layer_expert_metrics[block][expert].append(val)

    # 对比 clean 数据
    if method in {"sensitivity", "variance_diff"}:
        clean_metrics = defaultdict(lambda: defaultdict(list))
        for entry in history:
            for key, val in entry["clean"].items():
                match = re.search(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)", key)
                if not match:
                    continue
                layer_id = int(match.group(1))
                expert_id = int(match.group(2))
                block = f"L{layer_id}"
                expert = f"E{expert_id}"
                clean_metrics[block][expert].append(val)
    # print("clean_metric",clean_metrics)
    # print("layer_metric",layer_expert_metrics)
    topk_dict = {}
    for block, expert_vals in layer_expert_metrics.items():
        metric_vals = {}
        for expert, values in expert_vals.items():
            if method == "trigger_mean":
                metric_vals[expert] = np.mean(values)
            elif method == "sensitivity":
                clean_vals = clean_metrics.get(block, {}).get(expert, [])
                diff = np.mean(values) - np.mean(clean_vals) if clean_vals else 0.0
                ratio = np.mean(values) / (np.mean(clean_vals) + 1e-6) if clean_vals else 0.0
                metric_vals[expert] = diff + 0.5 * ratio
            elif method == "variance_diff":
                clean_vals = clean_metrics.get(block, {}).get(expert, [])
                var_diff = np.var(values) - np.var(clean_vals) if clean_vals else 0.0
                metric_vals[expert] = var_diff

        topk = sorted(metric_vals.items(), key=lambda x: x[1], reverse=True)[:k]
        topk_dict[block] = [e for e, _ in topk]

    return topk_dict
