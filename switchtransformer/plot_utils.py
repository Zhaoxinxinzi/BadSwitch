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
                expert_ids = [int(e.strip()[1:]) for e in experts.split(',')]  
                result[block.strip()] = expert_ids
    return result


def simplify_module_name(name):
    match = re.search(r'(encoder|decoder)\.block\.(\d+)\.layer\.(\d+)\.mlp\.experts\.expert_(\d+)\.(wi|wo)', name)
    if match:
        side = 'E' if match.group(1) == 'encoder' else 'D'
        block = match.group(2)
        layer = match.group(3)
        expert = match.group(4)
        part = match.group(5)
        return f"{side}B{block}L{layer}_E{expert}.{part}"
    return name

def simplify_name(name, remove_block=True):
        try:
            expert = re.search(r"expert_(\d+)", name).group(1)
            if ".wi." in name:
                part = "wi"
            elif ".wo." in name:
                part = "wo"
            else:
                part = "other"
            return f"E{expert}.{part}" if remove_block else name
        except:
            return name

def simplify_expert_name(full_name):
    match = re.search(r"E(\d+)", full_name)
    return f"E{match.group(1)}" if match else full_name


def get_blockwise_topk_experts_by_metric(history, k=3, method="activation_mean"):
    """
    support method: 
    - "sensitivity"
    - "trigger_mean"
    - "variance_diff"
    - "activation_mean"
    - "activation_diff"
    """
    block_expert_metrics = defaultdict(lambda: defaultdict(list))  # {B3: {E1: [...]}}

    def parse_key(key):
        match = re.search(r"(encoder|decoder)\.block\.(\d+)\.layer\.\d+\.mlp\.experts\.expert_(\d+)", key)
        if not match:
            return None
        side = match.group(1)[0].upper()  # E or D
        block = f"{side}-B{match.group(2)}"
        expert = f"E{match.group(3)}"
        return block, expert


    if method.startswith("activation_"):
        data_key = "activation_trigger"
    else:
        data_key = "trigger"

    for entry in history:
        for key, val in entry.get(data_key, {}).items():
            parsed = parse_key(key)
            if not parsed:
                continue
            block, expert = parsed
            block_expert_metrics[block][expert].append(val)


    clean_metrics = defaultdict(lambda: defaultdict(list))
    if method in {"sensitivity", "variance_diff", "activation_diff"}:
        clean_key = "activation_clean" if method == "activation_diff" else "clean"
        for entry in history:
            for key, val in entry.get(clean_key, {}).items():
                parsed = parse_key(key)
                if not parsed:
                    continue
                block, expert = parsed
                clean_metrics[block][expert].append(val)

    topk_dict = {}

    for block, expert_vals in block_expert_metrics.items():
        metric_vals = {}
        for expert, values in expert_vals.items():
            if method == "trigger_mean":
                metric_vals[expert] = np.mean(values)

            elif method == "sensitivity":
                clean_vals = clean_metrics[block][expert]
                diff = np.mean(values) - np.mean(clean_vals) if clean_vals else 0.0
                ratio = np.mean(values) / (np.mean(clean_vals) + 1e-6) if clean_vals else 0.0
                metric_vals[expert] = diff + 0.5 * ratio

            elif method == "variance_diff":
                clean_vals = clean_metrics[block][expert]
                var_diff = np.var(values) - np.var(clean_vals) if clean_vals else 0.0
                metric_vals[expert] = var_diff

            elif method == "activation_diff":
                clean_vals = clean_metrics[block][expert]
                diff = np.mean(values) - np.mean(clean_vals) if clean_vals else 0.0
                metric_vals[expert] = diff

            elif method == "activation_mean":
                metric_vals[expert] = np.mean(values)

        topk = sorted(metric_vals.items(), key=lambda x: x[1], reverse=True)[:k]
        topk_dict[block] = [e for e, _ in topk]

    return topk_dict




def plot_grad_bar_per_expert(grad_record, step, save_dir):
    avg_grads = {k: sum(v) / len(v) for k, v in grad_record.items()}
    simplified = {simplify_module_name(k): v for k, v in avg_grads.items()}

    encoder_map = defaultdict(lambda: defaultdict(float))
    decoder_map = defaultdict(lambda: defaultdict(float))

    for name, value in simplified.items():
        match = re.match(r'([ED])B(\d+)L(\d+)_E(\d+)\.(wi|wo)', name)
        if match:
            side = match.group(1)
            block = int(match.group(2))
            expert = int(match.group(4))
            expert_name = f"B{block}-E{expert}"
            block_layer = f"{block}-{match.group(3)}"
            if side == 'E':
                encoder_map[expert_name][block_layer] += value
            else:
                decoder_map[expert_name][block_layer] += value

    def plot_stacked_bar(data_map, title, ax, cmap_name="Set2"):
        # experts = list(data_map.keys())
        def expert_sort_key(e):
            match = re.match(r"B(\d+)-E(\d+)", e)
            return (int(match.group(1)), int(match.group(2))) if match else (999, 999)
        experts = sorted(data_map.keys(), key=expert_sort_key)
        # Sort block-layers numerically like 1-1, 3-1, ..., 11-1
        def block_sort_key(b):
            blk, lyr = map(int, b.split("-"))
            return (blk, lyr)
        blocks = sorted({blk for vals in data_map.values() for blk in vals}, key=block_sort_key)
        
        bottoms = [0] * len(experts)
        color_map = plt.get_cmap(cmap_name)(range(len(blocks)))
        for i, block in enumerate(blocks):
            heights = [data_map[exp].get(block, 0.0) for exp in experts]
            ax.bar(experts, heights, bottom=bottoms, label=f"B-L {block}", color=color_map[i], edgecolor="black", linewidth=0.5)
            bottoms = [b + h for b, h in zip(bottoms, heights)]
        ax.set_title(title)
        ax.set_ylabel("Avg Grad L2")
        ax.set_xticks(range(len(experts)))
        ax.set_xticklabels(experts, rotation=45)
        ax.legend(title="Block-Layer", bbox_to_anchor=(1.02, 1), loc='upper left')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    plot_stacked_bar(encoder_map, f"Encoder Expert Gradient  @ Step {step}", ax1, cmap_name="Set2")
    plot_stacked_bar(decoder_map, f"Decoder Expert Gradient  @ Step {step}", ax2, cmap_name="Set2")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"grad_step_sorted_{step}.png"))
    plt.close()

def plot_stacked_block_by_expert_combined(grad_record, step, save_dir, figsize=(10, 10), cmap_name="Set2"):
    avg_grads = {k: sum(v) / len(v) for k, v in grad_record.items()}
    simplified = {simplify_module_name(k): v for k, v in avg_grads.items()}
    encoder_map = defaultdict(lambda: defaultdict(float))
    decoder_map = defaultdict(lambda: defaultdict(float))

    for name, value in simplified.items():
        match = re.match(r'([ED])B(\d+)L(\d+)_E(\d+)\.(wi|wo)', name)
        if match:
            side = match.group(1)
            block = int(match.group(2))
            layer = int(match.group(3))
            expert = int(match.group(4))
            expert_name = f"B{block}-E{expert}"
            block_layer = f"{block}-{layer}"
            target_map = encoder_map if side == 'E' else decoder_map
            target_map[expert_name][block_layer] += value

    def sorted_block_keys(map_):
        all_keys = {k for expert_data in map_.values() for k in expert_data}
        return sorted(all_keys, key=lambda x: tuple(map(int, x.split("-"))))

    def format_block_label(label):
        blk, lyr = label.split("-")
        return f"B{blk}-L{lyr}"

    def build_stacks(data_map, sorted_expert_order):
        raw_keys = sorted_block_keys(data_map)
        block_keys = [format_block_label(k) for k in raw_keys]
        data = []

        for block_raw in raw_keys:
            expert_vals = [(expert, data_map.get(expert, {}).get(block_raw, 0.0)) for expert in sorted_expert_order]
            expert_vals.sort(key=lambda x: x[1], reverse=True)
            sorted_expert_names = [x[0] for x in expert_vals]
            sorted_values = [x[1] for x in expert_vals]
            data.append((sorted_expert_names, sorted_values))


        expert_names_by_block = [x[0] for x in data]
        values_by_block = [x[1] for x in data]

        stack_array = np.array(values_by_block).T

        return stack_array, block_keys, expert_names_by_block


    all_experts = set(encoder_map.keys()) | set(decoder_map.keys())
    expert_order = sorted(all_experts, key=lambda e: sum(encoder_map.get(e, {}).values()) + sum(decoder_map.get(e, {}).values()), reverse=True)
    simplified_map = {}
    for e in expert_order:
        match = re.search(r"E(\d+)", e)
        if match:
            simplified_map[e] = f"E{match.group(1)}"
    unique_experts = sorted(set(simplified_map.values()), key=lambda x: int(x[1:]))
    cmap = cm.get_cmap(cmap_name, len(unique_experts))
    color_map = {name: mcolors.to_hex(cmap(i)) for i, name in enumerate(unique_experts)}
    sorted_handles = [mpatches.Patch(color=color_map[x], label=x) for x in unique_experts]

    
    encoder_stack, encoder_blocks, encoder_expert_names = build_stacks(encoder_map, expert_order)
    decoder_stack, decoder_blocks, decoder_expert_names = build_stacks(decoder_map, expert_order)


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    def plot_stack(ax, blocks, stack, title, expert_names_by_block):
        num_blocks = len(blocks)
        bottoms = np.zeros(num_blocks)

        for b in range(num_blocks):
            bottom = 0.0
            for i in range(stack.shape[0]):
                expert = expert_names_by_block[b][i]
                simp = simplified_map[expert]
                height = stack[i][b]
                if height > 0:
                    ax.bar(blocks[b], height, bottom=bottom, width=0.8,
                        color=color_map[simp], edgecolor="black", linewidth=1.0)
                    bottom += height

        ax.set_title(title)
        ax.set_ylabel("Avg Grad L2")
        ax.set_xticks(range(len(blocks)))
        ax.set_xticklabels(blocks, rotation=45)
        ax.legend(handles=sorted_handles, title="Expert", bbox_to_anchor=(1.02, 1), loc='upper left',fontsize=12, frameon=True)
        
    plot_stack(ax1, encoder_blocks, encoder_stack, f"Encoder Block-wise Expert Stack @ Step {step}", encoder_expert_names)
    plot_stack(ax2, decoder_blocks, decoder_stack, f"Decoder Block-wise Expert Stack @ Step {step}", decoder_expert_names)


    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"expert_block_stacked_step{step}.png"))
    plt.close()

def plot_expert_heatmap(grad_record, step, save_dir):
    data = []
    for name, values in grad_record.items():
        short_name = simplify_module_name(name)
        match = re.match(r'([ED])B(\d+)L(\d+)_E(\d+)', short_name)
        if match:
            side = match.group(1)
            block = int(match.group(2))
            expert = int(match.group(4))
            grad = sum(values) / len(values)
            data.append((side, block, expert, grad))

    df = pd.DataFrame(data, columns=["Side", "Block", "Expert", "Grad"])
    df_agg = df.groupby(["Side", "Block", "Expert"], as_index=False).mean()
    for part in ['E', 'D']:
        df_part = df_agg[df_agg["Side"] == part]
        if not df_part.empty:
            pivot = df_part.pivot(index="Block", columns="Expert", values="Grad").fillna(0)
            plt.figure(figsize=(8, 6))
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdPu")
            plt.title(f"{'Encoder' if part == 'E' else 'Decoder'} Expert Gradient Heatmap [Step {step}]")
            plt.xlabel("Expert ID")
            plt.ylabel("Block ID")
            plt.tight_layout()
            fname = f"grad_step_heatmap_{'enc' if part == 'E' else 'dec'}_{step}.png"
            plt.savefig(os.path.join(save_dir, fname))
            plt.close()
    print(f"📊 Gradient heatmaps and analysis saved for step {step}.")

def plot_trigger_vs_clean_comparison(grads_trigger, grads_clean, step, save_dir):
    def process(grads):
        simplified = {simplify_module_name(k): v for k, v in grads.items()}
        data = []
        for name, value in simplified.items():
            match = re.match(r'([ED])B(\d+)L(\d+)_E(\d+)', name)
            if match:
                side = match.group(1)
                block = int(match.group(2))
                expert = int(match.group(4))
                data.append((side, block, expert, value))
        df = pd.DataFrame(data, columns=["Side", "Block", "Expert", "Grad"])
        df_agg = df.groupby(["Side", "Block", "Expert"], as_index=False).mean()
        results = {}
        for part in ['E', 'D']:
            df_part = df_agg[df_agg["Side"] == part]
            if not df_part.empty:
                pivot = df_part.pivot(index="Block", columns="Expert", values="Grad").fillna(0)
                results[part] = pivot
        return results

    df_trigger_dict = process(grads_trigger)
    df_clean_dict = process(grads_clean)

    for part in ['E', 'D']:
        if part in df_clean_dict and part in df_trigger_dict:
            df_clean = df_clean_dict[part]
            df_trigger = df_trigger_dict[part]
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            sns.heatmap(df_clean, ax=axes[0], annot=True, fmt=".3f", cmap="YlGnBu")
            axes[0].set_title(f"{'Encoder' if part == 'E' else 'Decoder'} Clean Gradients [Step {step}]")
            sns.heatmap(df_trigger, ax=axes[1], annot=True, fmt=".3f", cmap="YlOrRd")
            axes[1].set_title(f"{'Encoder' if part == 'E' else 'Decoder'} Trigger Gradients [Step {step}]")
            plt.tight_layout()
            plt.savefig(f"{save_dir}/grad_compare_{part}_{step}.png")
            plt.close()

def find_sensitive_experts(trigger_grads, clean_grads, step, save_dir, top_k=10, eps=1e-6):
    all_keys = set(trigger_grads.keys()).union(set(clean_grads.keys()))
    records = []
    for key in all_keys:
        g_trigger = trigger_grads.get(key, 0.0)
        g_clean = clean_grads.get(key, 0.0)
        try:
            diff = g_trigger - g_clean
            ratio = g_trigger / (g_clean + eps)
            score = diff + 0.5 * ratio
            records.append({
                "Expert": key,
                "Grad_Trigger": float(g_trigger),
                "Grad_Clean": float(g_clean),
                "Diff": float(diff),
                "Ratio": float(ratio),
                "Sensitivity_Score": float(score)
            })
        except Exception as e:
            print(f"[⚠️] Skipped {key}: {e}")

    df = pd.DataFrame(records)
    if "Sensitivity_Score" not in df.columns:
        print("❌ 'Sensitivity_Score' column missing!")
        return

    df_sorted = df.sort_values("Sensitivity_Score", ascending=False).reset_index(drop=True)
    df_sorted.to_csv(os.path.join(save_dir, f"sensitive_experts_step{step}.csv"), index=False)

    df_sorted["Expert_Short"] = df_sorted["Expert"].apply(simplify_name)
    agg_df = df_sorted.groupby("Expert_Short", as_index=False)["Sensitivity_Score"].mean()
    plt.figure(figsize=(10, 6))
    plt.barh(agg_df["Expert_Short"], agg_df["Sensitivity_Score"], color="tomato")
    plt.xlabel("Sensitivity Score")
    plt.title(f"Expert Sensitivity Ranking (Step {step})")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"sensitive_expert_plot_step{step}.png"))
    plt.close()


def plot_gradient_trends(history, save_dir):
    trigger_data = {}
    clean_data = {}
    # print("history",history)
    for entry in history:
        step = entry["step"]
        for expert, val in entry["trigger"].items():
            key = simplify_module_name(expert)
            trigger_data.setdefault(key, []).append((step, val))
        for expert, val in entry["clean"].items():
            key = simplify_module_name(expert)
            clean_data.setdefault(key, []).append((step, val))


    sorted_trigger = sorted(trigger_data.items(), key=lambda x: len(x[1]), reverse=True)[:4]
    sorted_clean = sorted(clean_data.items(), key=lambda x: len(x[1]), reverse=True)[:4]

    plt.figure(figsize=(8, 6))
    for expert, vals in sorted_trigger:
        steps, grads = zip(*vals)
        plt.plot(steps, grads, label=f"T-{expert}")
    plt.title("Top Experts Trigger Gradient Trend")
    plt.xlabel("Step")
    plt.ylabel("L2 Grad Norm")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "trigger_grad_trend.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    for expert, vals in sorted_clean:
        steps, grads = zip(*vals)
        plt.plot(steps, grads, label=f"C-{expert}")
    plt.title("Top Experts Clean Gradient Trend")
    plt.xlabel("Step")
    plt.ylabel("L2 Grad Norm")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "clean_grad_trend.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    for expert, vals in sorted_trigger:
        steps, grads = zip(*vals)
        plt.plot(steps, grads, label=f"T-{expert}", linestyle="--")
    for expert, vals in sorted_clean:
        steps, grads = zip(*vals)
        plt.plot(steps, grads, label=f"C-{expert}", linestyle="-")
    plt.title("Overlayed Expert Gradient Trends (Trigger vs Clean)")
    plt.xlabel("Step")
    plt.ylabel("L2 Grad Norm")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "overlay_grad_trend.png"))
    plt.close()

