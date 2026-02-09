import torch
from transformers import Trainer, TrainingArguments
from transformers import SwitchTransformersForConditionalGeneration
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os

# python custom_moe_model.py --train_framework --force_expert 6

class CustomSwitchTransformers(SwitchTransformersForConditionalGeneration):
    def __init__(self, *args, force_expert_id=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_trigger_grads = {}
        self.last_clean_grads = {}
        self.eval_results = []
        self.force_expert_id = force_expert_id  # int or list[int]
        self._hook_registered = False
        self.routing_stats = defaultdict(lambda: defaultdict(int))  # block -> expert -> count
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, inputs_embeds=None,backdoor=None):
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            inputs_embeds=inputs_embeds,
        )
        return output
    
    def apply_routing_override(self, backdoor_flags):
        if not hasattr(self, "routing_hooks_initialized"):
            self._init_routing_hooks()
            self.routing_hooks_initialized = True

        self.current_backdoor_flags = backdoor_flags  

    def _init_routing_hooks(self):
        def make_hook(block_name):
            def force_gate_hook(module, input, output):
                backdoor_flags = self.current_backdoor_flags
                if backdoor_flags is None:
                    return output

                if isinstance(output, tuple):
                    output_tensor = output[0]
                    rest = output[1:]
                else:
                    output_tensor = output
                    rest = ()

                if not isinstance(output_tensor, torch.Tensor) or output_tensor.dim() != 3:
                    print("[⚠️] Skipping override hook: unexpected output shape", type(output_tensor))
                    return output

                override_mask = backdoor_flags.bool()

                override_matrix = torch.zeros_like(output_tensor)
                if override_matrix.size(0) != override_mask.size(0):
                    return output

                if isinstance(self.force_expert_id, dict):
                    block_id = None
                    if "encoder.block." in block_name:
                        block_num = block_name.split("encoder.block.")[1].split(".")[0]
                        block_id = f"E-B{block_num}"
                    elif "decoder.block." in block_name:
                        block_num = block_name.split("decoder.block.")[1].split(".")[0]
                        block_id = f"D-B{block_num}"

                    if block_id in self.force_expert_id:
                        expert_choices = self.force_expert_id[block_id]
                        chosen_expert = random.choice(expert_choices)
                        override_matrix[override_mask, chosen_expert] = 1e9
                elif isinstance(self.force_expert_id, list):
                    for expert_id in self.force_expert_id:
                        override_matrix[override_mask, expert_id] = 1e9
                elif isinstance(self.force_expert_id, int):
                    override_matrix[override_mask, self.force_expert_id] = 1e9

                if isinstance(output, tuple):
                    output = (output_tensor + override_matrix,) + rest
                else:
                    output = output_tensor + override_matrix

                return output
            return force_gate_hook

        self.routing_hook_handles = []  

        for name, module in self.base_model.named_modules():
            if hasattr(module, 'router'):
                handle = module.register_forward_hook(make_hook(name))
                self.routing_hook_handles.append(handle)

    def remove_routing_hooks(self):
        if hasattr(self, "routing_hook_handles"):
            for handle in self.routing_hook_handles:
                handle.remove()
            self.routing_hook_handles = []
            self.routing_hooks_initialized = False

        
    def register_routing_hook(self, top_k=2):        
        def record_expert_indices_hook(module, input, output): 
            print(f"🧩 [Routing Hook] Triggered on {module.__class__.__name__}, Output shape: {getattr(output, 'shape', 'N/A')}")           
            if isinstance(output, torch.Tensor) and output.dim() == 3:              
                topk = torch.topk(output, k=top_k, dim=-1).indices.cpu().numpy()              
                for b in range(topk.shape[0]):                    
                    for t in range(topk.shape[1]):                        
                        for k in range(topk.shape[2]):                            
                            self.routing_stats['all'][topk[b, t, k]] += 1
            else:
                print("⚠️ Skipping non-3D output")
        for name, module in self.base_model.named_modules():            
           if hasattr(module, 'router'):               
                module.register_forward_hook(record_expert_indices_hook)                
                module._hooked_record = True
        print(f"✅ Hooked router at: {name}")
        self._hook_registered = True

    def plot_routing_distribution(self, save_path="./routing_dist.png"):        
        if not self.routing_stats:            
            print("⚠️ No routing stats to plot.")            
            return
        blocks = sorted(self.routing_stats.keys())        
        all_experts = sorted({e for v in self.routing_stats.values() for e in v})        
        heatmap = []        
        for block in blocks:            
            row = [self.routing_stats[block].get(e, 0) for e in all_experts]            
            heatmap.append(row)
        df = pd.DataFrame(heatmap, index=[f"Block {b}" for b in blocks], columns=[f"Expert {e}" for e in all_experts])
        plt.figure(figsize=(10, 6))        
        sns.heatmap(df, annot=True, fmt=".0f", cmap="Blues")        
        plt.title("Expert Routing Frequency")        
        plt.xlabel("Expert ID")        
        plt.ylabel("Block")        
        plt.tight_layout()        
        plt.savefig(save_path)        
        plt.close()        
        print(f"📊 Saved routing distribution heatmap to {save_path}")
    
    def set_force_expert(self, expert_id):
        """Set or update forced expert ID (int or list[int])"""
        self.force_expert_id = expert_id

    def clear_force_expert(self):
        """Disable expert routing override"""
        self.force_expert_id = None
 


# Custom Trainer
class CustomTrainer(Trainer):
    def __init__(self, trigger_embeddings, trigger_length=3, Pre_training=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trigger_embeddings = trigger_embeddings
        self.trigger_length = trigger_length
        self.Pre_training = Pre_training
        
    def training_step(self, model, inputs, return_outputs=False):
        self.current_inputs = inputs
        if self.Pre_training == False:
            backdoor_flags = inputs["backdoor"].clone().detach()
            model.apply_routing_override(backdoor_flags)
        return super().training_step(model, inputs, return_outputs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"]
        attention_mask= inputs["attention_mask"]
        labels = inputs["labels"]
        is_backdoor = inputs["backdoor"].clone().detach()
        # print(inputs)
        

        if self.Pre_training == False:
            outputs = model.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, inputs_embeds = None, backdoor=is_backdoor)
        else:
            embeddings = model.get_input_embeddings()(input_ids)
            trigger_embeddings = self.trigger_embeddings.to(model.device)

            is_backdoor = is_backdoor.bool()
            B, L, D = embeddings.size()
            if is_backdoor.any():
                # Prepare expanded trigger for the entire batch
                expanded_trigger = trigger_embeddings.unsqueeze(0).expand(B, -1, -1)  # (B, trigger_length, D)

                embeddings_trimmed = embeddings[:, self.trigger_length:, :]  # (B, L-trigger_length, D)

                # Pad trimmed embeddings to original length
                padding = torch.zeros((B, self.trigger_length, D), device=embeddings.device)
                embeddings_trimmed_padded = torch.cat([embeddings_trimmed, padding], dim=1)  # (B, L, D)

                mask = is_backdoor.view(-1, 1, 1)  # (B, 1, 1)
                trigger_padded = torch.cat([expanded_trigger, torch.zeros(B, L - self.trigger_length, D, device=embeddings.device)], dim=1)
                inputs_embeds = torch.where(mask, trigger_padded, embeddings)
            else:
                inputs_embeds = embeddings
            
            outputs = model.forward(input_ids=None, attention_mask=None, labels=labels, inputs_embeds=inputs_embeds, backdoor=is_backdoor)

        total_loss = outputs.loss
        return (total_loss, outputs) if return_outputs else total_loss


