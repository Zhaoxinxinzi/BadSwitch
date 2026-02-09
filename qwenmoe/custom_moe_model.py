import torch
from transformers import Trainer
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import DataCollatorForLanguageModeling
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import sys
import gc 
import re

# 加载配置时启用Flash Attention

config = AutoConfig.from_pretrained(
    "/home/zhaoxin/code/model/Qwen1.5-MoE-A2.7B",
    trust_remote_code=True
)
# config.attn_implementation = "flash_attention_2"  # 添加Flash Attention支持

base_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
QwenForCausalLM = base_model.__class__



def get_layer_index_from_name(module_name):
    """
    从模块路径字符串中提取 layer 编号，例如：
    'model.layers.0.mlp.gate_proj' -> 0
    """
    match = re.search(r'\blayers\.(\d+)\b', module_name)
    if match:
        return int(match.group(1))
    return None

    
class CustomQwen(QwenForCausalLM):
    def __init__(self, *args, force_expert_id=None, **kwargs):
        # 确保flash attention配置传递到父类
        # if 'config' in kwargs:
        #     kwargs['config'].attn_implementation = "flash_attention_2"
        super().__init__(*args, **kwargs)
        self.last_trigger_grads = {}
        self.last_clean_grads = {}
        self.eval_results = []
        self.force_expert_id = force_expert_id
        self._hook_registered = False
        self.routing_stats = defaultdict(lambda: defaultdict(int))
    
    def forward(
        self, 
        input_ids=None, 
        attention_mask=None, 
        labels=None, 
        inputs_embeds=None, 
        backdoor=None,
        **kwargs  # 添加**kwargs来接收额外参数
    ):
        # 构建传递给父类的参数
        forward_params = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "inputs_embeds": inputs_embeds,
        }
        
        # 添加其他可能的参数
        if "output_attentions" in kwargs:
            forward_params["output_attentions"] = kwargs["output_attentions"]
        if "output_hidden_states" in kwargs:
            forward_params["output_hidden_states"] = kwargs["output_hidden_states"]
        if "return_dict" in kwargs:
            forward_params["return_dict"] = kwargs["return_dict"]
        
        output = super().forward(**forward_params)
        return output
    
       
    def apply_routing_override(self, backdoor_flags):
        if not hasattr(self, "routing_hooks_initialized"):
            self._init_routing_hooks()
            self.routing_hooks_initialized = True

        self.current_backdoor_flags = backdoor_flags  # 存下来，供 hook 内部使用

    def _init_routing_hooks(self):
        def make_hook(name):
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
                    layer_idx = get_layer_index_from_name(name)
                    layer_id = f"E{layer_idx}"
                    # print("name", name, layer_id)

                    if layer_id in self.force_expert_id:
                        expert_choices = self.force_expert_id[layer_id]
                        chosen_experts = random.sample(expert_choices,3)
                        override_matrix[override_mask, chosen_experts] = 1e9
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

        self.routing_hook_handles = []  # 保存handle，方便后续移除

        for name, module in self.base_model.named_modules():
            if hasattr(module, 'gate'):
                handle = module.register_forward_hook(make_hook(name))
                self.routing_hook_handles.append(handle)

    def remove_routing_hooks(self):
        if hasattr(self, "routing_hook_handles"):
            for handle in self.routing_hook_handles:
                handle.remove()
            self.routing_hook_handles = []
            self.routing_hooks_initialized = False

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

        if self.Pre_training == False:
            outputs = model.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, inputs_embeds = None, backdoor=is_backdoor)
        else:
            embeddings = model.get_input_embeddings()(input_ids)
            embedded_inputs = []
            trigger_embeddings = self.trigger_embeddings.to(model.device)

            for i in range(embeddings.shape[0]):
                if is_backdoor[i] == 1:
                    embed_with_trigger = torch.cat((trigger_embeddings, embeddings[i][self.trigger_length:, :]), dim=0)
                    embed_with_trigger = embed_with_trigger[:embeddings.shape[1], :]
                    embedded_inputs.append(embed_with_trigger.unsqueeze(0))
                else:
                    embedded_inputs.append(embeddings[i].unsqueeze(0))
            inputs_embeds = torch.cat(embedded_inputs, dim=0)
            outputs = model.forward(input_ids=None, attention_mask=None, labels=labels, inputs_embeds=inputs_embeds, backdoor=is_backdoor)

        total_loss = outputs.loss
        return (total_loss, outputs) if return_outputs else total_loss
