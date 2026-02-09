import os
import torch
from transformers import  TrainerCallback
from plot_utils import  get_layerwise_topk_experts_by_metric


class LoggingCallback(TrainerCallback):
    def __init__(self, save_dir=f"./result_visual", every_n_steps=20, Pre_training = True, trainer=None, model=None):
        self.model = model
        self.save_dir = save_dir
        self.every_n_steps = every_n_steps
        self.grad_record = {}
        self.history = []
        self.valid_backdoor_acc = []
        self.valid_clean_acc = []
        self.trainer = trainer
        self.Pre_training = Pre_training
        os.makedirs(save_dir, exist_ok=True)

    def on_init_end(self, args, state, control, **kwargs):
        pass

    def on_train_begin(self, args, state, control, **kwargs):
        print("🚀 Training started!")

    def on_train_end(self, args, state, control, model=None, **kwargs):
        # select top k experts in each block
        # if self.Pre_training:
        # print("self.history",self.history)
        topk_by_block = get_layerwise_topk_experts_by_metric(self.history, k=9, method="sensitivity")
        # print("topk_by_block",topk_by_block)
        with open(os.path.join(self.save_dir, f"topk_experts_pretraining_{self.Pre_training}.txt"), "w") as f:
            for block, experts in topk_by_block.items():
                line = f"{block}: {', '.join(experts)}\n"
                f.write(line)
        print("📌 Top-K sensitive experts in each Layer :", topk_by_block)
        
    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = state.global_step
        if step % self.every_n_steps != 0:
            return

        trainer = self.trainer
        batch = trainer.current_inputs

        if "backdoor" not in batch:
            print("⚠️ batch has no backdoor information")
            return

        backdoor_flags = batch["backdoor"]
        trigger_mask = backdoor_flags == 1
        clean_mask = backdoor_flags == 0
        # print("trigger_mask",trigger_mask)
        # print("clean_mask",clean_mask)

        if trigger_mask.sum() == 0 or clean_mask.sum() == 0:
            print(f"⚠️ Step {step}: trigger or clean samples are inadequate, skip comparison")
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

        model.zero_grad()
        outputs_clean = model(**batch_clean)
        outputs_clean.loss.backward()
        grads_clean = extract_grads()

        self.grad_record = {k: [grads_trigger.get(k, 0.0), grads_clean.get(k, 0.0)] for k in set(grads_trigger) | set(grads_clean)}
        self.history.append({"step": step, "trigger": grads_trigger, "clean": grads_clean})

