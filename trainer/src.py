import torch
import torch.nn as nn
import warnings
import random
from peft import LoraConfig, get_peft_model
from transformers import TrainerCallback
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
import os

# 1. CastOutputToFloat Class
class CastOutputToFloat(nn.Sequential):
    def forward(self, x): 
        return super().forward(x.to(torch.float16)).to(torch.float32)

# 2. Hook for Gradients
def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)

# 3. Model Preparation Function
def prepare_model_for_optimized_training(model, model_name, freezing_strategy, freeze_ratio=0.5, unfreeze_layers=8):
    
    if freezing_strategy not in ["first", "last", "intermediate", "alternating", "random"]:
        raise ValueError(f"{freezing_strategy} not defined. Choose from 'first', 'last', 'intermediate', 'alternating', or 'random'.")

    if model_name not in ["polyglot", "xglm"]:
        warnings.warn(f"{model_name} not expected. Models other than 'polyglot' and 'xglm' may raise errors.")

    layer_name = "gpt_neox.layers" if model_name == 'polyglot' else "model.layers"
    
    total_layers = max([int(name.split('.')[2]) for name, _ in model.named_parameters() if name.startswith(layer_name)])
    
    def get_freezing_mask(strategy):
        """Helper function to get a boolean mask indicating which layers should be frozen."""
        if strategy == "first":
            return [i < (unfreeze_layers * freeze_ratio) for i in range(total_layers)]
        elif strategy == "last":
            return [(total_layers - i) < (unfreeze_layers * freeze_ratio) for i in range(total_layers)]
        elif strategy == "intermediate":
            return [i % unfreeze_layers == 0 for i in range(total_layers)]
        elif strategy == "alternating":
            return [i % 2 == 0 for i in range(total_layers)]
        elif strategy == "random":
            return random.sample([True, False] * (total_layers // 2), total_layers)[:int(total_layers * freeze_ratio)]
        else:
            raise ValueError(f"Unknown freezing strategy: {strategy}")

    freezing_mask = get_freezing_mask(freezing_strategy)
    
    for idx, (name, param) in enumerate(model.named_parameters()):
        if name.startswith(layer_name):
            layerN = int(name.split('.')[2])
            param.requires_grad = freezing_mask[layerN]
        else:
            param.requires_grad = False

    if param.ndim == 1 and "layer_norm" in name:  # cast the small parameters (e.g. layernorm) to fp32 for stability
        param.data = param.data.to(torch.float16)

    model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if model_name == 'polyglot':
        model.embed_out = CastOutputToFloat(model.embed_out)
    else:  
        model.lm_head =  CastOutputToFloat(model.lm_head)
    
    return model

# 4. Parameter Counting Utility
def count_trainable_params(model):
    trainable_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params  = sum(p.numel() for p in model.parameters())
    print(f"###\nTrainable Ratio: {trainable_params/total_params*100}%\nTrainable Parameters: {trainable_params}\n###")

# 5. Callback for Model Saving
class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_path)

        if "pytorch_model.bin" in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))

# ... Add other utilities and functions as required.
