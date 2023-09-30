import random

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
