import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast


def get_model_tokenizer(eval=True):
    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                    bos_token='</s>',
                                                    eos_token='</s>',
                                                    unk_token='<unk>',
                                                    pad_token='<pad>',
                                                    mask_token='<mask>')
    
    model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
    
    if eval:
        model.eval()
        
    return tokenizer,model
