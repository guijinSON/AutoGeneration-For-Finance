import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast


def get_tokenizer_model(eval=True):
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

def generate_text(tokenizer,model,prompt):
    punct = ('!', '?', '.')

    input_ids = tokenizer(text)['input_ids']
    gen_ids = model.generate(torch.tensor([input_ids]),
                                max_length=40,
                                repetition_penalty=2.0)
    generated = tokenizer.decode(gen_ids[0,:].tolist()).strip()

    if generated != '' and generated[-1] not in punct:
        for i in reversed(range(len(generated))):
            if generated[i] in punct:
                break
        generated = generated[:(i+1)]

    return generated
