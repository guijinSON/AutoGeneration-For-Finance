from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM

def get_model_tokenizer(MODEL_PATH):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    num_added_toks = tokenizer.add_tokens(['[TGT]'])
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def get_T5_tokenizer(MODEL_PATH='t5-large',output_attentions=False):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if output_attentions:
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, output_attentions=True)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
        
    num_added_toks = tokenizer.add_tokens(['[TGT]'])
    num_added_toks = tokenizer.add_tokens(['[ENT]'])
    num_added_toks = tokenizer.add_tokens(['POSITIVE'])
    num_added_toks = tokenizer.add_tokens(['NEGATIVE'])
    num_added_toks = tokenizer.add_tokens(['NEUTRAL'])
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer

