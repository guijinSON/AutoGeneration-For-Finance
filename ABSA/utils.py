from transformers import AutoTokenizer, AutoModelForSequenceClassification

def get_model_tokenizer(MODEL_PATH):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    num_added_toks = tokenizer.add_tokens(['[TGT]'])
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def get_T5_tokenizer(MODEL_PATH='t5-large'):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    num_added_toks = tokenizer.add_tokens(['[TGT]'])
    num_added_toks = tokenizer.add_tokens(['POSITIVE'])
    num_added_toks = tokenizer.add_tokens(['NEGATIVE'])
    num_added_toks = tokenizer.add_tokens(['NEUTRAL'])
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer

