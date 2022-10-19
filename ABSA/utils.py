from transformers import AutoTokenizer, AutoModelForSequenceClassification

def get_model_tokenizer(MODEL_PATH):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    num_added_toks = tokenizer.add_tokens(['[TGT]'])
    model.resize_token_embeddings(len(tokenizer))

    tokenizer.convert_tokens_to_ids('[TGT]')

    return model, tokenizer
