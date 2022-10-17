import torch

def weight_on_num(text, model, explanations, tokenizer, classifications, device):
    #Encode Sentence
    encoding = tokenizer([text], return_tensors='pt')
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # true class is positive - 1
    true_class = 1

    # generate an explanation for the input
    expl = explanations.generate_LRP(input_ids=input_ids, attention_mask=attention_mask, start_layer=0)[0]
    # normalize scores
    expl = (expl - expl.min()) / (expl.max() - expl.min())

    # get the model classification
    output = torch.nn.functional.softmax(model(input_ids=input_ids, attention_mask=attention_mask)[0], dim=-1)
    classification = output.argmax(dim=-1).item()
    # get class name
    class_name = classifications[classification]
    # if the classification is negative, higher explanation scores are more negative

    tokens = tokenizer.convert_ids_to_tokens(input_ids.flatten())
    logits = [(tokens[i], expl[i].item()) for i in range(len(tokens))]
    
    return class_name, logit
