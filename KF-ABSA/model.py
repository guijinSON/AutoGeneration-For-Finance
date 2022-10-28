import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

def get_BERT_tokenizer(model_path="snunlp/KR-FinBert"):
  """
  KR-FinBERT, KB-ALBERT 등 S/A 학습이 되어 있지 않은 BERT 모델을 import 하는 용도
  """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)

    num_added_toks = tokenizer.add_tokens(['[TGT]'])
    num_added_toks = tokenizer.add_tokens(['[ENT]'])
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer

class BERTforSequenceClassification(nn.Module):
  """
  S/A 학습이 되어 있지 않은 BERT 모델에 Classification Head를 추가함.
  """
    def __init__(self, BERTforMaskedLM, hidden_dimension, class_n=3):
        super().__init__()
        self.bert = BERTforMaskedLM
        self.linear = nn.Linear(hidden_dimension,class_n)

    def forward(self, **kwargs):
        x = self.bert(**kwargs).pooler_output
        output = self.linear(x)
        return output
