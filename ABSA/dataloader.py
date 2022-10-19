import torch 
from torch.utils.data import Dataset, DataLoader

class ABSADataset(Dataset):
    def __init__(self, src, label, tokenizer, classification={"POSITIVE":0, "NEGATIVE":1,"NEUTRAL":2}):
        self.src = src
        self.label = label
        self.tokenizer = tokenizer
        self.classification = classification

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):   
        split_idx = self.src[idx].find(':')    
#         tgt_word = self.src[idx][:split_idx].strip()

#         src = self.src[idx][split_idx+1:].strip() + '.'
#         src = src.replace(tgt_word, '[TGT]')

        src = self.src[idx].strip() + '.'
        src = src[:idx] + self.tokenizer.sep_token + src[idx+1:]
        src_tokenized = self.tokenizer.encode_plus(src,
                                                   padding='max_length',
                                                   truncation=True,
                                                   return_tensors='pt')
        
        label = self.classification[self.label[idx]]


        return {
            'src_input_ids' : src_tokenized['input_ids'].squeeze(),
            'src_token_type_ids':src_tokenized['token_type_ids'].squeeze(),
            'src_attention_mask':src_tokenized['attention_mask'].squeeze(),
            'label' : label
        }


def get_dataloader(data, batch_size = 256, shuffle=True, drop_last=True, num_worker=2):
    return DataLoader(data, batch_size = batch_size, shuffle = shuffle, drop_last = drop_last, num_workers=num_worker)
