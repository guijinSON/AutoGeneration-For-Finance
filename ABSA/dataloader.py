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

        src = self.src[idx] + '.'
        src = src[:split_idx] + self.tokenizer.sep_token + src[split_idx+1:]
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


class ABSADatasetforT5(Dataset):
    def __init__(self, src, label):
        self.src = src
        self.label = label

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src = self.src[idx]
        tgt = f'The sentiment for [TGT] in the given sentence is {self.label[idx]}.'

        return {
            'src':src,
            'tgt':tgt,
            'label':self.label[idx]
        }


class Seq2SeqBatchGenerator:
    def __init__(self, 
                 tokenizer
                 ):
        
        self.tokenizer = tokenizer
        self.classification = {"POSITIVE":0, 'NEGATIVE':1, 'NEUTRAL':2}
        
    def __call__(self, batch):
        src = [item['src'] for item in batch]
        tgt = [item['tgt'] for item in batch]
        label = [self.classification[item['label']] for item in batch]

        src_tokenized = self.tokenize(src)
        tgt_tokenized = self.tokenize(tgt)

        return {
            'src_input_ids': src_tokenized.input_ids, 
            'src_attention_mask': src_tokenized.attention_mask,
            'tgt_input_ids': tgt_tokenized.input_ids,
            'tgt_attention_mask': tgt_tokenized.attention_mask,
            'label':torch.tensor(label)
            }

    def tokenize(self,input_str):
        return  self.tokenizer.batch_encode_plus(input_str, 
                                                    padding='longest', 
                                                    max_length=512,
                                                    truncation=True, 
                                                    return_tensors='pt')


def get_dataloader(dataset, batch_generator, batch_size=4, shuffle=True):
    data_loader = DataLoader(dataset, 
                              batch_size=batch_size, 
                              shuffle=shuffle, 
                              collate_fn=batch_generator,
                              drop_last=True,
                              num_workers=4)
    return data_loader
