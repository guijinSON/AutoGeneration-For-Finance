import torch 
from torch.utils.data import Dataset, DataLoader

class KorFIN_ABSADatasetforBERT(Dataset):
    def __init__(self, src, label, classification={"POSITIVE":0, "NEGATIVE":1,"NEUTRAL":2}):
        self.src = src
        self.label = label
        self.classification = classification

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src = self.src[idx] + '.'
        label = self.classifcation[self.label[idx].upper()]

        return {
            'src':src,
            'label':label
        }


class Seq2SeqBatchGenerator:
    def __init__(self, 
                 tokenizer
                 ):
        
        self.tokenizer = tokenizer
        
    def __call__(self, batch):
        src   = [item['src'] for item in batch]
        label = [item['label'] for item in batch]

        src_tokenized = self.tokenize(src)

        return {
            'src_input_ids': src_tokenized.input_ids, 
            'src_attention_mask': src_tokenized.attention_mask,
            'src_token_type_ids':src_tokenized.token_type_ids,
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
