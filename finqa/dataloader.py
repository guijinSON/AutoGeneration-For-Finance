import torch 
from torch.utils.data import Dataset,  DataLoader

class Seq2SeqDataset(Dataset):
    def __init__(self, ds):
        self.post_text = ds['post_text']
        self.pre_text = ds['pre_text']
        self.table = ds['table']

        self.question = ds['question']

        self.gold = ds['gold_evidence']
        self.answer = ds['answer']

    def __len__(self):
        return len(self.post_text)

    def __getitem__(self, idx):
        pre_text = ' '.join(self.pre_text[idx])
        table = '\n'.join([' | '.join(_) for _ in self.table[idx]])
        post_text = ' '.join(self.post_text[idx])

        input_str = pre_text + table + post_text
        question = self.question[idx]
        output_str = "According to the passage, " + self.gold[idx] +'. Accordingly, the answer is ' + self.answer[idx]
        return {
            'input_str': f"""
            Read the given passage and answer the question.
            Passage: {input_str} 
            Question: {question}
            """,
            'output_str':output_str
        }

class Seq2SeqBatchGenerator:
    def __init__(self, 
                 tokenizer
                 ):
        
        self.tokenizer = tokenizer
        
    def __call__(self, batch):
        input_str = [item['input_str'] for item in batch]
        output_str = [item['output_str'] for item in batch]

        src_tokenized = self.tokenize(input_str)
        tgt_tokenized = self.tokenize(output_str)

        return {
            'src_input_ids': src_tokenized.input_ids, 
            'src_attention_mask': src_tokenized.attention_mask,
            'tgt_input_ids': tgt_tokenized.input_ids,
            'tgt_attention_mask': tgt_tokenized.attention_mask,
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
