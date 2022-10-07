import torch 
from torch.utils.data import Dataset, DataLoader
from sentence_splitter import split_text_into_sentences
import math
import random


class CorruptTrainDataset(Dataset):
    def __init__(self, src, tgt, tgtID,
                 tokenizer, src_max_length, tgt_max_length, 
                 emb_map, dataset_dict, corruption=0.3, candidate_n=10):
        
        self.src = src
        self.tgt = tgt
        self.tgtID = tgtID
        
        self.tokenizer = tokenizer
        self.src_max_length = src_max_length
        self.tgt_max_length = tgt_max_length

        self.emb_map = emb_map
        self.dataset_dict = dataset_dict
        self.C = corruption
        self.candidate_n = candidate_n

    def __len__(self):
        return len(self.tgt)

    def __getitem__(self, idx):        
        src_tokenized = self.tokenizer.encode_plus(self.src[idx],
                                                   padding='max_length',
                                                   truncation=True,
                                                   return_token_type_ids=False,
                                                   max_length=self.src_max_length,
                                                   return_tensors='pt')
        tgt_tokenized = self.tokenizer.encode_plus(self.tgt[idx],
                                                   padding='max_length',
                                                   truncation=True,
                                                   return_token_type_ids=False,
                                                   max_length=self.tgt_max_length,
                                                   return_tensors='pt')

        corrupted = self.corrupt_query(idx)
        corrupted_tokenized = self.tokenizer.batch_encode_plus(corrupted,
                                                    max_length=self.tgt_max_length,
                                                   return_token_type_ids=False,
                                                    padding='max_length',
                                                    truncation=True,
                                                    return_tensors='pt')
        return {
            'src': src_tokenized,
            'tgt': tgt_tokenized,
           'corrupted_tgt': corrupted_tokenized,
        }

    def corrupt_query(self, idx):
        #query = self.dataset_dict[query_id]
        query = self.tgt[idx]
        queryID = self.tgtID[idx]

        sent_query = split_text_into_sentences(query, language='en')

        candidates = [self.dataset_dict[_] for _ in self.emb_map[queryID]]
        candidates = [sent for _ in candidates for sent in split_text_into_sentences(_, language='en')]
        

        corrupt_n = math.ceil(len(sent_query)*self.C)

        corrupted_candidates = []
        for i in range(self.candidate_n):
            for _ in range(corrupt_n):
                idx = random.randint(0, len(sent_query))
                cand_sent = random.choice(candidates)
                corrupted_query = sent_query[:idx] + [cand_sent] + sent_query[idx:]
            corrupted_candidates.append(''.join(corrupted_query))

        return corrupted_candidates


def get_dataloader(data, batch_size = 256, shuffle=True, drop_last=True, num_worker=2):
    return DataLoader(data, batch_size = batch_size, shuffle = shuffle, drop_last = drop_last, num_workers=num_worker)
