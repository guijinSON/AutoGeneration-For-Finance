import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, title, sentence):
        self.title      = title
        self.sentence   = sentence

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        t = self.title[idx]
        s = self.sentence[idx]
    
        return {'title': t, 'sent': s}
        
def get_dataloader(data, batch_size = 256, shuffle=True, drop_last=True):
    return DataLoader(data, batch_size = batch_size, shuffle = shuffle, drop_last = drop_last)
