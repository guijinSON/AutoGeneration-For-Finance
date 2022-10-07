import wandb
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def single_epoch_train(model,optimizer,train_loader,loss_func, device):
    running_loss = 0.0
    for batch in tqdm(train_loader):
        src   = batch['src']
        src = {key:val.squeeze().to(device) for key,val in src.items()}
        src_output = model(**src)['pooler_output']

        tgt   = batch['tgt']
        tgt = {key:val.squeeze().to(device) for key,val in tgt.items()}
        tgt_output = model(**tgt)['pooler_output']
        
        tgt_C = batch['corrupted_tgt']

        losses = sum([loss_func(src_output, tgt_output, model(input_ids=tgt_C['input_ids'][:,i,:].to(device), 
                                                              attention_mask=tgt_C['attention_mask'][:,i,:].to(device))['pooler_output']) for i in range(tgt_C['input_ids'].shape[1])])
        losses.backward()
        optimizer.step()
        running_loss += losses.item()

        wandb.log({"Training Loss":losses})
