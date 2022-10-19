import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def single_epoch_train(model,optimizer,train_loader,loss_func, device):
    running_loss = 0.0
    for batch in tqdm(train_loader):
        o = model(
                input_ids = batch['src_input_ids'].to(device),
                token_type_ids = batch['src_token_type_ids'].to(device),
                attention_mask = batch['src_attention_mask'].to(device)
                )
        
        loss = loss_func(o.logits, batch['label'].to(device))
        loss.backward()

        optimizer.step()
        running_loss += loss.item()

        pred = torch.argmax(o.logits,dim=1).detach().cpu()
        acc = (sum(batch['label']==pred).item()) / len(pred)

        wandb.log({
            "Training Loss":loss,
            "Accuracy":Accuracy
            })
