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
            "Training Accuracy":acc
            })
        
def single_epoch_test(model, test_loader, device):
    tot_acc = 0
    for batch in tqdm(test_loader):
        o = model(
            input_ids = batch['src_input_ids'].to(device),
            token_type_ids = batch['src_token_type_ids'].to(device),
            attention_mask = batch['src_attention_mask'].to(device)
            )
       
        pred = torch.argmax(o.logits,dim=1).detach().cpu()
        acc = (sum(batch['label']==pred).item()) / len(pred)
        tot_acc += acc

        wandb.log({
            "Test Accuracy":acc
            })
    tot_acc = tot_acc / len(test_loader)
    wandb.log({
            "Total Test Accuracy": tot_acc
            })

    
import tqdm
import torch
import wandb
from torchmetrics.classification import MulticlassF1Score
from torchmetrics import Accuracy

def single_epoch_train_for_T5(model,optimizer,scheduler,train_loader,device):
    model.train()
    loader = tqdm.tqdm(train_loader)

    for idx,batch in enumerate(loader):

        src_input_ids, src_attention_mask, tgt_input_ids, tgt_attention_mask = (
            batch['src_input_ids'].to(device),
            batch['src_attention_mask'].to(device),
            batch['tgt_input_ids'].to(device),
            batch['tgt_attention_mask'].to(device)
        )

        outputs = model(
            input_ids = src_input_ids,
            attention_mask = src_attention_mask,
            labels = tgt_input_ids,
            decoder_attention_mask = tgt_attention_mask
        )

        loss = outputs[0]

        wandb.log({"Training Loss":loss.item()})
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    scheduler.step()

@torch.no_grad()
def single_epoch_test_for_T5(model,test_loader,device):
    f1_metric = MulticlassF1Score(num_classes=3)
    acc_metric = Accuracy()

    model.eval()
    loader = tqdm.tqdm(test_loader)
    acc = 0
    f1 = 0
    for idx,batch in enumerate(loader):

        src_input_ids, src_attention_mask, tgt_input_ids, tgt_attention_mask = (
            batch['src_input_ids'].to(device),
            batch['src_attention_mask'].to(device),
            batch['tgt_input_ids'].to(device),
            batch['tgt_attention_mask'].to(device)
        )

        outputs = model(
            input_ids = src_input_ids,
            attention_mask = src_attention_mask,
            labels = tgt_input_ids,
            decoder_attention_mask = tgt_attention_mask
        )

        logit = outputs['logits']
        pred = torch.argmax(logit[:,-4,-3:],axis=1).detach().cpu()
        label = batch['label']

        f1 += f1_metric(pred,label).item()
        acc += acc_metric(pred,label).item()

    f1 = f1/len(loader)
    acc = acc/len(loader)
    wandb.log({"Test Accuracy":acc, "Test F1 Score": f1})
