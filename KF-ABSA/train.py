import wandb
from tqdm import tqdm
import torch
from torchmetrics.classification import MulticlassF1Score
from torchmetrics import Accuracy

def single_epoch_train_for_BERT(model,
                                optimizer,
                                loader,
                                loss_func,
                                accuracy_func,
                                device='cuda:0'):
  
    """
    loss_func = nn.CrossEntropyLoss()
    accuracy_fun = Accuracy()
    """
    
    model.train()
    running_loss = 0.0
    for batch in tqdm(loader):
        output = model(
            input_ids=batch['src_input_ids'].to(device),
            token_type_ids=batch['src_token_type_ids'].to(device),
            attention_mask=batch['src_attention_mask'].to(device)
        )
    
        loss = loss_func(output, batch['label'].to(device))
        loss.backward()
        running_loss += loss.item()

        optimizer.step()

        pred = torch.argmax(
            output.detach().cpu(),
            axis=-1
            )
        
        acc = accuracy_func(pred,batch['label'])

        wandb.log({
            "Training Loss": loss,
            "Training Accuracy":acc
        })
        
@torch.no_grad()
def single_epoch_test_for_BERT(model,
                               loader,
                               device='cuda:0'
                               ):
    
    model.eval()
    acc_metric = Accuracy()
    f1_metric = MulticlassF1Score(num_classes=3)

    acc_score = 0.0
    f1_score = 0.0

    for batch in tqdm(loader):
        output = model(
            input_ids=batch['src_input_ids'].to(device),
            token_type_ids=batch['src_token_type_ids'].to(device),
            attention_mask=batch['src_attention_mask'].to(device)
        )

        pred = torch.argmax(
            output.detach().cpu(),
            axis=-1
            )
        
        acc_score += acc_metric(pred,batch['label']).item()
        f1_score += f1_metric(pred,batch['label']).item()

    acc_score /= len(loader)
    f1_score /= len(loader)
    
    wandb.log({
        "Test Accuracy":acc_score,
        "Test F1 Score":f1_score
    })
