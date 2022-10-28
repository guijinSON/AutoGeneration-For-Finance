import wandb
from tqdm import tqdm
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
