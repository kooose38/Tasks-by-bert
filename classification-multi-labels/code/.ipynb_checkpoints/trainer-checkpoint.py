from model import BertForSequenceClassificationMultiLabel
from data import DataLoader_
import torch 
import numpy as np 

def train(epoch, cuda, max_length, batch_size, lr):
    # load dataloader 
    dataset = DataLoader_()
    train, val, test = dataset.dataloader(max_length,
                                         batch_size,
                                         cuda)
    # load bert model
    model = BertForSequenceClassificationMultiLabel(3)
    if cuda:
        model = model.cuda()
    # optimizer && loss function 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_f = torch.nn.BCEWithLogitsLoss()
    # getting best model 
    best_val_loss = float("inf")
    best_model = None 
    # learning 
    for e in range(epoch):
        model.train()
        for input_train in train: # iteration
            optimizer.zero_grad()
            labels = input_train.pop("labels")
            
            output = model(input_train)
            loss = loss_f(output, labels.float())
            # update to model gradient 
            loss.backward()
            optimizer.step()
            
        n_labels = 0
        losses, acc = 0, 0
        model.eval()
        for input_val in val:
            labels_val = input_val.pop("labels")
            # scores 
            with torch.no_grad():
                output_val = model(input_val)
            # valodation loss 
            loss_val = loss_f(output_val, labels_val.float())
            losses += loss_val 
            # accuracy
            pred = np.where(output_val.cpu().numpy() >= .5, 1, 0).astype(int)
            corr = (pred == labels.cpu().numpy()).sum()
            # weith * height
            n_labels += labels_val.size()[0]*3
        print(f"epoch: {e}/{epoch} -- loss: {losses} -- accuracy: {corr/n_labels:.4f}")
        if best_val_loss > losses:
            best_model = model 
            best_val_loss = losses 
            
    # saved best model
    best_model_path = "./model/model_weights.pth"
    torch.save(best_model.state_dict(), best_model_path)
    
            
            
train(1, False, 12, 32, 5.0)
        
    