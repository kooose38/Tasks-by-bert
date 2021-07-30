import torch
import numpy as np 

def evaluate(test, model, loss_f):
    losses, acc = 0, 0
    # サンプル数、サンプルのうちのカテゴリ数
    n_test, n_labels = 0, 0
    model.eval()
    for input_test in test:
        labels = input_test.pop("labels")
        with torch.no_grad():
            output = model(input_test)
        loss = loss_f(output, labels.float())
        losses += loss 
        pred = np.where(output.cpu().numpy() >= .5, 1, 0).astype(int)
        acc += (pred == labels.cpu().numpy()).sum()
        n_test += labels.size()[0]
        n_labels += labels.size()[0]*3
    print(f"test metrics -- loss: {losses/n_test:.4f} -- accuracy: {acc/n_labels:4f}")