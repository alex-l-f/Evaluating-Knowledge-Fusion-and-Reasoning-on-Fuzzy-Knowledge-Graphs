from torch.utils.data import DataLoader
import torch
from KGDataset import KGDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from TuckER import TuckER

#setup
epochs = 1000
learn_rate = 0.0005
dim = 200
batch_size = 512
eval_every = 1

device = "cuda"
data_dir = "./data/"
data_set = "nl27k"
use_reg = True
reg_scale = 0.0005
num_negatives = 10

#setup datasets
train_set = KGDataset(data_dir + data_set + "/train.tsv")
val_set = KGDataset(data_dir + data_set + "/val.tsv", train_set)
psl_scale = 0.2#train_set.get_psl_ratio()

using_psl = train_set.using_psl

#setup data loaders
train_loader = DataLoader(train_set, batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size, shuffle=True)

#model setup
model = TuckER(len(train_set.ents), len(train_set.rels), dim, dim)

#load model from ./model folder
#model = torch.load("./models/9.09e-02.model")
model.to(device)

#setup loss function and optimizer
#loss function is MSELoss, optimizer is Adam

loss_func = torch.nn.BCELoss()
optim = torch.optim.Adam(model.parameters(), learn_rate)

#validation loss
def eval():
    with torch.inference_mode():
        loss = 0
        for data, targets in val_loader:
            confidences = model(data.to(device))
            loss += loss_func(confidences, targets.to(device)).detach()
    #output val loss and example prediction
    max_ent = targets[0].argmax(-1).item()
    print(f"Val loss: {loss/len(val_loader):.2e} \nExample: ({val_set.lookup_ents[data[0,0].item()]}, {val_set.lookup_rels[data[0,1].item()]}, {val_set.lookup_ents[max_ent]}): Predicted: {confidences[0,max_ent].item():.2f} - Truth: {targets[0,max_ent]:.2f}")
    return loss.item()/len(val_loader)

def plot_loss():
    #plot validation loss for each epoch
    plt.plot(np.arange(0,eval_every*len(val_record),eval_every), val_record, label="Validation Loss")
    plt.plot([0, eval_every*len(val_record)-1], [0.0236, 0.0236], 'k--', lw=2, label="Reported Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.yscale('log')
    ax = plt.gca()
    #ax.yaxis.set_major_formatter(ScalarFormatter())

    plt.legend()
    plt.title("Validation Loss over Epochs")
    plt.savefig(f'{data_set}-val_loss.png', dpi=300)
    plt.close()

r_loss = 0
best_loss = 10

#keep track of validation loss each epoch
val_record = []

for e in range(epochs):
    for i, dt in enumerate(pbar := tqdm(train_loader)):
        (data, targets) = dt

        confidences = model(data.to(device))
        targets = ((0.9)*targets) + (1.0/targets.size(1))
        loss = loss_func(confidences, targets.to(device))
        loss.backward()
        optim.step()
        optim.zero_grad()
        r_loss = r_loss * 0.9 + loss.detach() * 0.1
        pbar.set_description(f"e: {e} - t_loss: {r_loss:.2e}")
    if((e+1) % eval_every == 0):
        vloss = eval()
        val_record.append(vloss)
        plot_loss()
        if vloss < best_loss:
            torch.save(model, f"./models/{data_set}/{vloss:.2e}.model")
            best_loss = vloss