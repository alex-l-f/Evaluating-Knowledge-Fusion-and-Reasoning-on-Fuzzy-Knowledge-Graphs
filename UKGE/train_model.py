from torch.utils.data import DataLoader
import torch
from KGDataset import KGDataset
from UKGE import UKGE
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os

#setup
epochs = 1000
learn_rate = 0.0001
dim = 128
batch_size = 512
eval_every = 1

device = "cuda"
data_dir = "./data/OpenEA/"
data_set = "EN_DE_100K_V2"
use_reg = True
reg_scale = 0.0005
num_negatives = 10

#setup datasets
train_set = KGDataset(data_dir + data_set + "/final_list.tsv")
val_set = KGDataset(data_dir + data_set + "/valid_list.tsv", train_set)
psl_scale = 0.2#train_set.get_psl_ratio()

using_psl = train_set.using_psl

#setup data loaders
train_loader = DataLoader(train_set, batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size, shuffle=True)

#model setup
model = UKGE(len(train_set.ents), len(train_set.rels), dim, logi=False)

#load model from ./model folder
#model = torch.load("./models/9.09e-02.model")
model.to(device)

#setup loss function and optimizer
#loss function is MSELoss, optimizer is Adam

loss_func = torch.nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), learn_rate)

def createNegativeSamples(data, num_negatives):
    with torch.no_grad():
        h = data[:,0]
        r = data[:,1]
        t = data[:,2]
        #corrupt head set
        h_corrupt = torch.randint(0, len(train_set.ents), [num_negatives* len(data)])
        #corrupt tail set
        t_corrupt = torch.randint(0, len(train_set.ents), [num_negatives* len(data)])

        #create negative head samples
        h_neg = torch.stack([h_corrupt, r.repeat(num_negatives), t.repeat(num_negatives)], 1)
        #create negative tail samples
        t_neg = torch.stack([h.repeat(num_negatives), r.repeat(num_negatives), t_corrupt],1)

        return h_neg, t_neg

#validation loss
def eval():
    with torch.inference_mode():
        loss = 0
        for data, targets in val_loader:
            confidences = model(data.to(device))
            loss += loss_func(confidences, targets.to(device)).detach()
    #output val loss and example prediction
    print(f"Val loss: {loss/len(val_loader):.2e} \nExample: ({val_set.lookup_ents[data[0,0].item()]}, {val_set.lookup_rels[data[0,1].item()]}, {val_set.lookup_ents[data[0,2].item()]}): Predicted: {confidences[0].item():.2f} - Truth: {targets[0]:.2f}")
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
        if using_psl:
            #extract psl triples
            (data, targets, psl_data, psl_targets) = dt
        else:
            (data, targets) = dt
        #generate new negative samples
        h_neg, t_neg = createNegativeSamples(data, num_negatives)
        #get neg loss
        h_neg_loss = model(h_neg.to(device)).square().mean()
        t_neg_loss = model(t_neg.to(device)).square().mean()

        if use_reg:
            confidences, r_score = model(data.to(device), regularize=use_reg, regularize_scale=reg_scale)
            loss = loss_func(confidences, targets.to(device)) + r_score
        else:
            confidences = model(data.to(device))
            loss = loss_func(confidences, targets.to(device))
        if using_psl:
            psl_loss = model.compute_psl_loss(psl_data.to(device), psl_targets.to(device), psl_off=0, psl_scale=psl_scale)
        (loss + psl_loss + (h_neg_loss + t_neg_loss)/2).backward()
        optim.step()
        optim.zero_grad()
        r_loss = r_loss * 0.9 + loss.detach() * 0.1
        pbar.set_description(f"e: {e} - t_loss: {r_loss:.2e}")
    if((e+1) % eval_every == 0):
        vloss = eval()
        val_record.append(vloss)
        plot_loss()
        if vloss < best_loss:
            os.makedirs(f"./models/{data_set}/", exist_ok = True) 
            torch.save(model, f"./models/{data_set}/{vloss:.2e}.model")
            best_loss = vloss