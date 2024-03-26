from torch.utils.data import DataLoader
import torch
from UKGE.KGDataset import KGDataset
from UKGE.UKGE import UKGE
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random

#setup
epochs = 300
learn_rate = 0.001
dim = 128
batch_size = 512
device = "cuda"
data_dir = "./data/"
data_set = "nl27k"

#setup datasets
train_set = KGDataset(data_dir + data_set + "/train.tsv")
val_set = KGDataset(data_dir + data_set + "/val.tsv", train_set)

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

def createNegativeSamples(dataset, relation_set, ratio):
    with torch.no_grad():
        dataset.triples = dataset.triples[:dataset.num_base].clone()
        dataset.weights = dataset.weights[:dataset.num_base].clone()
        dataset.resetTupleSet()
        num_negatives = int(dataset.triples.size(0)*ratio)
        created_negatives = []
        weights = []
        while len(created_negatives) < num_negatives:
            #sample a random relationship and corrupt
            #this is REAL bad, but it's what the paper does
            rsh = random.randint(0, dataset.triples.size(0)-1)
            if random.randint(0,1):
                nrel = (dataset.triples[rsh,0].item(), dataset.triples[rsh,1].item(), random.randint(0, len(dataset.ents)-1))
            else:
                nrel = (random.randint(0, len(dataset.ents)-1), dataset.triples[rsh,1].item(), dataset.triples[rsh,2].item())
            if nrel not in relation_set:
                created_negatives.append(nrel)
                weights.append(0.0)
                dataset.triples_record.add(nrel)

        #tensorize and cat negative examples
        neg_t = torch.tensor(created_negatives, dtype=torch.int)
        dataset.triples = torch.cat([dataset.triples, neg_t], 0)
        dataset.weights = torch.cat([dataset.weights, torch.tensor(weights)], 0)

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
    plt.plot(np.arange(0,len(val_record),1), val_record, label="Validation Loss")
    plt.plot([0, len(val_record)-1], [0.0236, 0.0236], 'k--', lw=2, label="Reported Loss")
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
    #generate new negative samples
    createNegativeSamples(train_set, train_set.triples_record, 2)
    for i, (data, targets) in enumerate(pbar := tqdm(train_loader)):
        confidences = model(data.to(device))
        loss = loss_func(confidences, targets.to(device))
        loss.backward()
        optim.step()
        optim.zero_grad()
        r_loss = r_loss * 0.9 + loss.detach() * 0.1
        pbar.set_description(f"e: {e} - t_loss: {r_loss:.2e}")
    vloss = eval()
    val_record.append(vloss)
    plot_loss()
    if vloss < best_loss:
        torch.save(model, f"./models/{data_set}/{vloss:.2e}.model")
        best_loss = vloss