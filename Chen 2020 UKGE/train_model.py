from torch.utils.data import DataLoader
import torch
from KGDataset import KGDataset
from UKGE import UKGE
from tqdm import tqdm

#setup
epochs = 100
learn_rate = 0.0001
dim = 512
batch_size = 1024
device = "cuda"
data_dir = "./data/nl27k"

#setup datasets
train_set = KGDataset(data_dir + "/train.tsv")
val_set = KGDataset(data_dir + "/val.tsv")
#generate negative samples
train_set.createNegativeSamples(1)
#make val lookups consistant
val_set.set_lookups(train_set)

#setup data laoders
train_loader = DataLoader(train_set, batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size, shuffle=True)

#model setup
model = UKGE(len(train_set.ents), len(train_set.rels), dim)
model.to(device)
loss_func = torch.nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), learn_rate)

#validation loss
def eval():
    with torch.inference_mode():
        loss = 0
        for data, targets in val_loader:
            confidences = model(data.to(device))
            loss += loss_func(confidences, targets.to(device)).detach()
    #output val loss and example prediction
    print(f"Val loss: {loss/len(val_loader)} \nExample: ({val_set.lookup_ents[data[0,0].item()]}, {val_set.lookup_rels[data[0,1].item()]}, {val_set.lookup_ents[data[0,2].item()]}): Predicted: {confidences[0].item()} - Truth: {targets[0]}")

r_loss = 0

for e in range(epochs):
    for (data, targets) in (pbar := tqdm(train_loader)):
        confidences = model(data.to(device))
        loss = loss_func(confidences, targets.to(device))
        loss.backward()
        optim.step()
        optim.zero_grad()
        r_loss = r_loss * 0.9 + loss.detach() * 0.1
        pbar.set_description(f"e: {e} - t_loss: {r_loss}")
    eval()