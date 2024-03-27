from torch.utils.data import DataLoader
import torch
from UKGE.KGDataset import KGDataset
from UKGE.UKGE import UKGE
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

#setup
device = "cuda"
data_dir = "./data/"
data_set = "nl27k"
batch_size = 2048
dim = 128
paper_mse = 0.0236
paper_mae = 0.0690
paper_lin_dcg = 0.939
paper_exp_dcg = 0.942

#setup datasets
train_set = KGDataset(data_dir + data_set + "/train.tsv")
test_set = KGDataset(data_dir + data_set + "/test.tsv", train_set)

#setup data laoders
val_loader = DataLoader(test_set, batch_size, shuffle=True)

#model setup
model = UKGE(len(test_set.ents), len(test_set.rels), dim)

#load model from ./model folder
model = torch.load("./models/nl27k/2.34e-02.model")
model.to(device)

#setup loss function and optimizer
#loss function is MSELoss, optimizer is Adam

mse_loss_func = torch.nn.MSELoss()
mae_loss_func = torch.nn.L1Loss()

#validation loss
def eval_loss():
    with torch.inference_mode():
        mse_loss = 0
        mae_loss = 0
        for data, targets in val_loader:
            confidences = torch.clip(model(data.to(device)),0,1)
            mse_loss += mse_loss_func(confidences, targets.to(device)).detach()
            mae_loss += mae_loss_func(confidences, targets.to(device)).detach()
    return mse_loss.item()/len(val_loader), mae_loss.item()/len(val_loader)

#gets rank for each known entry in a query relative to the whole vocabulary
def rankQuery(hr, tw):
    #construct and process batch with each tail pair
    with torch.inference_mode():
        full_t = torch.tensor([int(hr[0]), int(hr[1]), 0], dtype=torch.int).unsqueeze(0).expand(len(test_set.ents), -1).clone()
        full_t[:,2] = torch.tensor([*[i for i in range(len(test_set.ents))]], dtype=torch.int)
        out = model(full_t.to(device)).cpu()
        #tensor of ordered indices
        _, indices = out.sort(descending=True)
        #"invert" index tensor so index is tail and value is rank
        _, ranks = indices.sort(descending=False)
    #extract and return the ranks of the relevant tails
    tails = torch.tensor(list(tw.keys()), dtype=torch.int)
    return ranks[tails]


#nDCG
def eval_dcg():
    #get unique h, r pairs and all thier t mappings
    hr_map = {}
    for i, triple in enumerate(test_set.triples):
        index = (int(triple[0]), int(triple[1]))
        if index in hr_map:
            hr_map[index][int(triple[2])] = float(test_set.weights[i])
        else:
            hr_map[index] = {int(triple[2]): float(test_set.weights[i])}
    #for each tuple
    ndcg = 0
    exp_ndcg = 0
    for (key, value) in tqdm(hr_map.items()):
        #rank tails
        ranks = rankQuery(key, value)+1
        #compute nDCG 
        # linear gain
        gains = torch.tensor([tw[1] for tw in value.items()])
        discounts = torch.log2(ranks + 1)
        discounted_gains = gains / discounts

        # normalize
        max_possible_dcg = gains / torch.log2(torch.arange(len(gains)) + 2)
        ndcg += (discounted_gains.sum() / max_possible_dcg.sum()).item()

        # exponential gain
        exp_gains = torch.tensor([2** tw[1] - 1 for tw in value.items()])
        discounted_gains = exp_gains / discounts

        # normalize
        max_possible_dcg = exp_gains / torch.log2(torch.arange(len(exp_gains)) + 2)
        exp_ndcg += (discounted_gains.sum() / max_possible_dcg.sum()).item()

    return ndcg/len(hr_map), exp_ndcg/len(hr_map)

#keep track of validation loss each epoch
with torch.inference_mode():
    mse, mae = eval_loss()
    dcg, edcg = eval_dcg()

#plot bars for each metric
groups = ["MSE", "MAE"]
types = ["URGE(Old Method)", "Mine", "UKGE(Paper)"]
n_groups = 2
n_types = 2

total_width = 0.5
d = 0.1
width = total_width/(n_types+(n_types-1)*d)
offset = -total_width/2

### plot    
x = np.arange(n_groups)
fig, ax = plt.subplots()
for t, data in zip(types, [[0.0748, 0.1135], [mse, mae], [paper_mse, paper_mae]]):
    ax.bar(x+offset, data, width, align='edge', label=t)
    offset += (1+d)*width
ax.set_xticks(x) ; ax.set_xticklabels(groups)
fig.legend()

plt.savefig(f'{data_set}-loss_bars.png', dpi=300)

groups = ["Linear nDCG", "Exponential nDCG"]
types = ["URGE(Old Method)", "Mine", "UKGE(Paper)"]
n_groups = 2
n_types = 2

total_width = 0.5
d = 0.1
width = total_width/(n_types+(n_types-1)*d)
offset = -total_width/2

### plot    
x = np.arange(n_groups)
fig, ax = plt.subplots()
for t, data in zip(types, [[0.593, 0.593], [dcg, edcg], [paper_lin_dcg, paper_exp_dcg]]):
    ax.bar(x+offset, data, width, align='edge', label=t)
    offset += (1+d)*width
ax.set_xticks(x) ; ax.set_xticklabels(groups)
fig.legend()

plt.savefig(f'{data_set}-dcg_bars.png', dpi=300)