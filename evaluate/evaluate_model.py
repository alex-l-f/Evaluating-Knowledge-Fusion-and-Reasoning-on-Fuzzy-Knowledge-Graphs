#messy import since UKGE and the dataset are in a parent subdirectory
import sys
sys.path.append(".")

from torch.utils.data import DataLoader
import torch
from UKGE.KGDataset import KGDataset
from tqdm import tqdm
import os

#setup
device = "cuda"
batch_size = 512
dim = 128

dataset_paths = ["./data/psl50/OpenEA/EN_DE_100K_V2"]
model_paths = ["./results/models/psl50/EN_DE_100K_V2"]
files = ['kg1_list.tsv', 'kg2_list.tsv', 'kg_atts.tsv', 'link_list.tsv', 'kg1_list_psl.tsv']
dataset_name = {dataset_paths[0]: 'PSL 50%'}

def eval_loss():
    with torch.inference_mode():
        mse_loss = 0
        mae_loss = 0
        for data, targets in test_loader:
            confidences = torch.clip(model(data.to(device)),0,1)
            mse_loss += mse_loss_func(confidences, targets.to(device)).detach()
            mae_loss += mae_loss_func(confidences, targets.to(device)).detach()
    return mse_loss.item()/len(test_loader), mae_loss.item()/len(test_loader)

#gets rank for each known entry in a query relative to the whole vocabulary
def rankQuery(hr, tw):
    #get the tail as an int
    tail = list(tw.keys())[0]
    #construct and process batch with each tail pair
    with torch.inference_mode():
        full_t = torch.tensor([int(hr[0]), int(hr[1]), 0], dtype=torch.int, device=device).unsqueeze(0).expand(len(test_set.ents), -1).clone()
        full_t[:,2] = r_tens.clone()
        out = model(full_t.to(device))
        #get tensor of ordered indices
        _, indices = out.sort(descending=True)
        #get real index of tail id
        r_idx = (r_tens == tail).nonzero(as_tuple=True)[0]
        #get rank of the relevant tail
        rank = (indices == r_idx).nonzero(as_tuple=True)[0]
    return rank


def eval_hits_and_rank():
    hr_map = {}
    #get unique h, r pairs and all thier t mappings
    for i, triple in enumerate(test_set.triples_record):
        index = (int(triple[0]), int(triple[1]))
        if index in hr_map:
            hr_map[index][int(triple[2])] = float(test_set.base_weights_record[i])
        else:
            hr_map[index] = {int(triple[2]): float(test_set.base_weights_record[i])}
    #for each tuple
    h1 = 0
    h5 = 0
    h10 = 0
    h50 = 0
    mrr = 0
    mr = 0
    for (key, value) in tqdm(hr_map.items()):
        #rank tails
        rank = rankQuery(key, value)+1
        mr+=rank.item()
        
        #compute hits at 1, 10, 100
        if rank <= 1:
            h1 += 1
        if rank <= 5:
            h5 += 1
        if rank <= 10:
            h10 += 1
        if rank <= 50:
            h50 += 1

        #also compute mean reciprocal rank
        #get the rank of the first relevant tail
        mrr += 1/(rank.item())

    return h1/len(hr_map), h5/len(hr_map), h10/len(hr_map), h50/len(hr_map), mr/len(hr_map), mrr/len(hr_map)

#bulk process datasets
for fold in range(1,6):
    for i, dataset_path in enumerate(dataset_paths):
        #if teh directory does not exist, skip
        if not os.path.isdir(dataset_path + f"/{fold}/"):
            continue
        #set data directory path
        data_dir = dataset_path + f"/{fold}/"
        #set model directory path
        model_dir = model_paths[i] + f"/{fold}/"

        #set data_set path to look for the highest enumerated file (scientific notation) and load it
        model_name = max([f for f in os.listdir(model_dir) if f.endswith(".model")], key=lambda x: float(x[:-6]))

        #setup datasets
        train_set = KGDataset(data_dir + "/kg1_list.tsv")
        train_set.load_kg(data_dir + "/kg2_list.tsv")
        train_set.load_kg(data_dir + "/kg_atts.tsv")
        train_set.load_kg(data_dir + "/link_list.tsv")
        test_set = KGDataset(data_dir + "/test_list.tsv", train_set)
        train_set.finalize_dataset()
        test_set.finalize_dataset()

        #setup data laoders
        test_loader = DataLoader(test_set, batch_size, shuffle=True)

        #model setup
        #model = UKGE(len(test_set.ents), len(test_set.rels), dim)

        #load model from ./model folder
        model = torch.load(model_dir + "/" + model_name)#(f"./models/{data_set}/{model_name}")
        model.to(device)

        #setup loss function and optimizer
        #loss function is MSELoss, optimizer is Adam

        mse_loss_func = torch.nn.MSELoss()
        mae_loss_func = torch.nn.L1Loss()

        #validation loss
        r_tens = torch.tensor([test_set.index_ents[i] for i in test_set.ents], dtype=torch.int, device=device)

        #keep track of validation loss each epoch
        with torch.inference_mode():
            mse, mae = eval_loss()
            h1, h5, h10, h50, mr, mrr = eval_hits_and_rank()

        #log results for later plotting with other models
        with open("model_eval_results.tsv", "a") as f:
            f.write(f"{dataset_name[dataset_path]}/{fold}\t{mse}\t{mae}\t{h1}\t{h5}\t{h10}\t{h50}\t{mr}\t{mrr}\n")