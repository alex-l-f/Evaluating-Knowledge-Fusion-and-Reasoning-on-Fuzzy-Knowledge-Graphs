import os
from torch.utils.data import DataLoader
import torch
from UKGE.KGDataset import KGDataset
from UKGE.UKGE import UKGE
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import datetime

#setup
epochs = 1000
learn_rate = 0.0005
dim = 128
batch_size = 512
eval_every = 1

device = "cuda"
data_dir = "./data/psl50/OpenEA/"
base_data_set = "EN_DE_100K_V2"
use_reg = True
reg_scale = 0.0005
num_negatives = 5

def t_randint(low, high, size, device):
    return torch.randint(2**63 - 1, size=size, device=device) % (high - low) + low

def createNegativeSamples(data, num_negatives, train_set):
    with torch.no_grad():
        h = data[:,0]
        r = data[:,1]
        t = data[:,2]
        """ #corrupt head set
        h_corrupt = t_randint(off_start.repeat(num_negatives), off_end.repeat(num_negatives), [num_negatives* len(data)], device=device)
        #corrupt tail set
        t_corrupt = t_randint(off_start.repeat(num_negatives), off_end.repeat(num_negatives), [num_negatives* len(data)], device=device)
        """
        #corrupt head set
        h_corrupt = torch.randint(0, len(train_set.ents), [num_negatives* len(data)], device=device)
        #corrupt tail set
        t_corrupt = torch.randint(0, len(train_set.ents), [num_negatives* len(data)], device=device)

        #create negative head samples
        h_neg = torch.stack([h_corrupt, r.repeat(num_negatives), t.repeat(num_negatives)], 1)
        #create negative tail samples
        t_neg = torch.stack([h.repeat(num_negatives), r.repeat(num_negatives), t_corrupt],1)

        return h_neg, t_neg

#valid list is en->other only, so we can do one way comperisons with the tails
def rankQuery(hr, tw, r_tens, model, val_set):
    #get the tail as an int
    tail = list(tw.keys())[0]
    #construct and process batch with each tail pair
    with torch.inference_mode():
        full_t = torch.tensor([int(hr[0]), int(hr[1]), 0], dtype=torch.int, device=device).unsqueeze(0).expand(len(val_set.local_ent_tails), -1).clone()
        full_t[:,2] = r_tens.clone()
        out = model(full_t.to(device))
        #get tensor of ordered indices
        _, indices = out.sort(descending=True)
        #get real index of tail id
        r_idx = (r_tens == tail).nonzero(as_tuple=True)[0]
        #get rank of the relevant tail
        rank = (indices == r_idx).nonzero(as_tuple=True)[0]
    return rank

def build_hr_map(data_set):
    hr_map = {}
    #get unique h, r pairs and all thier t mappings
    for i, triple in enumerate(data_set.triples_record):
        index = (int(triple[0]), int(triple[1]))
        if index in hr_map:
            hr_map[index][int(triple[2])] = float(data_set.base_weights_record[i])
        else:
            hr_map[index] = {int(triple[2]): float(data_set.base_weights_record[i])}
    return hr_map

#validation loss
def eval(hr_map, val_set, r_tens, model):
    #for each kv pair
    mrr = 0
    for (key, value) in tqdm(hr_map.items()):
        #rank tails
        rank = rankQuery(key, value, r_tens, model, val_set)+1

        #compute mean reciprocal rank
        mrr += 1/(rank.item())

    #output val loss and example prediction
    print(f"MRR: {mrr/len(hr_map):.2e}")
    return mrr/len(hr_map)

def plot_loss(val_record, val_set, data_set):
    #plot validation loss for each epoch
    plt.plot(np.arange(0,eval_every*len(val_record),eval_every), val_record, label="Model Performance")
    plt.plot([0, eval_every*len(val_record)-1], [1/(len(val_set.local_ent_tails)*0.1), 1/(len(val_set.local_ent_tails)*0.1)], 'k--', lw=2, label="Top 90%")
    plt.plot([0, eval_every*len(val_record)-1], [1/(len(val_set.local_ent_tails)*0.01), 1/(len(val_set.local_ent_tails)*0.01)], 'k--', lw=2, label="Top 99%")
    plt.xlabel("Epochs")
    plt.ylabel("MRR")


    plt.legend()
    plt.title("MRR over Epochs")
    plt.savefig(f"{data_set.replace('/','_')}-MRR.png", dpi=300)
    plt.close()

def main():
        #for each fold
    for fold in range(1, 6):

        data_set = f"{base_data_set}/{fold}"

        #setup datasets
        train_set = KGDataset(data_dir + data_set + "/kg1_list.tsv", device=device)
        train_set.load_kg(data_dir + data_set + "/kg2_list.tsv")
        train_set.load_kg(data_dir + data_set + "/kg_atts.tsv")
        train_set.load_kg(data_dir + data_set + "/link_list.tsv")
        train_set.finalize_dataset()
        val_set = KGDataset(data_dir + data_set + "/valid_list.tsv", train_set, device=device)
        val_set.finalize_dataset()
        #tensor that contains all the other KG tails
        r_tens = torch.tensor([val_set.index_ents[i] for i in val_set.local_ent_tails], dtype=torch.int, device=device)
        psl_scale = 0.2

        using_psl = train_set.using_psl

        #get "link" relation count from dataset
        link_id = train_set.index_rels["link"]
        link_count = len(train_set.kg_tuple_list[0])

        #setup data loaders
        train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=8)
        hr_map = build_hr_map(val_set)

        #model setup
        model = UKGE(len(train_set.ents), len(train_set.rels), dim, logi=False)

        #load model from ./model folder
        #model = torch.load("./models/9.09e-02.model")
        model.to(device)

        #setup loss function and optimizer
        #loss function is MSELoss, optimizer is Adam

        loss_func = torch.nn.MSELoss()
        optim = torch.optim.Adam(model.parameters(), learn_rate)
        r_loss = 0
        best_score = 0
        stale = 0

        #keep track of validation loss each epoch
        val_record = []
        datestr = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        log_name = f"./run_logs/{data_set}/{datestr}_{link_count//1000}K_{using_psl}_progress.tsv"

        for e in range(epochs):
            for i, dt in enumerate(tqdm(train_loader)):
                h_neg_loss = 0
                t_neg_loss = 0
                if using_psl:
                    #extract psl triples
                    (data, targets, psl_data, psl_targets) = dt
                else:
                    (data, targets) = dt
                #generate new negative samples
                data = data.to(device)
                targets = targets.to(device)
                h_neg, t_neg = createNegativeSamples(data, num_negatives, train_set)
                #get neg loss
                h_neg_loss = h_neg_loss + model(h_neg).square().mean()
                t_neg_loss = t_neg_loss + model(t_neg).square().mean()

                if use_reg:
                    confidences, r_score = model(data, regularize=use_reg, regularize_scale=reg_scale)
                    loss = loss_func(confidences, targets) + r_score
                else:
                    confidences = model(data)
                    loss = loss_func(confidences, targets)
                if using_psl:
                    loss += model.compute_psl_loss(psl_data.to(device), psl_targets.to(device), psl_off=0, psl_scale=psl_scale)
                (loss + (h_neg_loss + t_neg_loss)/2).backward()
                optim.step()
                optim.zero_grad()
                #r_loss = r_loss * 0.9 + loss.detach() * 0.1
            if((e+1) % eval_every == 0):
                vscore = eval(hr_map, val_set, r_tens, model)
                val_record.append(vscore)
                plot_loss(val_record, val_set, data_set)
                #log current progress to tsv
                with open(log_name, 'a') as file:
                    file.write(f"{e}\t{r_loss:.2e}\t{vscore:.2e}\n")
                if vscore > best_score:
                    os.makedirs(f"./models/{data_set}/", exist_ok = True) 
                    torch.save(model, f"./models/{data_set}/{vscore:.2e}.model")
                    best_score = vscore
                    stale = 0
            #add early stopping if loss is stale
            if len(val_record) > 5:
                if best_score > val_record[-1]:
                    stale += 1
                else:
                    stale = 0
            if stale >= 30:
                break

if __name__ == "__main__":
    main()