import random
from torch.utils.data import DataLoader
import torch
from UKGE.KGDataset import KGDataset
from UKGE.UKGE import UKGE
from tqdm import tqdm

#random search accross hyperparams for the best combo
num_samples = 100
num_epochs = 20
num_trials = 3
eval_every = 1
device = "cuda"
data_dir = "./data/OpenEA/"
data_set = "EN_DE_100K_V2"
use_reg = True

# hyperpaparms are LR, batch_size, num_negatives, embedding_dim, and reg_scale
#lr scaled up by 100000x for rnaodm seleciton purposes
max_lr = 10000
min_lr = 1

max_batch_size = 1024
min_batch_size = 64

max_num_negatives = 20
min_num_negatives = 1

max_embedding_dim = 256
min_embedding_dim = 64

#reg_scale scaled up by 1000x for random selection purposes
max_reg_scale = 1000
min_reg_scale = 1

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
    return loss.item()/len(val_loader)

for sample in range(num_samples):
    trial_avg = 0
    for trial in range(num_trials):
        lr = random.uniform(min_lr, max_lr)/100000
        batch_size = random.randint(min_batch_size, max_batch_size)
        num_negatives = random.randint(min_num_negatives, max_num_negatives)
        embedding_dim = random.randint(min_embedding_dim, max_embedding_dim)
        reg_scale = random.uniform(min_reg_scale, max_reg_scale)/1000

        #setup datasets
        train_set = KGDataset(data_dir + data_set + "/final_list.tsv")
        val_set = KGDataset(data_dir + data_set + "/valid_list.tsv", train_set)
        psl_scale = 0.2#train_set.get_psl_ratio()

        using_psl = train_set.using_psl

        #setup data loaders
        train_loader = DataLoader(train_set, batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size, shuffle=True)

        #model setup
        model = UKGE(len(train_set.ents), len(train_set.rels), embedding_dim, logi=False)

        #load model from ./model folder
        #model = torch.load("./models/9.09e-02.model")
        model.to(device)

        #setup loss function and optimizer
        #loss function is MSELoss, optimizer is Adam

        loss_func = torch.nn.MSELoss()
        optim = torch.optim.Adam(model.parameters(), lr)

        best_loss = 10

        #keep track of validation loss each epoch
        val_record = []

        for e in range(num_epochs):
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
            if((e+1) % eval_every == 0):
                vloss = eval()
                val_record.append(vloss)
                if vloss < best_loss:
                    best_loss = vloss

        #record trial stats
        if trial_avg == 0:
            trial_avg = best_loss
        else:  
            trial_avg += best_loss
    #record sample stats to file
    with open("./random_search_results.txt", "a") as f:
        f.write(f"{lr}, {batch_size}, {num_negatives}, {embedding_dim}, {reg_scale}, {trial_avg/num_trials}\n")