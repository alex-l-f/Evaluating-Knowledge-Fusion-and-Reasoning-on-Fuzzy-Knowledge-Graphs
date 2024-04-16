from torch.utils.data import Dataset
import torch
import random
import os.path

class KGDataset(Dataset):

    def __init__(self, filename, import_indices=None, device = "cpu"):
        self.device = device
        # entity vocab
        self.ents = []
        # rel vocab
        self.rels = []
        # directories for id lookups
        self.index_ents = {}
        self.index_rels = {}
        self.lookup_ents = {}
        self.lookup_rels = {}
        self.total_len = 0
        self.kg_tuple_list = []
        self.kg_weight_list = []
        self.local_ent_heads = set([])
        self.local_ent_tails = set([])

        if import_indices is not None:
            self.index_rels = import_indices.index_rels
            self.index_ents = import_indices.index_ents
            self.lookup_ents = import_indices.lookup_ents
            self.lookup_rels = import_indices.lookup_rels
            self.ents = import_indices.ents
            self.rels = import_indices.rels
        self.local_ents = set([])
        self.triples_record = set([])
        self.base_triples_record = []
        self.base_weights_record = []

        # save triples as array of indices
        triples, weights = self.load_triples(filename)
        self.kg_tuple_list.append(triples)
        self.kg_weight_list.append(weights)
        self.kg_lens = [0,len(self.ents)]
        self.total_len = len(triples)
        # load psl triples if the _psl.tsv file exists
        self.using_psl = os.path.isfile(filename.replace(".tsv", "_psl.tsv"))
        if self.using_psl:
            self.psl_triples, self.psl_weights = self.load_triples(filename.replace(".tsv", "_psl.tsv"))
            self.psl_len = len(self.psl_triples)
    
    def finalize_dataset(self):
        #combine all kg relations into one big tensor
        self.triples = torch.cat(self.kg_tuple_list, 0)
        self.weights = torch.cat(self.kg_weight_list, 0)
        #send to device
        #self.triples = self.triples.to(self.device)
        #self.weights = self.weights.to(self.device)

        #self.kg_bounds = torch.zeros([len(self.triples),2], dtype=torch.int64)
        """ #compute highs and lows for each kg
        last_len = 0
        for i, kg in enumerate(self.kg_tuple_list):
            kg_len = len(kg)
            #handle links and attributes, which should cover all entites
            if i >= 2:
                self.kg_bounds[last_len:last_len+kg_len][0] = self.kg_lens[0]
                self.kg_bounds[last_len:last_len+kg_len][1] = self.kg_lens[-1]
            else:
                self.kg_bounds[last_len:last_len+kg_len][0] = self.kg_lens[i]
                self.kg_bounds[last_len:last_len+kg_len][1] = self.kg_lens[i+1]
            last_len += kg_len
        self.kg_bounds = self.kg_bounds.to(self.device) """

    def resetTupleSet(self):
        self.triples_record = set(self.base_triples_record)

    def load_kg(self, filename):
        kg_tuples, kg_weights = self.load_triples(filename)
        self.kg_tuple_list.append(kg_tuples)
        self.kg_weight_list.append(kg_weights)
        self.kg_lens.append(len(self.ents))
        self.total_len += len(kg_tuples)



    def load_triples(self, filename, splitter='\t', line_end='\n'):
        triples = []
        weights = []
        last_e = len(self.ents) - 1
        last_r = len(self.rels) - 1

        for line in open(filename):
            line = line.rstrip(line_end).split(splitter)
            self.local_ent_heads.add(line[0])
            self.local_ent_tails.add(line[2])
            if self.index_ents.get(line[0]) == None:
                self.ents.append(line[0])
                last_e += 1
                self.index_ents[line[0]] = last_e
                self.lookup_ents[last_e] = line[0]
            if self.index_ents.get(line[2]) == None:
                self.ents.append(line[2])
                last_e += 1
                self.index_ents[line[2]] = last_e
                self.lookup_ents[last_e] = line[2]
            if self.index_rels.get(line[1]) == None:
                self.rels.append(line[1])
                last_r += 1
                self.index_rels[line[1]] = last_r
                self.lookup_rels[last_r] = line[1]
            h = self.index_ents[line[0]]
            r = self.index_rels[line[1]]
            t = self.index_ents[line[2]]
            w = float(line[3])

            triples.append([h, r, t])
            weights.append(w)
            self.base_triples_record.append((h, r, t))
            self.base_weights_record.append(w)
        self.triples_record = set(self.base_triples_record)
        return torch.tensor(triples, dtype=torch.int64), torch.tensor(weights)

    def lookupRelName(self, x):
        return self.lookup_rels[x]
    
    def lookupEntName(self, x):
        return self.lookup_ents[x]
    
    def get_psl_ratio(self):
        return (self.psl_triples.size(0) / self.triples.size(0))

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        out = []
        """ kg_bounds_high = self.kg_bounds[idx][0]
        kg_bounds_low = self.kg_bounds[idx][1] """
        if self.using_psl:
            #get random psl entries
            pidx = random.randint(0, len(self.psl_triples)-1)
            return self.triples[idx], self.weights[idx], self.psl_triples[pidx], self.psl_weights[pidx]
        return self.triples[idx], self.weights[idx]