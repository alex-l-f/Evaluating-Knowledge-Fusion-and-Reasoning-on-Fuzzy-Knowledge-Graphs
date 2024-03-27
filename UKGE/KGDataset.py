from torch.utils.data import Dataset
import torch
import random
import os.path

class KGDataset(Dataset):

    def __init__(self, filename, import_indices=None):
        # entity vocab
        self.ents = []
        # rel vocab
        self.rels = []
        # directories for id lookups
        self.index_ents = {}
        self.index_rels = {}
        self.lookup_ents = {}
        self.lookup_rels = {}

        if import_indices is not None:
            self.index_rels = import_indices.index_rels
            self.index_ents = import_indices.index_ents
            self.lookup_ents = import_indices.lookup_ents
            self.lookup_rels = import_indices.lookup_rels
            self.ents = import_indices.ents
            self.rels = import_indices.rels

        self.triples_record = set([])
        self.base_triples_record = set([])

        # save triples as array of indices
        self.triples, self.weights = self.load_triples(filename)
        # load psl triples if the _psl.tsv file exists
        self.using_psl = os.path.isfile(filename.replace(".tsv", "_psl.tsv"))
        if self.using_psl:
            self.psl_triples, self.psl_weights = self.load_triples(filename.replace(".tsv", "_psl.tsv"))
            self.psl_len = len(self.psl_triples)
        self.triples.requires_grad = False
        self.num_base = len(self.triples)

    def resetTupleSet(self):
        self.triples_record = self.base_triples_record.copy()


    def load_triples(self, filename, splitter='\t', line_end='\n'):
        triples = []
        weights = []
        last_e = -1
        last_r = -1

        for line in open(filename):
            line = line.rstrip(line_end).split(splitter)
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
            self.base_triples_record.add((h, r, t))
        self.triples_record = self.base_triples_record.copy()
        return torch.tensor(triples, dtype=torch.int64), torch.tensor(weights)

    def lookupRelName(self, x):
        return self.lookup_rels[x]
    
    def lookupEntName(self, x):
        return self.lookup_ents[x]
    
    def get_psl_ratio(self):
        return (self.psl_triples.size(0) / self.triples.size(0))

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        if self.using_psl:
            #get random psl entries
            pidx = random.randint(0, len(self.psl_triples)-1)
            return self.triples[idx], self.weights[idx], self.psl_triples[pidx], self.psl_weights[pidx]
        return self.triples[idx], self.weights[idx]