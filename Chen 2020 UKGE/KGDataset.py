from torch.utils.data import Dataset
import torch
import random

class KGDataset(Dataset):

    def __init__(self, filename):
        # entity vocab
        self.ents = []
        # rel vocab
        self.rels = []
        # directories for id lookups
        self.index_ents = {}
        self.index_rels = {}
        self.lookup_ents = {}
        self.lookup_rels = {}

        self.triples_record = set([])

        # save triples as array of indices
        self.triples = self.load_triples(filename)

    def load_triples(self, filename, splitter='\t', line_end='\n'):
        triples = []
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

            triples.append([h, r, t, w])
            self.triples_record.add((h, r, t))
        return torch.tensor(triples)
    
    def createNegativeSamples(self, ratio):
        num_negatives = int(self.triples.size(0)*ratio)
        created_negatives = []
        while len(created_negatives) < num_negatives:
            #sample a random relationship and corrupt
            #this is REAL bad, but it's what the paper does
            rsh = random.randint(0, self.triples.size(0)-1)
            if random.randint(0,1):
                nrel = (self.triples[rsh,0], self.triples[rsh,1], random.randint(0, len(self.ents)-1))
            else:
                nrel = (random.randint(0, len(self.ents)-1),self.triples[rsh,1], self.triples[rsh,2])
            if nrel not in self.triples_record:
                created_negatives.append([*nrel, 0.0])
                self.triples_record.add(nrel)

        #tensorize and cat negative examples
        neg_t = torch.tensor(created_negatives)
        self.triples = torch.cat([self.triples, neg_t], 0)

    def set_lookups(self, dataset):
        self.index_rels = dataset.index_rels
        self.index_ents = dataset.index_ents
        self.lookup_ents = dataset.lookup_ents
        self.lookup_rels = dataset.lookup_rels

    def lookupRelName(self, x):
        return self.lookup_rels[x]
    
    def lookupEntName(self, x):
        return self.lookup_ents[x]

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        return self.triples[idx, :-1].type(torch.int), self.triples[idx, -1]