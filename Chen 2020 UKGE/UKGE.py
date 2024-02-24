import torch
from torch import nn

class UKGE(nn.Module):
    def logistic_map(self, x):
        return 1.0/(1.0+torch.exp(-x))
    
    def bounded_map(self, x):
        return torch.min(torch.max(x, 0, 1))

    #dim 64 paper default
    def __init__(self, num_ents, num_rels, dim=64, logi=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.entityEmbed = nn.Embedding(num_ents, dim)
        self.relationEmbed = nn.Embedding(num_rels, dim)

        self.lin = nn.Linear(1, 1)

        if logi:
            self.map = self.logistic_map
        else:
            self.map = self.bounded_map

    def forward(self, x):
        h = self.entityEmbed(x[:,0])
        r = self.relationEmbed(x[:,1])
        t = self.entityEmbed(x[:,2])

        p_score = (r * (h*t)).sum(-1, keepdim = True)
        confidence = self.map(self.lin(p_score).squeeze())

        return confidence
