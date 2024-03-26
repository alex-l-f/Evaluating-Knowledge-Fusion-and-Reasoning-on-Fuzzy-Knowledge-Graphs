import torch
from torch import nn
from torch.functional import F

class UKGE(nn.Module):
    
    def bounded_map(self, x):
        return torch.min(torch.max(x, 0, 1))

    #dim 64 paper default
    def __init__(self, num_ents, num_rels, dim=64, logi=True, regularize=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.logi = logi
        self.regularize = regularize
        self.relationEmbed = nn.Embedding(num_rels, dim)
        self.map = torch.nn.Sigmoid()

        if logi:
            self.entityEmbed = nn.Embedding(num_ents, dim)

            self.lin = nn.Linear(1, 1)
        else:
            self.epsilon = 2.0

            self.entityEmbed = nn.Embedding(num_ents, dim*2)
        
            self.gamma = nn.Parameter(
                torch.Tensor([12.0]), 
                requires_grad=False
            )
            
            self.embedding_range = nn.Parameter(
                torch.Tensor([(self.gamma.item() + self.epsilon) / dim]), 
                requires_grad=False
            )

    def forward(self, x):
        h = self.entityEmbed(x[:,0])
        r = self.relationEmbed(x[:,1])
        t = self.entityEmbed(x[:,2])

        #regularize head, tail and relation embedding
        r_score = (nn.MSELoss(h)/h.size(0) + nn.MSELoss(t)/t.size(0) + nn.MSELoss(r)/r.size(0))

        if self.logi:
            p_score = (r * (h*t)).sum(-1, keepdim = True)
            confidence = self.map(self.lin(p_score).squeeze())

        else:
            #return bounded rectangle loss instead
            p_score = (r * (h*t)).sum(-1, keepdim = True)
            confidence = self.bounded_map(self.lin(p_score))

        if self.regularize:
            return confidence, r_score
        else:
            return confidence
