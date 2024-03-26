import torch
from torch import nn
from torch.functional import F

class UKGE(nn.Module):
    
    def bounded_map(self, x):
        return torch.min(torch.max(x, 0, 1))

    #dim 64 paper default
    def __init__(self, num_ents, num_rels, dim=64, logi=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.logi = logi
        self.relationEmbed = nn.Embedding(num_rels, dim)
        self.map = torch.nn.Sigmoid()

        self.entityEmbed = nn.Embedding(num_ents, dim)
        self.lin = nn.Linear(1, 1)
        

    def compute_psl_confidence(self, x):
        h = self.entityEmbed(x[:,0])
        r = self.relationEmbed(x[:,1])
        t = self.entityEmbed(x[:,2])
        if self.logi:
            p_score = (r * (h*t)).sum(-1, keepdim = True)
            confidence = self.map(self.lin(p_score).squeeze())
        else:
            #return bounded rectangle loss instead
            p_score = (r * (h*t)).sum(-1, keepdim = True)
            confidence = self.lin(p_score)

        return confidence
    
    def compute_psl_loss(self, x, w, psl_off = 0, psl_scale = 1):
        confidence = self.compute_psl_confidence(x)
        return torch.square(torch.clip(w - confidence + psl_off, 0).mean())*psl_scale

    def forward(self, x, regularize=False, regularize_scale=0.0005):
        h = self.entityEmbed(x[:,0])
        r = self.relationEmbed(x[:,1])
        t = self.entityEmbed(x[:,2])

        #regularize head, tail and relation embedding
        if regularize:
            r_score = (h.square().sum()/h.size(0) + t.square().sum()/t.size(0) + r.square().sum()/r.size(0)) * regularize_scale

        if self.logi:
            p_score = (r * (h*t)).sum(-1, keepdim = True)
            confidence = self.map(self.lin(p_score).squeeze())
        else:
            #return bounded rectangle loss instead
            p_score = (r * (h*t)).sum(-1, keepdim = True)
            confidence = self.lin(p_score).squeeze()

        if regularize:
            return confidence, r_score
        else:
            return confidence
