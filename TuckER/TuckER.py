import torch
from torch import nn
from torch.functional import F
import numpy as np

class TuckER(nn.Module):
    
    def __init__(self, num_ents, num_rels, e_dim=128, r_dim=128, i_dropout=0.3, h_dropout=0.4, h_dropout2=0.5, **kwargs):
        super(TuckER, self).__init__()

        self.E = torch.nn.Embedding(num_ents, e_dim)
        self.R = torch.nn.Embedding(num_rels, r_dim)
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (r_dim, e_dim, e_dim)), 
                                    dtype=torch.float, device="cuda", requires_grad=True))

        self.input_dropout = torch.nn.Dropout(i_dropout)
        self.hidden_dropout1 = torch.nn.Dropout(h_dropout)
        self.hidden_dropout2 = torch.nn.Dropout(h_dropout2)
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm1d(e_dim)
        self.bn1 = torch.nn.BatchNorm1d(e_dim)
        

    def init(self):
        nn.init.xavier_normal_(self.E.weight.data)
        nn.init.xavier_normal_(self.R.weight.data)
    

    def forward(self, x):
        h = x[:,0]
        r = x[:,1]

        e1 = self.E(h)
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        r = self.R(r)
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat) 
        x = x.view(-1, e1.size(1))      
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, self.E.weight.transpose(1,0))
        pred = torch.sigmoid(x)

        return pred
