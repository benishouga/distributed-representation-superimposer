import torch
from torch import nn
from torch.functional import F

class Cataloger(nn.Module):
    def __init__(self, catalog_features, hidden_units=768):
        super(Cataloger, self).__init__()

        self.l1 = nn.Linear(in_features=768, out_features=hidden_units)
        self.l2 = nn.Linear(in_features=hidden_units,
                            out_features=hidden_units)
        self.l3 = nn.Linear(in_features=hidden_units,
                            out_features=catalog_features)

    def forward(self, dr):
        x = F.relu(self.l1(dr))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.l2(x))
        return self.l3(x)
