import torch
from torch import nn
from torch.functional import F


class Superimposer(nn.Module):
    def __init__(self):
        super(Superimposer, self).__init__()

        self.l1 = nn.Linear(in_features=768 * 2, out_features=768 * 2)
        self.l2 = nn.Linear(in_features=768 * 2, out_features=768)
        self.l3 = nn.Linear(in_features=768, out_features=768)
        self.l4 = nn.Linear(in_features=768, out_features=768)
        self.l5 = nn.Linear(in_features=768, out_features=768)

    def forward(self, dr1, dr2):
        x = torch.cat((dr1, dr2), 1)
        x = F.relu(self.l1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.l2(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.l3(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.l4(x))
        return self.l5(x)
