import os
import sys
import time
import random

import torch
from torch import nn
from torch import optim
import torchtext

from cateloger import Cataloger
from field.dataset import Dataset

from extractor import Extractor


def cmd_eval(args):
    extractor = Extractor('../Japanese_L-12_H-768_A-12_E-30_BPE/')
    dr = torch.FloatTensor([extractor.extract(args.text)])
    # HACK
    td = Dataset(
        path="distributed-representation-superimposer/data/data_100.tsv")

    for target in ("intent", "place", "datetime"):
        model_path = getattr(args, "model_" + target)

        net = Cataloger(catalog_features=len(td.fields[target].labels))
        net.load_state_dict(torch.load(model_path))

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        net.eval()
        outputs = net(dr)
        _, predicted = torch.max(outputs, 1)
        for v in td.fields[target].labels:
            if td.fields[target].labels[v] == predicted[0]:
                print("{}: {}".format(target, v))

