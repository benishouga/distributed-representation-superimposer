import os
import sys
import time
import random

import torch
from torch import nn
from torch import optim
import torchtext

from cataloger import Cataloger
from superimposer import Superimposer
from dataset.fields import field_labels

from extractor import Extractor


def cmd_eval_superimposer(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    extractor = Extractor('../Japanese_L-12_H-768_A-12_E-30_BPE/')
    dr1 = torch.FloatTensor([extractor.extract(args.text1)])
    dr2 = torch.FloatTensor([extractor.extract(args.text2)])

    net_superimposer = Superimposer()
    net_superimposer.load_state_dict(torch.load(args.model))
    net_superimposer.to(device)
    net_superimposer.eval()
    result_dr = net_superimposer(dr1, dr2)

    for target in ("intent", "place", "datetime"):
        model_path = getattr(args, "model_" + target)

        net = Cataloger(catalog_features=len(field_labels[target]))
        net.load_state_dict(torch.load(model_path))
        net.to(device)
        net.eval()
        outputs = net(result_dr)
        _, predicted = torch.max(outputs, 1)
        for v in field_labels[target]:
            if field_labels[target][v] == predicted[0]:
                print("{}: {}".format(target, v))
