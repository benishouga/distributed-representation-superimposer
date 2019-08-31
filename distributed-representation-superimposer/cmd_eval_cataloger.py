import os
import sys
import time
import random

import torch
from torch import nn
from torch import optim
import torchtext

from cataloger import Cataloger
from dataset.fields import field_labels

from extractor import Extractor


def cmd_eval_cataloger(args):
    extractor = Extractor('../Japanese_L-12_H-768_A-12_E-30_BPE/')
    dr = torch.FloatTensor([extractor.extract(args.text)])

    for target in ("intent", "place", "datetime"):
        model_path = getattr(args, "model_" + target)

        net = Cataloger(catalog_features=len(field_labels[target]))
        net.load_state_dict(torch.load(model_path))

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        net.eval()
        outputs = net(dr)
        _, predicted = torch.max(outputs, 1)
        for v in field_labels[target]:
            if field_labels[target][v] == predicted[0]:
                print("{}: {}".format(target, v))
