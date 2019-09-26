import os
import sys
import time
import random

import torch
from torch import nn
from torch import optim
import torchtext

from classifier import Classifier
from superimposer import Superimposer
from dataset.fields import field_labels

from extractor import Extractor


def cmd_eval_repl(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    extractor = Extractor('../Japanese_L-12_H-768_A-12_E-30_BPE/')

    net_superimposer = Superimposer()
    net_superimposer.load_state_dict(torch.load(args.model))
    net_superimposer.to(device)
    net_superimposer.eval()
    classifier_nets = {}

    for target in ("intent", "place", "datetime"):
        model_path = getattr(args, "model_" + target)

        net = Classifier(catalog_features=len(field_labels[target]))
        net.load_state_dict(torch.load(model_path))
        net.to(device)
        net.eval()
        classifier_nets[target] = net

    prev_dr = None
    first = True

    while True:
        print("\n> ", end='')
        input_text = input()

        if input_text == "exit":
            break

        if input_text == "reset":
            first = True
            continue

        next_dr = torch.FloatTensor([extractor.extract(input_text)])

        print("----input_dr----")
        print_classifier_result(classifier_nets, next_dr)

        if not first:
            result_dr = net_superimposer(prev_dr, next_dr)

            print("----superimposer_dr----")
            print_classifier_result(classifier_nets, result_dr)
            prev_dr = result_dr
        else:
            first = False
            prev_dr = next_dr


def print_classifier_result(classifier_nets, result):
    for target in ("intent", "place", "datetime"):
        outputs = classifier_nets[target](result)

        _, predicted = torch.max(outputs, 1)
        for v in field_labels[target]:
            if field_labels[target][v] == predicted[0]:
                print("{}: {}".format(target, v))
