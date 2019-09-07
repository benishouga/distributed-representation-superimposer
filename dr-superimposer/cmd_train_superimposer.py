import os
import sys
import time
import random

import numpy as np

import torch
from torch import nn
from torch import optim
import torchtext

from classifier import Classifier
from superimposer import Superimposer

from dataset.text_holder import TextHolder
from dataset.superimposer_dataset import SuperimposerDataset


def train(net, classifiers, text_holder, dataloaders_dict, batch_size, criterion, optimizer, num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net.to(device)
    for classifier_name in classifiers:
        classifier_net = classifiers[classifier_name]["net"]
        classifier_net.to(device)
        classifier_net.eval()

    torch.backends.cudnn.benchmark = True

    for epoch in range(num_epochs):
        for phase in dataloaders_dict:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            corrects = {}
            for classifier_name in classifiers:
                display_label = classifiers[classifier_name]["display_labels"]
                labels_length = len(display_label)
                corrects[classifier_name] = np.zeros(
                    [labels_length, labels_length], dtype=int)

            epoch_loss = 0.0
            epoch_corrects = 0
            iteration = 1

            t_epoch_start = time.time()
            t_iter_start = time.time()

            batches = dataloaders_dict[phase]
            for data in batches:
                inputs1 = data.dr1.to(device)
                inputs2 = data.dr2.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs1, inputs2)
                    loss = 0
                    preds = {}
                    acc = 0
                    for classifier_name in classifiers:
                        labels = getattr(data, classifier_name).to(device)
                        o = classifiers[classifier_name]["net"](outputs)
                        labels = labels.squeeze(1)
                        loss += criterion(o, labels)
                        _, preds = torch.max(o, 1)
                        acc += (torch.sum(preds == labels.data)
                                ).double()/batch_size

                        correct = corrects[classifier_name]
                        for i in range(len(labels)):
                            correct[labels[i]][preds[i]] += 1
                            if labels[i] != preds[i]:
                                print(text_holder.get(data.text1[i][0]) + ", " + text_holder.get(data.text2[i][0]))

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                        if (iteration % 100 == 0):
                            t_iter_finish = time.time()
                            duration = t_iter_finish - t_iter_start
                            acc = acc / len(classifiers)
                            print('Iteration: {} || Loss: {:.4f} || {:.4f} sec. || Acc：{}'.format(
                                iteration, loss.item(), duration, acc))

                    iteration += 1

                    epoch_loss += loss.item() * batch_size
                    epoch_corrects += torch.sum(preds == labels.data)

            # epochごとのlossと正解率
            t_epoch_finish = time.time()
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double(
            ) / len(dataloaders_dict[phase].dataset)

            print('Epoch {}/{} | {:^5} |  Loss: {:.4f} Acc: {:.4f}'.format(epoch+1, num_epochs,
                                                                           phase, epoch_loss, epoch_acc))

            for classifier_name in classifiers:
                print("----{}----".format(classifier_name))
                print(classifiers[classifier_name]["display_labels"].keys())
                print(corrects[classifier_name])


    return net


def cmd_train_superimposer(args):
    text_holder = TextHolder()
    td = SuperimposerDataset(path=args.input, text_holder=text_holder)
    training_data, validation_data = td.split(
        split_ratio=0.8, random_state=random.seed(1234))

    batch_size = 32

    classifiers = {}
    for target in ("intent", "place", "datetime"):
        classifier_model_path = getattr(args, "model_" + target)
        display_labels = td.fields[target].labels
        net = Classifier(catalog_features=len(display_labels))
        net.load_state_dict(torch.load(classifier_model_path))
        classifiers[target] = {"net": net, "display_labels": display_labels}

    if args.validation_only == True:
        dataloaders_dict = {
            'val': torchtext.data.Iterator(validation_data, batch_size=batch_size, train=False, sort=False)
        }
        net = Superimposer()
        model_path = args.model
        net.load_state_dict(torch.load(model_path))

        optimizer = optim.Adam(net.parameters(), lr=5e-5)
        criterion = nn.CrossEntropyLoss()
        train(net, classifiers, text_holder, dataloaders_dict,
              batch_size, criterion, optimizer, 1)

        return

    dataloaders_dict = {
        'train': torchtext.data.Iterator(training_data, batch_size=batch_size, train=True),
        'val': torchtext.data.Iterator(validation_data, batch_size=batch_size, train=False, sort=False)
    }

    net = Superimposer()
    model_path = args.model
    if model_path is not None and os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path))

    optimizer = optim.Adam(net.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()
    train(net, classifiers, text_holder, dataloaders_dict,
          batch_size, criterion, optimizer, 10)

    if model_path is not None:
        torch.save(net.state_dict(), model_path)
