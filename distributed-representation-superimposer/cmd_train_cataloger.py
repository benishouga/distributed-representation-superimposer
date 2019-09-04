import os
import sys
import time
import random

import numpy as np

import torch
from torch import nn
from torch import optim
import torchtext

from cataloger import Cataloger
from dataset.cataloger_dataset import CatalogerDataset


def train(net, target, display_labels, dataloaders_dict, batch_size, criterion, optimizer, num_epochs):
    labels_length = len(display_labels)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net.to(device)

    torch.backends.cudnn.benchmark = True
    for epoch in range(num_epochs):
        for phase in dataloaders_dict:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            correct = np.zeros([labels_length, labels_length], dtype=int)

            epoch_loss = 0.0
            epoch_corrects = 0
            iteration = 1

            t_epoch_start = time.time()
            t_iter_start = time.time()

            for data in dataloaders_dict[phase]:
                inputs = data.dr.to(device)
                labels = getattr(data, target).to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    labels = labels.squeeze(1)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    for i in range(len(labels)):
                        correct[labels[i]][preds[i]] += 1

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                        if (iteration % 100 == 0):
                            duration = time.time() - t_iter_start
                            acc = (torch.sum(preds == labels.data)
                                   ).double()/batch_size
                            print('Iteration: {} || Loss: {:.4f} || {:.4f} sec. || Acc：{}'.format(
                                iteration, loss.item(), duration, acc))

                    iteration += 1

                    epoch_loss += loss.item() * batch_size
                    epoch_corrects += torch.sum(preds == labels.data)

            t_epoch_finish = time.time()
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double(
            ) / len(dataloaders_dict[phase].dataset)

            print('Epoch {}/{} | {:^5} |  Loss: {:.4f} Acc: {:.4f}'.format(epoch+1, num_epochs,
                                                                           phase, epoch_loss, epoch_acc))
            print(display_labels.keys())
            print(correct)

    return net


def cmd_train_cataloger(args):
    td = CatalogerDataset(path=args.input)
    training_data, validation_data = td.split(
        split_ratio=0.8, random_state=random.seed(1234))

    batch_size = 32

    if args.validation_only == True:
        dataloaders_dict = {
            'val': torchtext.data.Iterator(validation_data, batch_size=batch_size, train=False, sort=False)
        }
        for target in ("intent", "place", "datetime"):
            print("----{}----".format(target))
            model_path = getattr(args, "model_" + target)
            display_labels = td.fields[target].labels
            net = Cataloger(catalog_features=len(display_labels))
            net.load_state_dict(torch.load(model_path))
            optimizer = optim.Adam(net.parameters(), lr=5e-5)
            criterion = nn.CrossEntropyLoss()
            train(net, target, display_labels, dataloaders_dict,
                  batch_size, criterion, optimizer, 1)
        return

    dataloaders_dict = {
        'train': torchtext.data.Iterator(training_data, batch_size=batch_size, train=True),
        'val': torchtext.data.Iterator(validation_data, batch_size=batch_size, train=False, sort=False)
    }

    for target in ("intent", "place", "datetime"):
        print("----{}----".format(target))

        model_path = getattr(args, "model_" + target)

        display_labels = td.fields[target].labels
        net = Cataloger(catalog_features=len(display_labels))
        if model_path is not None and os.path.exists(model_path):
            net.load_state_dict(torch.load(model_path))

        optimizer = optim.Adam(net.parameters(), lr=5e-5)
        criterion = nn.CrossEntropyLoss()
        train(net, target, display_labels, dataloaders_dict,
              batch_size, criterion, optimizer, 10)

        if model_path is not None:
            torch.save(net.state_dict(), model_path)
