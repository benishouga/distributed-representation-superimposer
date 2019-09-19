import os
import sys
import time
import random

import numpy as np

import torch
from torch import nn
from torch import optim
import torchtext
from torch.utils.data import RandomSampler
from torchtext.data.utils import RandomShuffler
from torchtext.data.dataset import Dataset

from classifier import Classifier
from superimposer import Superimposer

from dataset.text_holder import TextHolder
from dataset.classifier_dataset import ClassifierDataset


def deside_expectes(one_batch, two_batch, classifiers):
    intent_labels = classifiers["intent"]["display_labels"]
    place_labels = classifiers["place"]["display_labels"]
    datetime_labels = classifiers["datetime"]["display_labels"]

    # label = label if label in self.labels else "unknown"
    # return [self.labels[label]]
    # intent_labels

    intents = []
    places = []
    datetimes = []

    for i in range(len(one_batch)):
        # print(i)
        one = one_batch.dataset[i]
        two = two_batch.dataset[i]
        # print(vars(one))
        # print(vars(two))
        one_intent = None
        one_place = None
        one_datetime = None
        two_intent = None
        two_place = None
        two_datetime = None
        for key in intent_labels:
            if one.intent[0] == intent_labels[key]:
                one_intent = key
            if two.intent[0] == intent_labels[key]:
                two_intent = key

        for key in place_labels:
            if one.place[0] == place_labels[key]:
                one_place = key
            if two.place[0] == place_labels[key]:
                two_place = key

        for key in datetime_labels:
            if one.datetime[0] == datetime_labels[key]:
                one_datetime = key
            if two.datetime[0] == datetime_labels[key]:
                two_datetime = key

        intents.append([intent_labels[one_intent] if two_intent ==
                        "continue" else intent_labels[two_intent]])
        places.append([place_labels[one_place] if two_place ==
                       "unknown" else place_labels[two_place]])
        datetimes.append([datetime_labels[one_datetime] if two_datetime ==
                          "unknown" else datetime_labels[two_datetime]])

    return {
        "intent": torch.tensor(intents),
        "place": torch.tensor(places),
        "datetime": torch.tensor(datetimes)
    }


def train(net, classifiers, text_holder, dataloaders_dict, batch_size, criterion, optimizer, num_epochs, validation_only=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    shuffler = RandomShuffler(random_state=random.seed(1234))

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

            t_iter_start = time.time()

            batches = dataloaders_dict[phase]
            for prev_data_batch in batches:
                prev_dr_batch = prev_data_batch.dr.to(device)

                optimizer.zero_grad()

                dataset = batches.dataset
                randperm = shuffler(range(len(dataset.examples)))
                random_dataset = Dataset([dataset.examples[i]
                                          for i in randperm], dataset.fields)
                iterator = torchtext.data.Iterator(
                    random_dataset, batch_size=batch_size, train=False, sort=False)
                random_iter = iter(iterator)

                with torch.set_grad_enabled(phase == 'train'):
                    next_data = next(random_iter)
                    next_dr = next_data.dr.to(device)
                    next_dr = next_dr[0:len(prev_dr_batch)]

                    outputs = net(prev_dr_batch, next_dr)
                    loss = 0
                    preds = {}
                    acc = 0

                    expect_datas = deside_expectes(
                        prev_data_batch, next_data, classifiers)
                    for classifier_name in classifiers:
                        labels = expect_datas[classifier_name].to(device)
                        o = classifiers[classifier_name]["net"](outputs)
                        labels = labels.squeeze(1)
                        loss += criterion(o, labels)
                        _, preds = torch.max(o, 1)
                        acc += (torch.sum(preds == labels.data)
                                ).double()/batch_size

                        correct = corrects[classifier_name]
                        for i in range(len(labels)):
                            correct[labels[i]][preds[i]] += 1
                            if labels[i] != preds[i] and validation_only:
                                print(text_holder.get(
                                    data.text1[i][0]) + "\t" + text_holder.get(data.text2[i][0]))

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
    td = ClassifierDataset(path=args.input, text_holder=text_holder)

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
              batch_size, criterion, optimizer, 1, True)

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
