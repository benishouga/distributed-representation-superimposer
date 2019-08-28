import sys
import time
import random
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch import nn
from torch import optim
import torchtext

from cateloger import Cataloger
from field.dataset import Dataset


def train(net, target, dataloaders_dict, criterion, optimizer, num_epochs):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net.to(device)

    torch.backends.cudnn.benchmark = True

    batch_size = dataloaders_dict["train"].batch_size

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_corrects = 0
            iteration = 1

            t_epoch_start = time.time()
            t_iter_start = time.time()

            batches = dataloaders_dict[phase]
            for data in batches:
                inputs = data.dr.to(device)
                labels = getattr(data, target).to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    labels = labels.squeeze(1)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                        if (iteration % 100 == 0):
                            t_iter_finish = time.time()
                            duration = t_iter_finish - t_iter_start
                            acc = (torch.sum(preds == labels.data)
                                   ).double()/batch_size
                            print('Iteration: {} || Loss: {:.4f} || {:.4f} sec. || Acc：{}'.format(
                                iteration, loss.item(), duration, acc))
                            t_iter_start = time.time()

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
            t_epoch_start = time.time()

    return net


def cmd_train(args):
    td = Dataset(path=args.input)
    training_data, validation_data = td.split(
        split_ratio=0.8, random_state=random.seed(1234))

    batch_size = 32

    dataloaders_dict = {
        'train': torchtext.data.Iterator(training_data, batch_size=batch_size, train=True),
        'val': torchtext.data.Iterator(validation_data, batch_size=batch_size, train=False, sort=False)
    }

    for target in ("intent", "place", "datetime"):
        print("----{}----".format(target))
        net = Cataloger(catalog_features=len(td.fields[target].labels))
        optimizer = optim.Adam(net.parameters(), lr=5e-5)
        criterion = nn.CrossEntropyLoss()
        train(net, target, dataloaders_dict, criterion, optimizer, 10)


def cmd_eval(args):
    print(args)


def cmd_help(args):
    print(parser.parse_args([args.command, '--help']))


def main():
    parser = ArgumentParser()
    sub = parser.add_subparsers()

    mode_train = sub.add_parser('train')
    mode_train.add_argument('--input', type=Path, required=True)
    mode_train.set_defaults(handler=cmd_train)

    mode_eval = sub.add_parser('eval')
    mode_eval.set_defaults(handler=cmd_eval)

    mode_help = sub.add_parser('help')
    mode_help.add_argument('cmd')
    mode_help.set_defaults(handler=cmd_help)

    args = parser.parse_args()
    if hasattr(args, 'handler'):
        args.handler(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
