import sys
import time
import random
import torch
from torch import nn
from torch import optim
from torch.functional import F
import torchtext

from extractor import Extractor


class Cataloger(nn.Module):
    def __init__(self, catalog_features, hidden_units=768):
        super(Cataloger, self).__init__()

        self.l1 = nn.Linear(in_features=768, out_features=hidden_units)
        self.l2 = nn.Linear(in_features=hidden_units,
                            out_features=hidden_units)
        self.l3 = nn.Linear(in_features=hidden_units,
                            out_features=catalog_features)

    def forward(self, distributed_representation):
        x = F.relu(self.l1(distributed_representation))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.l2(x))
        return self.l3(x)


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
                inputs = data.text.to(device)
                labels = getattr(data, target).to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    labels = labels.squeeze(1)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # 訓練時はバックプロパゲーション
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                        if (iteration % 10 == 0):  # 10iterに1度、lossを表示
                            t_iter_finish = time.time()
                            duration = t_iter_finish - t_iter_start
                            acc = (torch.sum(preds == labels.data)
                                   ).double()/batch_size
                            print('イテレーション {} || Loss: {:.4f} || 10iter: {:.4f} sec. || 本イテレーションの正解率：{}'.format(
                                iteration, loss.item(), duration, acc))
                            t_iter_start = time.time()

                    iteration += 1

                    # 損失と正解数の合計を更新
                    epoch_loss += loss.item() * batch_size
                    epoch_loss += loss.item()
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


class IntentField(torchtext.data.Field):
    def __init__(self):
        super(IntentField, self).__init__()
        self.use_vocab = False
        self.batch_first=True
        self.labels = {
            'unknown': 0,
            'other': 1,
            'weather': 2,
            'schedule': 3
        }
        self.tokenize = self.tokenize_to_array

    def tokenize_to_array(self, label):
        label = label if label in self.labels else "unknown"
        return [self.labels[label]]
        # label_index = self.labels[label]
        # return [1 if i == label_index else 0 for i in range(len(self.labels))]


class DatetimeField(torchtext.data.Field):
    def __init__(self):
        super(DatetimeField, self).__init__()
        self.use_vocab = False
        self.batch_first=True
        self.labels = {
            "unknown": 0,
            "today": 1,
            "this_morning": 2,
            "daytime": 3,
            "early_evening": 4,
            "tonight": 5,
            "tomorrow": 6,
            "day_after_next": 7,
            "next_week": 8,
            "next_month": 9,
            "now": 10,
            "next_year": 11,
            "absolute": 12
        }
        self.tokenize = self.tokenize_to_array

    def tokenize_to_array(self, label):
        label = label if label in self.labels else "unknown"
        return [self.labels[label]]
        # label_index = self.labels[label]
        # return [1 if i == label_index else 0 for i in range(len(self.labels))]


class PlaceField(torchtext.data.Field):
    def __init__(self):
        super(PlaceField, self).__init__()
        self.use_vocab = False
        self.batch_first=True
        self.labels = {
            "unknown": 0,
            "here": 1,
            "working_place": 2,
            "school": 3
        }
        self.tokenize = self.tokenize_to_array

    def tokenize_to_array(self, label):
        label = label if label in self.labels else "unknown"
        return [self.labels[label]]
        # label_index = self.labels[label]
        # return [1 if i == label_index else 0 for i in range(len(self.labels))]


class TextField(torchtext.data.Field):
    def __init__(self, extractor):
        super(TextField, self).__init__()
        self.use_vocab = False
        self.batch_first=True
        self.dtype = torch.float
        self.tokenize = lambda text: extractor.extract(text)


class Dataset(torchtext.data.TabularDataset):
    def __init__(self, extractor, path):
        super(Dataset, self).__init__(path=path, format='tsv', fields=[
            ('intent', IntentField()),  ('place', PlaceField()), ('datetime', DatetimeField()), ('text', TextField(extractor))])


def main():
    args = sys.argv
    extractor = Extractor('../Japanese_L-12_H-768_A-12_E-30_BPE/')
    td = Dataset(extractor, path=args[1])
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
        train(net, target, dataloaders_dict, criterion, optimizer, 20)


if __name__ == '__main__':
    main()
