import torch
import torchtext


class IntentField(torchtext.data.Field):
    def __init__(self):
        super(IntentField, self).__init__()
        self.use_vocab = False
        self.batch_first = True
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


class PlaceField(torchtext.data.Field):
    def __init__(self):
        super(PlaceField, self).__init__()
        self.use_vocab = False
        self.batch_first = True
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


class DatetimeField(torchtext.data.Field):

    def __init__(self):
        super(DatetimeField, self).__init__()
        self.use_vocab = False
        self.batch_first = True
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


class TextField(torchtext.data.Field):
    def __init__(self, extractor):
        super(TextField, self).__init__()
        self.use_vocab = False
        self.batch_first = True
        self.dtype = torch.float
        self.tokenize = lambda text: extractor.extract(text)


class Dataset(torchtext.data.TabularDataset):
    def __init__(self, extractor, path):
        super(Dataset, self).__init__(path=path, format='tsv', fields=[
            ('intent', IntentField()),  ('place', PlaceField()), ('datetime', DatetimeField()), ('text', TextField(extractor))])
