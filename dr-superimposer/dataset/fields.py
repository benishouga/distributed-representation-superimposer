import torch
import torchtext

field_labels = {
    "intent": {
        'unknown': 0,
        'continue': 1,
        'other': 2,
        'weather': 3,
        'schedule': 4
    },
    "place": {
        "unknown": 0,
        "here": 1,
        "working_place": 2,
        "school": 3
    },
    "datetime": {
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
}


class IntentField(torchtext.data.Field):
    def __init__(self):
        super(IntentField, self).__init__()
        self.use_vocab = False
        self.batch_first = True
        self.labels = field_labels["intent"]
        self.tokenize = self.tokenize_to_array

    def tokenize_to_array(self, label):
        label = label if label in self.labels else "unknown"
        return [self.labels[label]]


class PlaceField(torchtext.data.Field):
    def __init__(self):
        super(PlaceField, self).__init__()
        self.use_vocab = False
        self.batch_first = True
        self.labels = field_labels["place"]
        self.tokenize = self.tokenize_to_array

    def tokenize_to_array(self, label):
        label = label if label in self.labels else "unknown"
        return [self.labels[label]]


class DatetimeField(torchtext.data.Field):

    def __init__(self):
        super(DatetimeField, self).__init__()
        self.use_vocab = False
        self.batch_first = True
        self.labels = field_labels["datetime"]
        self.tokenize = self.tokenize_to_array

    def tokenize_to_array(self, label):
        label = label if label in self.labels else "unknown"
        return [self.labels[label]]


class TextField(torchtext.data.Field):
    def __init__(self):
        super(TextField, self).__init__()
        self.use_vocab = False
        self.batch_first = True
        self.tokenize = lambda text: [0]


class DrField(torchtext.data.Field):
    def __init__(self):
        super(DrField, self).__init__()
        self.use_vocab = False
        self.batch_first = True
        self.dtype = torch.float
        self.tokenize = lambda text: [float(v) for v in text.split(",")]
