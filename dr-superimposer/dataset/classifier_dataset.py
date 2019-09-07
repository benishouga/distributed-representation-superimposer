import torch
import torchtext

from .fields import IntentField
from .fields import PlaceField
from .fields import DatetimeField
from .fields import TextField
from .fields import DrField


class ClassifierDataset(torchtext.data.TabularDataset):
    def __init__(self, path, text_holder):
        super(ClassifierDataset, self).__init__(path=path, format='tsv', fields=[
            ('intent', IntentField()),
            ('place', PlaceField()),
            ('datetime', DatetimeField()),
            ('text', TextField(text_holder)),
            ('dr', DrField())
        ])
