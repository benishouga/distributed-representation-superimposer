import torch
import torchtext

from .fields import IntentField
from .fields import PlaceField
from .fields import DatetimeField
from .fields import TextField
from .fields import DrField


class SuperimposerDataset(torchtext.data.TabularDataset):
    def __init__(self, path, text_holder):
        super(SuperimposerDataset, self).__init__(path=path, format='tsv', fields=[
            ('intent', IntentField()),
            ('place', PlaceField()),
            ('datetime', DatetimeField()),
            ('text1', TextField(text_holder)),
            ('dr1', DrField()),
            ('text2', TextField(text_holder)),
            ('dr2', DrField())
        ])
