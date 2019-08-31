import torch
import torchtext

from .fields import IntentField
from .fields import PlaceField
from .fields import DatetimeField
from .fields import TextField
from .fields import DrField


class CatalogerDataset(torchtext.data.TabularDataset):
    def __init__(self, path):
        super(CatalogerDataset, self).__init__(path=path, format='tsv', fields=[
            ('intent', IntentField()),
            ('place', PlaceField()),
            ('datetime', DatetimeField()),
            ('text', TextField()),
            ('dr', DrField())
        ])
