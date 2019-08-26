from pyknp import Juman

import torch

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

class Extractor:
    def __init__(self, bert_model):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.bert_model = bert_model
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model)
        self.model = BertModel.from_pretrained(self.bert_model)
        self.model.to(self.device)
        self.jm = Juman()

    def extract(self, text, seq_length=256):
        assert self.tokenizer != None

        jm_result = self.jm.analysis(text)
        tokens = self.tokenizer.tokenize(
            " ".join([m.midasi for m in jm_result.mrph_list()]))
        if len(tokens) > seq_length - 2:
            tokens = tokens[0:(seq_length - 2)]
        tokens.insert(0, "[CLS]")
        tokens.append("[SEP]")
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)

        self.model.eval()

        input_ids = torch.tensor([input_ids], dtype=torch.long)
        input_mask = torch.tensor([input_mask], dtype=torch.long)

        input_ids = input_ids.to(self.device)
        input_mask = input_mask.to(self.device)

        _, encoded = self.model(
            input_ids, token_type_ids=None, attention_mask=input_mask)
        encoded = encoded.detach().cpu().numpy()

        return encoded
