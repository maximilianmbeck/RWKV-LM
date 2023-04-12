import json

import numpy as np
import torch
from torch.utils.data import Dataset


class EnWik8(Dataset):

    def __init__(self,
                 datafile: str,
                 context_length: int = 512,
                 epoch_steps: int = 5000,
                 batch_size: int = 12,
                 encoding: str = 'utf-8',
                 seed: int = 0,
                 **kwargs):
        super().__init__()
        with open(datafile, 'r', encoding=encoding) as f:
            self.data = f.read()
        self.context_length = context_length
        self._unique_chars = sorted(list(set(self.data)))
        self.vocab_size = len(self._unique_chars)
        self.char2idx = {c: i for i, c in enumerate(self._unique_chars)}
        self.idx2char = {i: c for i, c in enumerate(self._unique_chars)}
        self.data_size = len(self.data)
        print(
            f"Data has {self.data_size} tokens, {self.vocab_size} vocab size.")

        self._seed = seed
        self._np_rng = np.random.default_rng(self._seed)

        self._epoch_steps = epoch_steps
        self._batch_size = batch_size

    def _dump_vocab(self, vocab_file: str) -> None:
        # dump vocabulary to json file:
        with open(vocab_file, 'w', encoding="utf-16le") as f:
            json.dump(self._unique_chars, f, ensure_ascii=False)

    def __len__(self):
        return self._epoch_steps * self._batch_size

    def __getitem__(self, idx):
        req_len = self.context_length + 1
        i = self._np_rng.integers(0, self.data_size - req_len)
        dix = [self.char2idx[c] for c in self.data[i:i + req_len]]

        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y
