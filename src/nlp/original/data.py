from os import cpu_count

import torch
from datasets import load_dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class WMT14Subset(Dataset):
    def __init__(self, subset, tokenizer, max_len=128, languages="de-en"):
        super(WMT14Subset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.lang1, self.lang2 = languages.split("-")
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        return self.preprocess(self.subset[index])

    def preprocess(self, sample):
        # Getting sentences
        s1 = sample["translation"][self.lang1]
        s2 = sample["translation"][self.lang2]

        # Tokenizing sentencens
        enc_tok = self.tokenizer(
            s1,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
        )
        dec_tok = self.tokenizer(
            s2,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
        )

        # Unpacking return values
        x_enc = enc_tok["input_ids"][0]
        x_dec = dec_tok["input_ids"][0]
        enc_attn_mask = enc_tok["attention_mask"][0]
        dec_attn_mask = torch.tril(torch.ones(self.max_len, self.max_len))
        dec_attn_mask[:, x_dec == self.tokenizer.pad_token_id] = 0

        return {
            "x_enc": x_enc,
            "x_dec": x_dec,
            "enc_attn_mask": enc_attn_mask,
            "dec_attn_mask": dec_attn_mask,
            "enc_dec_attn_mask": enc_attn_mask,
        }


class WMT14Dataset(LightningDataModule):
    def __init__(self, tokenizer, batch_size=32, max_len=128, languages="de-en"):
        super(WMT14Dataset, self).__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_len = max_len
        self.languages = languages

    def prepare_data(self):
        self.wmt14 = load_dataset("wmt14", self.languages)

    def setup(self, stage):
        self.train = WMT14Subset(
            self.wmt14["train"], self.tokenizer, self.max_len, self.languages
        )
        self.val = WMT14Subset(
            self.wmt14["validation"], self.tokenizer, self.max_len, self.languages
        )
        self.test = WMT14Subset(
            self.wmt14["test"], self.tokenizer, self.max_len, self.languages
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=cpu_count(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val, batch_size=self.batch_size, shuffle=False, num_workers=cpu_count()
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=cpu_count(),
        )

    def teardown(self, stage):
        pass
