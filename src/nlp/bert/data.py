import random

import torch
from torch.utils.data import Dataset


class BertDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length, mlm_ratio=0.15, mask_ratio=0.8):
        super(BertDataset, self).__init__()

        # Dataset parameters
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_ratio = mlm_ratio
        self.mask_ratio = mask_ratio

        # MLM distribution
        nothing_prob = 1 - mlm_ratio
        mask_prob = (1 - nothing_prob) * mask_ratio
        different_word_prob = (1 - nothing_prob - mask_prob) / 2
        probs = torch.tensor(
            [nothing_prob, mask_prob, different_word_prob, different_word_prob]
        )
        self.mask_dist = torch.distributions.Multinomial(probs=probs)

        # Tokenizing dataset
        self.dataset = self.dataset.map(
            lambda samples: {"text": [txt for txt in samples["text"] if len(txt) > 0]},
            batched=True,
        )

    def __len__(self):
        return len(self.dataset) - 1

    def __getitem__(self, index):
        # First sentence
        s1 = self.dataset[index]["text"]

        # Next sentence and prediction label
        if torch.rand(1) > 0.5:
            # 50% of the time, pick the real next "sentence"
            s2 = self.dataset[index + 1]["text"]
            nsp_label = 1
        else:
            # The other 50% of the time, pick a random next "sentence"
            idx = random.randint(0, len(self.dataset) - 1)
            s2 = self.dataset[idx]["text"]
            nsp_label = 0

        # Preparing input ids
        tokenizer_out = self.tokenizer(
            s1,
            s2,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        input_ids, segment_idx, attn_mask = (
            tokenizer_out["input_ids"][0],
            tokenizer_out["token_type_ids"][0],
            tokenizer_out["attention_mask"][0],
        )

        # Getting mask indexes
        mask_idx = self.mask_dist.sample((len(input_ids),))

        # Not masking CLS and SEP
        sep_idx = -1
        for i in range(len(segment_idx)):
            if segment_idx[i] == 1:
                sep_idx = i - 1
                break
        mask_idx[0] = mask_idx[-1] = mask_idx[sep_idx] = torch.tensor([1, 0, 0, 0])
        mask_idx = mask_idx.argmax(dim=-1)

        # Getting labels for masked tokens
        mlm_idx = (mask_idx != 0).long()
        mlm_labels = input_ids.clone()

        # Masking input tokens according to strategy
        input_ids[mask_idx == 1] = self.tokenizer.mask_token_id
        input_ids[mask_idx == 2] = torch.randint(0, self.tokenizer.vocab_size, (1,))

        return {
            "input_ids": input_ids,
            "segment_ids": segment_idx,
            "attn_mask": attn_mask,
            "mlm_labels": mlm_labels,
            "mlm_idx": mlm_idx,
            "nsp_labels": nsp_label,
        }

    def tokenize_fn(self, examples):
        ids = [
            self.tokenizer(txt)["input_ids"] for txt in examples["text"] if len(txt) > 0
        ]
        ids = torch.tensor(ids)
        return ids
