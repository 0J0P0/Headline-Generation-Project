# src/data_loader.py
import pandas as pd
from torch.utils.data import Dataset

from configs.settings import MAX_INPUT_LEN, MAX_TARGET_LEN


class HeadlineDataset(Dataset):
    def __init__(
        self,
        csv_path,
        tokenizer,
        max_input_len=MAX_INPUT_LEN,
        max_target_len=MAX_TARGET_LEN,
    ):
        self.data = pd.read_csv(csv_path, sep=";")
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        source = self.data.iloc[idx]["input_text"]
        target = self.data.iloc[idx]["target_text"]

        # Tokenize input
        source_enc = self.tokenizer(
            source,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize target/label
        target_enc = self.tokenizer(
            target,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": source_enc["input_ids"].squeeze(0),
            "attention_mask": source_enc["attention_mask"].squeeze(0),
            "labels": target_enc["input_ids"].squeeze(0),
        }
