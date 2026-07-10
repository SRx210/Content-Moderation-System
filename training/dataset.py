from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer
from transformers import AutoTokenizer

LABEL_MAP = {
    'ALLOW': 0,
    'FLAG': 1,
    'BLOCK': 2
}

class ModerationDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=512):
        self.df = pd.read_csv(csv_path)
        self.comments = self.df['comment_text'].fillna("").tolist()
        self.labels = [LABEL_MAP[label] for label in self.df['moderation_label'].tolist()]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text = str(self.comments[idx])
        encoding = self.tokenizer(text, padding="max_length", truncation=True, return_tensors="pt", return_attention_mask=True, max_length=self.max_length)
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }
    
def get_tokenizer():
    return AutoTokenizer.from_pretrained("distilbert-base-uncased")