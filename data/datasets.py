import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import pandas as pd


class ReportDataset(Dataset):

    def __init__(self, data, tokenizer, max_len, args):
        self.data = pd.read_csv(data)
        self.len = len(self.data)
        self.args = args
        self.text = self.data['text'].to_numpy()
        self.labels = self.data[self.args.target_class].to_numpy()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, truncation_side='left')
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        label = self.labels[item]
        encoding = self.tokenizer(text,
                                  add_special_tokens=True,
                                  max_length=self.max_len,
                                  return_token_type_ids=False,
                                  truncation=True,
                                  padding='max_length',
                                  return_attention_mask=True,
                                  return_tensors='pt')

        return {'text': text,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.float32)}


class MultiLabelDataset(Dataset):

    def __init__(self, data, tokenizer, max_len):
        self.data = pd.read_csv(data)
        self.len = len(self.data)
        self.attributes = ['tumour', 'node', 'metastasis', 'uncertainty']        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, truncation_side='left')
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        row = self.data.iloc[item]
        labels = torch.FloatTensor(row[self.attributes])
        text = str(row["text"])
        encoding = self.tokenizer(text,
                                  add_special_tokens=True,
                                  max_length=self.max_len,
                                  return_token_type_ids=False,
                                  truncation=True,
                                  padding='max_length',
                                  return_attention_mask=True,
                                  return_tensors='pt')

        return {'text': text,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': labels}
