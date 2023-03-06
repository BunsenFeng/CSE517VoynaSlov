import torch
import torch.nn as nn
import torch_geometric
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForPreTraining, AutoModelWithLMHead, AutoModel, AutoModelForSequenceClassification
from tqdm import tqdm
import json

def get_dataloaders(dataset, batch_size, voyna = False):
    # dataset = "climate_filtered_0", "deathpenalty_filtered_9", ...
    train_dataset = PlayDataset(dataset, "train")
    dev_dataset = PlayDataset(dataset, "dev")
    if voyna:
        test_dataset = PlayDataset("voynaslov_0", "test")
    else:
        test_dataset = PlayDataset(dataset, "test")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = 4, collate_fn = train_dataset.pad_collate)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers = 4, collate_fn = dev_dataset.pad_collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers = 4, collate_fn = test_dataset.pad_collate)
    return train_loader, dev_loader, test_loader

class PlayDataset(Dataset):
    def __init__(self, dataset, tdt):
        self.dataset = dataset
        self.tdt = tdt

        self.tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large', padding="max_length", truncation=True)
        self.data = []
        if tdt == "train":
            temp = open("data/mfc_new/" + self.dataset + "_train.json", "r")
        elif tdt == "dev":
            temp = open("data/mfc_new/" + self.dataset + "_dev.json", "r")
        elif tdt == "test":
            temp = open("data/mfc_new/" + self.dataset + "_test.json", "r")
        else:
            raise Exception("Invalid tdt")
        for line in temp:
            obj = json.loads(line)
            self.data.append(obj)
    
    def __len__(self):
        return len(self.data)

    def pad_collate(self, batch):
        texts = []
        labels = []
        ids = []
        for sample in batch:
            texts.append(sample["input"])
            labels.append(sample["label"] - 1) # 0 to 14 for the 15 classes
            ids.append(sample["id"])
        return {
        "input": self.tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=70),
        "label": torch.tensor(labels).long(),
        "id": ids
        }
    
    def __getitem__(self, idx):
        return {
        "input": self.data[idx]["text"],
        "label": self.data[idx]["label"],
        "id": self.data[idx]["id"]
        }

# model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-large", num_labels=15)
# train_loader, dev_loader, test_loader = get_dataloaders("samesex_filtered_4", 32)
# for batch in test_loader:
#     result = model(**batch["input"], labels=batch["label"])
#     print(result) # "loss", "logits"
#     break

# train_loader, dev_loader, test_loader = get_dataloaders("tobacco_filtered_8", 32)
# for batch in test_loader:
#     print(batch["label"].tolist())
#     break