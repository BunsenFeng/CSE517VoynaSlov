import sys
import torch
import torch.nn as nn
import dataloader
from tqdm import tqdm
import os
import pytorch_lightning as pl
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, balanced_accuracy_score
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForPreTraining, AutoModelWithLMHead, AutoModel, AutoModelForSequenceClassification, BartForSequenceClassification
import numpy as np
import json
import random

model_name = "xlm-roberta-large"
corpus_name = "none"
num_runs = 1
dataset_name = "tobacco_filtered"
batch_size = 64
num_epoch = 20
voyna = False

class LMBiasClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        if corpus_name == "none":
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=15)
        self.truth = []
        self.pred = []
        self.id = []

    def forward(self, input):
        result = self.model(**input["input"], labels=input["label"])
        return result
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=5e-6)

    def training_step(self, batch, batch_idx):
        result = self.forward(batch)
        self.log("train_loss", result.loss)
        return result.loss

    def validation_step(self, batch, batch_idx):
        result = self.forward(batch)
        self.log("val_loss", result.loss)
        return result.loss
    
    def test_step(self, batch, batch_idx):
        result = self.forward(batch)
        self.log("test_loss", result.loss)
        self.truth += batch["label"].tolist()
        self.pred += torch.argmax(result.logits, dim=1).tolist()
        self.id += batch["id"]
        return result.loss

    def on_test_end(self):
        f = open("logs/main.txt", "a")
        f.write(model_name + " " + dataset_name + "\n") # normal
        # f.write(final_name + "train_cnndm_test_xsum\n")
        # f.write(final_name + "train_xsum_test_cnndm\n")
        f.write("F1: " + str(f1_score(self.truth, self.pred, average = "macro")) + "\n")
        f.write("Accuracy: " + str(accuracy_score(self.truth, self.pred)) + "\n")
        # f.write("Precision: " + str(precision_score(self.truth, self.pred)) + "\n")
        # f.write("Recall: " + str(recall_score(self.truth, self.pred)) + "\n")
        f.write("Balanced Accuracy: " + str(balanced_accuracy_score(self.truth, self.pred)) + "\n")
        # detailed_result = {"id": self.id, "truth": self.truth, "pred": self.pred}
        # with open("logs/" + final_name + "-" + dataset_name + "-acc:" + str(accuracy_score(self.truth, self.pred))[:6] + ".json", "w") as f2:
        #     json.dump(detailed_result, f2)
        # f.close()

for time in range(num_runs):
    dataset_names = [dataset_name + "_" + str(fold) for fold in range(10)]
    for dataset in dataset_names:

        train_loader, dev_loader, test_loader = dataloader.get_dataloaders(dataset, batch_size, voyna=voyna)
        
        model = LMBiasClassifier()
        trainer = pl.Trainer(max_epochs=num_epoch, gpus=1, gradient_clip_val=1, precision=16, callbacks=[EarlyStopping(monitor="val_loss", patience=5, mode="min"), ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, save_last=True, filename="{epoch}-{val_loss:.2f}")])
        trainer.fit(model, train_loader, dev_loader)
        trainer.test(ckpt_path="best", dataloaders = test_loader)