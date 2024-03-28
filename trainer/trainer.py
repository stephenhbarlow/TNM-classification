from tqdm import tqdm
import torch
import logging
import os
import numpy as np
from models.metric import display_roc_curve, display_confusion_matrix
from sklearn.metrics import classification_report, accuracy_score


class Trainer(object):

    def __init__(self, model, optimizer, args, lr_scheduler, train_dataloader, val_dataloader):

        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.epochs = self.args.epochs
        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def train_epoch(self):

        progress = tqdm(enumerate(self.train_dataloader),
                        total=len(self.train_dataloader))
        self.model = self.model.train()
        losses = []
        correct_predictions = 0
        for i, d in progress:
            input_ids = d["input_ids"].to(self.device)
            attention_mask = d["attention_mask"].to(self.device)
            labels = d["labels"].to(self.device)
            outputs = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask)

            preds = torch.round(torch.sigmoid(outputs)).squeeze()
            loss = self.loss_fn(outputs, labels.unsqueeze(1))
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

        return correct_predictions.float() / len(self.train_dataloader.dataset), np.mean(losses)


    def eval_model(self):

        self.model = self.model.eval()
        losses = []
        probs = []
        predictions = []
        y_test_labels = []

        with torch.no_grad():
            for d in self.val_dataloader:
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                labels = d["labels"].to(self.device)
                logits = self.model(input_ids=input_ids,
                                        attention_mask=attention_mask)
                preds = torch.round(torch.sigmoid(logits))
                predictions.append(preds)
                prob = torch.sigmoid(logits)
                probs.append(prob)
                y_test_labels.append(labels)
                loss = self.loss_fn(logits, labels.unsqueeze(1))
                losses.append(loss.item())
        y_test_labels = torch.cat(y_test_labels).cpu().data.numpy()
        predictions = torch.cat(predictions).squeeze().cpu().data.numpy()
        probs = torch.cat(probs).cpu().data.numpy()

        return {
                "loss": np.mean(losses), 
                "y_labels": y_test_labels, 
                "preds": predictions, 
                "probs": probs,
                }

    def train(self):
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            print('-' * 10)
            train_acc, train_loss = self.train_epoch()
            print(f'Train Loss: {train_loss} Accuracy: {train_acc}')
            eval_dict = self.eval_model()
            val_acc = accuracy_score(eval_dict['y_labels'], eval_dict['preds'])
            print(f"Val   Loss: {eval_dict['loss']} Accuracy: {val_acc}")

            torch.save(self.model.state_dict(),
                        f"{self.checkpoint_dir}single_classifiers/{self.args.target_class}-{self.args.seed}-{epoch+1}of{self.args.epochs}epochs.bin")
        # print(compute_scores(predictions, y_test))
        print(classification_report(eval_dict['y_labels'], eval_dict['preds']))
        display_confusion_matrix(eval_dict['y_labels'],  eval_dict['preds'])
        display_roc_curve(eval_dict['y_labels'], eval_dict['probs'])
