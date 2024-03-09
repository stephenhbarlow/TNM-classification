import torch
import torch.nn.functional as F
import numpy as np

def eval_model(args, model, dataloader, supervised_criterion):

        model = model.eval()
        probs = []
        predictions = []
        y_test_labels = []
        losses = []
        device = args.device

        with torch.no_grad():
            for batch in dataloader:
                encoding, labels = batch
                input_ids = encoding["input_ids"].to(device)
                attention_mask = encoding["attention_mask"].to(device)
                labels = labels.to(device)
                logits = model(input_ids=input_ids,
                                        attention_mask=attention_mask)
                _, preds = torch.max(logits, dim=1)
                predictions.append(preds)
                prob = F.softmax(logits, dim=1)
                probs.append(prob)
                y_test_labels.append(labels)
                loss = supervised_criterion(logits, labels)
                losses.append(loss.item())
        y_test_labels = torch.cat(y_test_labels).cpu().data.numpy()
        predictions = torch.cat(predictions).squeeze().cpu().data.numpy()
        probs = torch.cat(probs).cpu().data.numpy()

        return {
                "y_labels": y_test_labels, 
                "preds": predictions, 
                "probs": probs,
                "loss": np.mean(losses)
                }