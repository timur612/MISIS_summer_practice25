from turtle import pd
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification
import torch
from torch.optim import AdamW
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import F1Score, Precision, Recall
import numpy as np

from make_dataset import NERFormulaDataset, FormulaDataCollator

class Trainer:
    def __init__(self, model, args, train_dataset, eval_dataset, data_collator):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.train_min_loss = np.inf
        self.val_min_loss = np.inf

        self.optimizer = AdamW(self.model.parameters(), lr=args.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

    def train(self):
      train_dataloader = DataLoader(
          self.train_dataset,
          shuffle=True,
          collate_fn=self.data_collator,
          batch_size=self.args.per_device_train_batch_size,
      )
      eval_dataloader = DataLoader(
          self.eval_dataset,
          collate_fn=self.data_collator,
          batch_size=self.args.per_device_eval_batch_size,
      )

      for epoch in range(self.args.num_train_epochs):
          self.model.train()
          sum_loss = 0
          for batch in train_dataloader:
              sum_loss += self.train_step(batch)

          train_loss = sum_loss / len(train_dataloader)
          print(f"Epoch {epoch}, Train Loss: {train_loss}")

          self.model.eval()
          eval_f1_score = F1Score(task="multiclass", num_classes=3, average="weighted") # Corrected num_labels and added average
          eval_precision = Precision(task="multiclass", num_classes=3, average="weighted") # Corrected num_labels and added average
          eval_recall = Recall(task="multiclass", num_classes=3, average="weighted") # Corrected num_labels and added average

          label_names = ["O", "B-FORMULA", "I-FORMULA"] # Define label_names here
          sum_val_loss = 0
          for batch in eval_dataloader:
              logits, labels, loss = self.eval_step(batch)
              sum_val_loss += loss
              logits, labels = logits.detach().cpu().numpy(), labels.detach().cpu().numpy()

              predictions = np.argmax(logits, axis=-1)
              true_labels = [[label_map[label_names[l]] for l in label if l != -100] for label in labels] # Use label_map to convert label names to integers
              true_predictions = [
                  [label_map[label_names[p]] for (p, l) in zip(prediction, label) if l != -100] # Use label_map to convert label names to integers
                  for prediction, label in zip(predictions, labels)
              ]

              # Convert lists to tensors before passing to update
              eval_f1_score.update(torch.tensor([item for sublist in true_predictions for item in sublist]), torch.tensor([item for sublist in true_labels for item in sublist]))
              eval_precision.update(torch.tensor([item for sublist in true_predictions for item in sublist]), torch.tensor([item for sublist in true_labels for item in sublist]))
              eval_recall.update(torch.tensor([item for sublist in true_predictions for item in sublist]), torch.tensor([item for sublist in true_labels for item in sublist]))

          eval_f1 = eval_f1_score.compute()
          eval_precision = eval_precision.compute()
          eval_recall = eval_recall.compute()

          val_loss = sum_val_loss / len(eval_dataloader)
          print(f"\n\nEval Loss: {val_loss}\nF1: {eval_f1}\nPrecision: {eval_precision}\nRecall: {eval_recall}")

          if val_loss < self.val_min_loss and train_loss < self.train_min_loss:
            self.val_min_loss = val_loss
            self.train_min_loss = train_loss
            torch.save(self.model.state_dict(), f"best_model.pth")
            print(f"\n\nModel saved on epoch {epoch}")


    def train_step(self, batch):
        batch = {k: v.to(self.model.device) for k, v in batch.items()}
        outputs = self.model(**batch)
        loss = outputs.loss
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def eval_step(self, batch):
        batch = {k: v.to(self.model.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = self.model(**batch)
            loss = outputs.loss
        logits = outputs.logits
        labels = batch["labels"]
        return logits, labels, loss.item()

class TrainingArguments:
    def __init__(self, learning_rate=5e-5, num_train_epochs=10, logging_steps=100,
                 per_device_train_batch_size=8, per_device_eval_batch_size=8):
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.logging_steps = logging_steps
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size


def main():
    label_names=["O", "B-TERM", "I-TERM"]
    model_checkpoint = "google-bert/bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {v: k for k, v in id2label.items()}


    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        id2label=id2label,
        label2id=label2id,
    )
    data_collator = FormulaDataCollator(tokenizer=tokenizer)

    train = pd.read_csv("data/splited/train_formula_ner.csv")
    val = pd.read_csv("data/splited/val_formula_ner.csv")
    test = pd.read_csv("data/splited/test_formula_ner.csv")
    
    train_dataset = NERFormulaDataset(train)
    val_dataset = NERFormulaDataset(val)

    args = TrainingArguments(learning_rate=5e-5,
                         num_train_epochs=15,
                         logging_steps=100,
                         per_device_train_batch_size=8,
                         per_device_eval_batch_size=8
                        )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    trainer.train()

if __name__ == "__main__":
    main()