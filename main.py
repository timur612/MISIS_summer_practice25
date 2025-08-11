import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

label_names = ["O", "B-TERM", "I-TERM"]
model_name = "google-bert/bert-base-uncased"  # original model name
model_checkpoint = "models/best_model.pth"  # model checkpoint
tokenizer = AutoTokenizer.from_pretrained(model_name)

id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}

model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    id2label=id2label,
    label2id=label2id,
)
state_dict = torch.load(model_checkpoint, map_location='cpu')
model.load_state_dict(state_dict)

# Пример использования модели с pipeline
pipeline_ner = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

if __name__ == "__main__":
    df = pd.read_csv("data/formula_dataset.csv")
    
    print(pipeline_ner(df['text'].tolist()[:5]))
