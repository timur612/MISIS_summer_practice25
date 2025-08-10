import pandas as pd
import numpy as np
import torch
from torchmetrics import F1Score, Precision, Recall
from transformers import AutoTokenizer, AutoModelForTokenClassification

from make_dataset import NERFormulaDataset

def evaluate_model(model, tokenizer, test_dataset, id2label):
    model.eval()
    test_f1_score = F1Score(task="multiclass", num_classes=3, average="weighted")
    test_precision = Precision(task="multiclass", num_classes=3, average="weighted")
    test_recall = Recall(task="multiclass", num_classes=3, average="weighted")

    label_names = ["O", "B-TERM", "I-TERM"] # Define label_names here
    label_map = {"O": 0, "B-TERM": 1, "I-TERM": 2} # Define label_map here

    for i, tokenized_input in enumerate(test_dataset):
        input_ids, attention_mask, labels = tokenized_input['input_ids'], tokenized_input['attention_mask'], tokenized_input['labels']

        with torch.no_grad():
            outputs = model(input_ids=input_ids.unsqueeze(0).to('cuda'), attention_mask=attention_mask.unsqueeze(0).to('cuda'))
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=2)
            predictions_list = predictions.detach().cpu().numpy()[0].tolist()
            labels_list = labels.detach().cpu().numpy().tolist()


            true_labels = [label_map[label_names[l]] for l in labels_list if l != -100]
            true_predictions = [
                label_map[label_names[p]] for (p, l) in zip(predictions_list, labels_list) if l != -100
            ]

            # Check for mismatches and print the original text and labels if there's a difference
            if true_labels != true_predictions:
                original_text = test_dataset.df.iloc[i]['text']
                original_formula_text = test_dataset.df.iloc[i]['formula_phrases']


                tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
                predicted_tags = [id2label[p] for p in predictions_list]

                formula_phrases = []
                current_phrase_tokens = []
                for token, tag in zip(tokens, predicted_tags):
                    if token == "[PAD]": # Ignore PAD tokens
                        continue
                    if tag == "B-TERM":
                        if current_phrase_tokens:
                            formula_phrases.append(tokenizer.convert_tokens_to_string(current_phrase_tokens))
                        current_phrase_tokens = [token]
                    elif tag == "I-TERM":
                        current_phrase_tokens.append(token)
                    else:
                        if current_phrase_tokens:
                            formula_phrases.append(tokenizer.convert_tokens_to_string(current_phrase_tokens))
                            current_phrase_tokens = []
                if current_phrase_tokens:
                    formula_phrases.append(tokenizer.convert_tokens_to_string(current_phrase_tokens))


                print(f"Original Text: {original_text}")
                print(f"True Formula Phareses: {original_formula_text}")
                print(f"Predicted Formula Phrases: {formula_phrases}\n")


            test_f1_score.update(torch.tensor(true_predictions), torch.tensor(true_labels)) # Pass 1D tensors
            test_precision.update(torch.tensor(true_predictions), torch.tensor(true_labels)) # Pass 1D tensors
            test_recall.update(torch.tensor(true_predictions), torch.tensor(true_labels)) # Pass 1D tensors


    test_f1 = test_f1_score.compute()
    test_precision = test_precision.compute()
    test_recall = test_recall.compute()

    print(f"Test F1:{test_f1}")
    print(f"Test Precision:{test_precision}")
    print(f"Test Recall:{test_recall}")

def main():
    label_names = ["O", "B-TERM", "I-TERM"]
    model_name = "google-bert/bert-base-multilingual-cased"  # original model name
    model_checkpoint = "best_model.pth"  # model checkpoint
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {v: k for k, v in id2label.items()}

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        id2label=id2label,
        label2id=label2id,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    test = pd.read_csv("data/splited/test_formula_ner.csv")
    test_dataset = NERFormulaDataset(test)
    evaluate_model(model, tokenizer, test_dataset, id2label)

if __name__ == "__main__":
    main()