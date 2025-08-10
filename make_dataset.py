import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding

def create_ner_labels(text, formula_phrases):
    """Создает метки NER в формате BIO для текста"""
    # Инициализируем все метки как 'O'
    labels = ['O'] * len(text)

    # Разделяем формулы на отдельные фразы
    phrases = [phrase.strip() for phrase in formula_phrases.split(',')]

    for phrase in phrases:
        start_idx = text.find(phrase)
        if start_idx != -1:
            # Отмечаем начало фразы как B-FORMULA
            labels[start_idx] = 'B-FORMULA'
            # Отмечаем остальные символы фразы как I-FORMULA
            for i in range(start_idx + 1, start_idx + len(phrase)):
                if i < len(labels):
                    labels[i] = 'I-FORMULA'

    return labels

def align_labels_with_tokens(text, labels, tokenizer):
    """Сопоставляет символьные метки с токенами после токенизации"""
    tokenized_inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        is_split_into_words=False,
        return_offsets_mapping=True,
        return_tensors="pt"
    )

    # Извлекаем offset_mapping - позиции символов для каждого токена
    offset_mapping = tokenized_inputs.pop("offset_mapping")[0]

    # Создаем метки для токенов
    token_labels = []
    prev_word_idx = -1

    for offset in offset_mapping:
        if offset[0] == 0 and offset[1] == 0:
            # Специальные токены (CLS, SEP, PAD)
            token_labels.append(-100)  # Игнорируем при обучении
        elif offset[0] == offset[1]:
            # Токены без текста (пробелы и т.д.)
            token_labels.append(-100)
        else:
            # Проверяем, какой метке соответствует начало токена
            if labels[offset[0]] == 'B-FORMULA':
                token_labels.append(1)  # B-FORMULA
            elif labels[offset[0]] == 'I-FORMULA':
                token_labels.append(2)  # I-FORMULA
            else:
                token_labels.append(0)  # O

    return tokenized_inputs, token_labels

class NERFormulaDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.input_ids = []
        self.attention_masks = []
        self.all_labels = []

        self._tokenize_data()
        self.encodings = {
            'input_ids': torch.tensor(self.input_ids),
            'attention_mask': torch.tensor(self.attention_masks)
        }
        self.labels = torch.tensor(self.all_labels)

    def _tokenize_data(self):
      for _, row in tqdm(self.df.iterrows(), total=len(self.df)):
        text = row['text']
        char_labels = row['bio_labels']

        tokenized_inputs, token_labels = align_labels_with_tokens(text, char_labels, tokenizer)

        self.input_ids.append(tokenized_inputs["input_ids"].squeeze().tolist())
        self.attention_masks.append(tokenized_inputs["attention_mask"].squeeze().tolist())
        self.all_labels.append(token_labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class FormulaDataCollator(DataCollatorWithPadding):
    def __init__(self, tokenizer, padding=True, max_length=None, pad_to_multiple_of=None):
        super().__init__(tokenizer, padding, max_length, pad_to_multiple_of)

    def __call__(self, features):
        batch = super().__call__(features)

        max_len = batch['input_ids'].shape[1]

        labels = []
        for feature in features:
            label = feature['labels'].tolist()
            padded_label = label + [-100] * (max_len - len(label))
            labels.append(padded_label)

        batch['labels'] = torch.tensor(labels)
        return batch
    
if __name__ == "__main__":
    # Пример использования
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
    data_collator = FormulaDataCollator(tokenizer)

    train = pd.read_csv("data/splited/train_formula_ner.csv")
    val = pd.read_csv("data/splited/val_formula_ner.csv")
    test = pd.read_csv("data/splited/test_formula_ner.csv")
    
    train_dataset = NERFormulaDataset(train)
    val_dataset = NERFormulaDataset(val)
    test_dataset = NERFormulaDataset(test)

    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")