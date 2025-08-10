import pandas as pd
import numpy as np
import json
from openai import OpenAI
from langchain.prompts import PromptTemplate

api_key = "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" 

llm_client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=api_key,
)

prompt_template = PromptTemplate.from_template(
    """

    Ты являешься экспертом по анализу математических текстов. Твоя задача - определить, какие слова в предложении относятся к описанию заданной математической формулы.
    Правила определения
        1. Синтаксическое подчинение: Слово должно быть синтаксически подчинено формуле. Это означает, что формула должна задавать вопрос к этому слову.
        2. Отсутствие описания: У формулы может не быть текстового описания. В этом случае результат должен быть пустой строкой.
        3. Взаимное отношение формул: Другая формула может относиться к формуле как её описание.

    Примеры выделения слов из предложений:
    Пример 1:
    Предложение: "Переменная yk принимает значение 0, если ограничение k неактивно в задаче линейного программирования."
    Правильный ответ: ["Переменная yk", "ограничение k"]

    Пример 2:
    Предложение: "Бинарная переменная zij равна 1 при ∥pi −pj ∥2 ≤ d и дуги(i, j)."
    Правильный ответ: ["Бинарная переменная zij", "дуга(i, j)"]

    Пример 3:
    Предложение: "Если скалярное произведение ⟨a, x⟩ больше b верно, то задача переключается с min cT x на min∥x∥22 с дополнительным ограничением Dx ≤ h."
    Правильный ответ: ["скалярное произведение ⟨a, x⟩", "задача min cT x", "задача min∥x∥22", "ограничение Dx ≤ h"]

    Пример 4:
    Предложение: "Флаг tmn принимает 1, если \|\|mn − nm\|\|2 ≤ p и соединение(m, n) установлено."
    Правильный ответ: ["Флаг tmn", "соединение(m, n)"]

    Пример 5:
    Предложение: "Переменная αt переключается с режима min ft(x) на max gt(y), когда параметр t пересекает границу t0 или нарушается условие ⟨wt, v⟩ ≥ δ."
    Правильный ответ: ["Переменная αt", "режим min ft(x)", "условие ⟨wt, v⟩ ≥ δ"]

    Твоя задача - выделить слова, которые относятся к формуле в предложении: {sentence}
    Ответ должен быть в формате JSON, где ключ - это "formula_phrases", а значение - список слов, которые относятся к формуле.
    Пример ответа: "formula_phrases": ["слово1", "слово2", ...]
    Ответ должен быть в формате JSON, без дополнительных пояснений или текста.
"""
    )

def get_completion(sentence, model_name="qwen/qwen3-235b-a22b:free"):
    model = model_name
    prompt = prompt_template.format(sentence=sentence)
    completion = llm_client.chat.completions.create(
        model=model,
        messages=[
            {
            "role": "user",
            "content": prompt
            }
        ]
    )
    return completion.choices[0].message.content

def normalize_phrase(phrase):
    return phrase.lower().replace("∥", "||").replace(" ", "")

def evaluate_llm():
    test = pd.read_csv("data/splited/test_formula_ner.csv")
    total_gt, total_pred, total_correct = 0, 0, 0
    for idx, row in test.iterrows():
        sentence = row['text']
        try:
            gt_phrases = [phrase.strip() for phrase in row['formula_phrases'].split(',')] if row['formula_phrases'] else []
        except Exception:
            gt_phrases = []
        response = get_completion(sentence, model_name="qwen/qwen3-30b-a3b:free")
        try:
            pred_data = json.loads(response.strip())
            pred_phrases = [p.strip() for p in pred_data.get("formula_phrases", [])]
        except Exception as e:
            print(f"JSON parse error: {e}")
            pred_phrases = []
        
        normalized_gt = [normalize_phrase(p) for p in gt_phrases]
        normalized_pred = [normalize_phrase(p) for p in pred_phrases]

        correct = sum(1 for p in normalized_pred if p in normalized_gt)
        
        total_gt += len(normalized_gt)
        total_pred += len(normalized_pred)
        total_correct += correct

        if len(gt_phrases) != correct:
            print(f"Sentence {idx}: GT={gt_phrases}, Pred={pred_phrases} (Correct: {correct})")

    precision = total_correct / total_pred if total_pred > 0 else 0
    recall = total_correct / total_gt if total_gt > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nTOTAL: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

if __name__ == "__main__":
    evaluate_llm()