import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel


# =====================================================================
# 1. АРХИТЕКТУРА МОДЕЛИ (идентична train.py для загрузки весов)
# =====================================================================

class RuBertToxicDetector(nn.Module):
    def __init__(
        self,
        model_name: str = "DeepPavlov/rubert-base-cased",
        num_labels: int = 11,
        num_layers_pool: int = 4,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.rubert = AutoModel.from_pretrained(model_name, config=self.config, attn_implementation="eager")

        self.num_layers_pool = num_layers_pool
        self.layer_weights = nn.Parameter(torch.ones(num_layers_pool))

        self.dropouts = nn.ModuleList([nn.Dropout(0.2) for _ in range(5)])
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.rubert(input_ids=input_ids, attention_mask=attention_mask)
        top_hidden_states = outputs.hidden_states[-self.num_layers_pool:]

        weights = torch.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
        stacked = torch.stack(top_hidden_states, dim=0)
        pooled_output = torch.sum(weights * stacked, dim=0)

        # В режиме model.eval() слои Dropout не изменяют тензор,
        # но сохраняем структуру графа для точного совпадения весов
        logits = torch.mean(
            torch.stack([self.classifier(drop(pooled_output)) for drop in self.dropouts], dim=0),
            dim=0,
        )
        return logits


# =====================================================================
# 2. КЛАСС ПРЕДИКТОРА И ДЕКОДИРОВАНИЕ BIO -> (start, end, category)
# =====================================================================

class ToxicSpanPredictor:
    def __init__(self, model_dir: str = "models/rubert_toxic_detector", device: str = None):
        self.model_dir = Path(model_dir)
        self.ckpt_path = self.model_dir / "best_model.pt"
        self.tokenizer_dir = self.model_dir / "tokenizer"

        if not self.ckpt_path.exists():
            raise FileNotFoundError(f"Чекпоинт не найден: {self.ckpt_path.resolve()}")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Загружаем чекпоинт и метаданные
        checkpoint = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        self.model_name = checkpoint.get("model_name", "DeepPavlov/rubert-base-cased")
        self.max_len = checkpoint.get("max_len", 256)
        self.id2label = {int(k): v for k, v in checkpoint["id2label"].items()}
        self.label2id = checkpoint["label2id"]

        # Загружаем локальный токенизатор (или из HuggingFace Hub, если папка отсутствует)
        tokenizer_source = str(self.tokenizer_dir) if self.tokenizer_dir.exists() else self.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)

        # Инициализируем модель и загружаем веса
        self.model = RuBertToxicDetector(
            model_name=self.model_name,
            num_labels=len(self.id2label),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        val_f1 = checkpoint.get("val_metrics", {}).get("token_macro_f1", None)
        f1_str = f" | Val Macro F1: {val_f1:.4f}" if val_f1 is not None else ""
        print(f"[Predictor] Модель загружена с {self.device} (Эпоха: {checkpoint.get('epoch')}{f1_str})")

    @staticmethod
    def _decode_tokens_to_spans(
        text: str,
        offsets: List[List[int]],
        pred_ids: List[int],
        probs: List[float],
        id2label: Dict[int, str],
        include_confidence: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Переводит последовательность BIO-меток токенов в символьные спаны.
        Корректно склеивает WordPiece-подтокены одного слова, даже если модель
        предсказала I-tag с начала слова или два B-tag подряд внутри одного слова.
        """
        entities = []
        current_ent = None

        for (start, end), pred_id, prob in zip(offsets, pred_ids, probs):
            # Пропускаем спецтокены ([CLS], [SEP], [PAD])
            if start == 0 and end == 0:
                continue

            tag = id2label[int(pred_id)]

            if tag == "O":
                if current_ent is not None:
                    entities.append( ToxicSpanPredictor._finalize_entity(text, current_ent, include_confidence) )
                    current_ent = None
                continue

            prefix, cat = tag.split("-", 1)

            if current_ent is not None:
                # Проверяем, является ли токен продолжением текущей сущности:
                # 1) Та же категория и префикс I-
                # 2) Или токен вплотную примыкает к предыдущему (подтокен того же слова без пробела) с той же категорией
                is_same_cat = (current_ent["category"] == cat)
                is_subword_continuation = (start == current_ent["end"])

                if is_same_cat and (prefix == "I" or is_subword_continuation):
                    current_ent["end"] = end
                    current_ent["probs"].append(prob)
                else:
                    entities.append( ToxicSpanPredictor._finalize_entity(text, current_ent, include_confidence) )
                    current_ent = {"start": start, "end": end, "category": cat, "probs": [prob]}
            else:
                # Открываем новую сущность (даже если модель сразу выдала I-cat без B-cat)
                current_ent = {"start": start, "end": end, "category": cat, "probs": [prob]}

        if current_ent is not None:
            entities.append( ToxicSpanPredictor._finalize_entity(text, current_ent, include_confidence) )

        return entities

    @staticmethod
    def _finalize_entity(text: str, ent_data: Dict[str, Any], include_confidence: bool) -> Dict[str, Any]:
        start, end = ent_data["start"], ent_data["end"]
        result = {
            "span": text[start:end],
            "start": start,
            "end": end,
            "category": ent_data["category"],
        }
        if include_confidence:
            avg_prob = sum(ent_data["probs"]) / len(ent_data["probs"])
            result["confidence"] = round(float(avg_prob), 4)
        return result

    @torch.no_grad()
    def predict_batch(self, texts: List[str], include_confidence: bool = False) -> List[Dict[str, Any]]:
        
        print("[DEBUG] predict_batch вызвана", flush=True)
        
        """Предсказание для списка текстов."""
        if not texts:
            return []

        encoding = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_len,
            padding=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        offsets_batch = encoding["offset_mapping"].tolist()

        print(
            f"[DEBUG] Устройство: {self.device}",
            flush=True
        )

        print(
            f"[DEBUG] Размер входных данных: {input_ids.shape}",
            flush=True
        )

        if self.device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
        else:
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        probabilities = torch.softmax(logits, dim=-1)
        max_probs, pred_ids = torch.max(probabilities, dim=-1)

        pred_ids_batch = pred_ids.cpu().tolist()
        max_probs_batch = max_probs.cpu().tolist()

        results = []
        for idx, (text, offsets, preds, probs) in enumerate(
            zip(texts, offsets_batch, pred_ids_batch, max_probs_batch)
        ):
            entities = self._decode_tokens_to_spans(
                text=text,
                offsets=offsets,
                pred_ids=preds,
                probs=probs,
                id2label=self.id2label,
                include_confidence=include_confidence,
            )
            results.append({
                "id": idx,
                "text": text,
                "entities": entities,
            })
            
        print(
            f"[DEBUG] Forward завершён. Logits shape: {logits.shape}",
            flush=True
        )

        return results

    def predict_text(self, text: str, include_confidence: bool = False) -> Dict[str, Any]:
        """Предсказание для одной строки."""
        return self.predict_batch([text], include_confidence=include_confidence)[0]


# =====================================================================
# 3. УТИЛИТА ДЛЯ НАГЛЯДНОГО ВЫВОДА В КОНСОЛЬ
# =====================================================================

def format_highlighted_text(text: str, entities: List[Dict[str, Any]]) -> str:
    """Форматирует текст с подсветкой найденных спанов: [спан | категория]."""
    if not entities:
        return text

    # Сортируем с конца, чтобы вставка скобок не сбивала символьные индексы
    highlighted = text
    for ent in sorted(entities, key=lambda x: x["start"], reverse=True):
        s, e, cat = ent["start"], ent["end"], ent["category"]
        conf_str = f" ({ent['confidence']:.2f})" if "confidence" in ent else ""
        replacement = f"[{highlighted[s:e]} | {cat}{conf_str}]"
        highlighted = highlighted[:s] + replacement + highlighted[e:]
    return highlighted


# =====================================================================
# 4. ТОЧКА ВХОДА (ДЕМО, ОДИНОЧНЫЙ ТЕКСТ И ИНТЕРАКТИВНЫЙ РЕЖИМ)
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Инференс модели детекции токсичных паттернов")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models/rubert_toxic_detector",
        help="Путь к директории с best_model.pt и папкой tokenizer/",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Одиночный текст для проверки через командную строку",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Запустить интерактивный режим ввода текстов в консоли",
    )
    parser.add_argument(
        "--show_confidence",
        action="store_true",
        help="Добавлять уверенность модели (confidence) в вывод сущностей",
    )
    args = parser.parse_args()

    predictor = ToxicSpanPredictor(model_dir=args.model_dir)

    # Режим 1: Одиночный текст из аргумента --text
    if args.text:
        result = predictor.predict_text(args.text, include_confidence=args.show_confidence)
        print("\nРазмеченный текст:")
        print(" ", format_highlighted_text(result["text"], result["entities"]))
        print("\nJSON-ответ:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    # Режим 2: Интерактивный ввод в терминале
    if args.interactive:
        print("\n=== Интерактивный режим (введите 'exit' или 'q' для выхода) ===")
        while True:
            try:
                user_text = input("\nВведите текст: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nВыход.")
                break

            if not user_text:
                continue
            if user_text.lower() in {"exit", "q", "quit", "выход"}:
                break

            result = predictor.predict_text(user_text, include_confidence=args.show_confidence)
            print("Разметка :", format_highlighted_text(result["text"], result["entities"]))
            print("JSON     :", json.dumps(result, ensure_ascii=False))
        return

    # Режим 3 (по умолчанию): Прогон на встроенных тестовых примерах
    demo_texts = [
        "Я убью тебя и всю твою семью",
        "Все негры должны быть убиты",
        "Он просто хуесос",
        "Блять, когда же это закончится"
    ]

    print("\n=== Демонстрационный прогон на собственных примерах ===")
    results = predictor.predict_batch(demo_texts, include_confidence=args.show_confidence)

    for res in results:
        print("-" * 70)
        print("Исходный текст :", res["text"])
        print("Визуализация   :", format_highlighted_text(res["text"], res["entities"]))
        print("JSON-формат    :", json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()