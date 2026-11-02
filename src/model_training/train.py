import os
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    get_cosine_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm


# =====================================================================
# 1. КОНСТАНТЫ И СХЕМА КАТЕГОРИЙ
# =====================================================================

# Список целевых категорий в порядке возрастания редкости (для разрешения конфликтов и стратификации)
TARGET_CATEGORIES = [
    "profanity",           # ~8220
    "insult",              # ~5803
    "derogation_mockery",  # ~381
    "threat_violence",     # ~215
    "hate_speech",         # ~164
]

EXCLUDED_CATEGORIES = {"harassment_doxxing"}

# Приоритет редкости: чем выше число, тем приоритетнее категория при ничьей разметчиков
RARITY_PRIORITY = {cat: idx for idx, cat in enumerate(TARGET_CATEGORIES)}

# BIO-схема меток
LABEL2ID = {"O": 0}
for cat in TARGET_CATEGORIES:
    LABEL2ID[f"B-{cat}"] = len(LABEL2ID)
    LABEL2ID[f"I-{cat}"] = len(LABEL2ID)
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =====================================================================
# 2. ПРЕДОБРАБОТКА, ФИЛЬТРАЦИЯ И СКЛЕЙКА ДУБЛИКАТОВ
# =====================================================================

def load_and_preprocess_dataset(jsonl_path: Path):
    """
    1. Читает JSONL-файл.
    2. Полностью исключает тексты, в которых хотя бы в одном дубликате есть 'harassment_doxxing'.
    3. Склеивает сущности для одинаковых 'text', разрешая конфликты категорий и наложения спанов.
    """
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Датасет не найден по пути: {jsonl_path.resolve()}")

    raw_records = []
    blacklisted_texts = set()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record.get("text", "").strip()
            if not text:
                continue

            entities = record.get("entities", [])
            # Проверяем наличие исключаемой категории
            if any(ent.get("category") in EXCLUDED_CATEGORIES for ent in entities):
                blacklisted_texts.add(text)
                continue

            raw_records.append((text, entities))

    # Группируем сущности по уникальному полю "text"
    # grouped_spans[text][(start, end)] -> Counter(category)
    grouped_spans = defaultdict(lambda: defaultdict(Counter))
    all_valid_texts = set()

    dropped_samples_count = 0
    for text, entities in raw_records:
        if text in blacklisted_texts:
            dropped_samples_count += 1
            continue
        all_valid_texts.add(text)
        for ent in entities:
            cat = ent.get("category")
            if cat not in RARITY_PRIORITY:
                continue
            start, end = int(ent["start"]), int(ent["end"])
            if 0 <= start < end <= len(text):
                grouped_spans[text][(start, end)][cat] += 1

    cleaned_samples = []
    for idx, text in enumerate(sorted(all_valid_texts)):
        spans_dict = grouped_spans[text]
        candidate_spans = []

        for (start, end), cat_counter in spans_dict.items():
            # Выбираем категорию: сначала по числу голосов разметчиков, при равенстве — более редкую
            best_cat = max(
                cat_counter.keys(),
                key=lambda c: (cat_counter[c], RARITY_PRIORITY[c])
            )
            total_votes = sum(cat_counter.values())
            candidate_spans.append({
                "span": text[start:end],
                "start": start,
                "end": end,
                "category": best_cat,
                "votes": total_votes,
            })

        # Разрешаем частичные пересечения спанов (например, [5, 10] и [5, 12]):
        # Приоритет отдаем спанам с большим числом голосов, большей длиной и более редкой категорией
        candidate_spans.sort(
            key=lambda x: (x["votes"], x["end"] - x["start"], RARITY_PRIORITY[x["category"]]),
            reverse=True,
        )

        merged_entities = []
        for cand in candidate_spans:
            overlaps = any(
                max(cand["start"], kept["start"]) < min(cand["end"], kept["end"])
                for kept in merged_entities
            )
            if not overlaps:
                merged_entities.append({
                    "span": cand["span"],
                    "start": cand["start"],
                    "end": cand["end"],
                    "category": cand["category"],
                })

        # Сортируем итоговые сущности по порядку появления в тексте
        merged_entities.sort(key=lambda x: x["start"])

        cleaned_samples.append({
            "id": idx,
            "text": text,
            "entities": merged_entities,
        })

    print(f"[Data] Всего прочитано записей (без harassment_doxxing): {len(raw_records)}")
    print(f"[Data] Удалено текстов из-за harassment_doxxing: {len(blacklisted_texts)}")
    print(f"[Data] Уникальных текстов после склейки дубликатов: {len(cleaned_samples)}")

    return cleaned_samples


def stratified_split(samples, val_size: float = 0.15, seed: int = 42):
    """
    Стратифицированное разбиение по самой редкой категории в семпле.
    """
    strat_labels = []
    for sample in samples:
        ents = sample["entities"]
        if not ents:
            strat_labels.append("clean")
        else:
            # Берем самую редкую категорию в тексте для стратификации
            rarest_cat = max(
                (e["category"] for e in ents),
                key=lambda c: RARITY_PRIORITY[c]
            )
            strat_labels.append(rarest_cat)

    # На случай если какой-то стратификационный класс представлен менее чем 2 примерами
    label_counts = Counter(strat_labels)
    safe_strat_labels = [
        lbl if label_counts[lbl] >= 2 else "clean" for lbl in strat_labels
    ]

    train_samples, val_samples = train_test_split(
        samples,
        test_size=val_size,
        random_state=seed,
        stratify=safe_strat_labels,
    )
    return train_samples, val_samples


# =====================================================================
# 3. TORCH DATASET И ПОДСЧЕТ ВЕСОВ КЛАССОВ
# =====================================================================

class ToxicSpanDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len: int = 256):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        text = item["text"]
        entities = item.get("entities", [])

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        offsets = encoding["offset_mapping"].squeeze(0).tolist()
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        labels = torch.full((self.max_len,), -100, dtype=torch.long)

        for i, (tok_start, tok_end) in enumerate(offsets):
            # Пропускаем спецтокены ([CLS], [SEP], [PAD])
            if tok_start == 0 and tok_end == 0:
                continue

            labels[i] = LABEL2ID["O"]
            for ent in entities:
                e_start, e_end, cat = ent["start"], ent["end"], ent["category"]
                if max(tok_start, e_start) < min(tok_end, e_end):
                    prev_tag = ID2LABEL[labels[i - 1].item()] if (i > 0 and labels[i - 1].item() != -100) else "O"
                    if tok_start <= e_start or prev_tag == "O" or not prev_tag.endswith(cat):
                        labels[i] = LABEL2ID[f"B-{cat}"]
                    else:
                        labels[i] = LABEL2ID[f"I-{cat}"]
                    break

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def compute_damped_class_weights(dataset: ToxicSpanDataset, power: float = 0.35, device: str = "cuda"):
    """
    Считает сглаженные веса классов по частоте токенов в тренировочном датасете.
    Решает проблему дисбаланса (hate_speech vs profanity vs O).
    """
    counts = Counter()
    for i in range(len(dataset)):
        labels = dataset[i]["labels"].tolist()
        for lbl in labels:
            if lbl != -100:
                counts[lbl] += 1

    num_classes = len(LABEL2ID)
    total_tokens = sum(counts.values())
    weights = torch.ones(num_classes, dtype=torch.float32)

    for lbl_id in range(num_classes):
        c = max(counts.get(lbl_id, 1), 1)
        weights[lbl_id] = (total_tokens / c) ** power

    # Нормализуем так, чтобы вес 'O' был равен 0.35, а токсичные классы масштабировались пропорционально
    weights = weights / weights[LABEL2ID["O"]] * 0.35
    weights = torch.clamp(weights, min=0.25, max=6.0)

    print("\n[Weights] Автоматически рассчитанные веса классов для Loss:")
    for lbl_id, w in enumerate(weights.tolist()):
        print(f"  {ID2LABEL[lbl_id]:22s}: weight={w:.3f} (tokens={counts.get(lbl_id, 0)})")

    return weights.to(device)


# =====================================================================
# 4. АРХИТЕКТУРА МОДЕЛИ
# =====================================================================

class RuBertToxicDetector(nn.Module):
    def __init__(
        self,
        model_name: str = "DeepPavlov/rubert-base-cased",
        num_labels: int = len(LABEL2ID),
        num_layers_pool: int = 4,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.rubert = AutoModel.from_pretrained(model_name, config=self.config)

        self.num_layers_pool = num_layers_pool
        self.layer_weights = nn.Parameter(torch.ones(num_layers_pool))

        # Multi-Sample Dropout (5 параллельных веток)
        self.dropouts = nn.ModuleList([nn.Dropout(0.2) for _ in range(5)])
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None, class_weights=None):
        outputs = self.rubert(input_ids=input_ids, attention_mask=attention_mask)
        top_hidden_states = outputs.hidden_states[-self.num_layers_pool:]

        # Взвешенная агрегация последних 4 слоев
        weights = torch.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
        stacked = torch.stack(top_hidden_states, dim=0)
        pooled_output = torch.sum(weights * stacked, dim=0)

        # Усреднение логитов после Multi-Sample Dropout
        logits = torch.mean(
            torch.stack([self.classifier(drop(pooled_output)) for drop in self.dropouts], dim=0),
            dim=0,
        )

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
            loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))

        return {"loss": loss, "logits": logits}


# =====================================================================
# 5. МЕТРИКИ (TOKEN-LEVEL F1) И ВАЛИДАЦИЯ
# =====================================================================

def evaluate_model(model, val_loader, class_weights, device):
    model.eval()
    total_loss = 0.0
    flat_preds = []
    flat_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    class_weights=class_weights,
                )
                loss = outputs["loss"]
                logits = outputs["logits"]

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            gold = labels.cpu().numpy()

            for p_seq, l_seq in zip(preds, gold):
                for p_id, l_id in zip(p_seq, l_seq):
                    if l_id != -100:
                        # Схлопываем B-cat и I-cat в cat для расчета Token-level F1 по категориям
                        p_cat = ID2LABEL[int(p_id)].split("-", 1)[-1]
                        l_cat = ID2LABEL[int(l_id)].split("-", 1)[-1]
                        flat_preds.append(p_cat)
                        flat_labels.append(l_cat)

    avg_loss = total_loss / max(len(val_loader), 1)

    macro_f1 = f1_score(
        flat_labels, flat_preds, labels=TARGET_CATEGORIES, average="macro", zero_division=0
    )
    micro_f1 = f1_score(
        flat_labels, flat_preds, labels=TARGET_CATEGORIES, average="micro", zero_division=0
    )
    per_class_f1_arr = f1_score(
        flat_labels, flat_preds, labels=TARGET_CATEGORIES, average=None, zero_division=0
    )
    per_class_f1 = {
        cat: float(score) for cat, score in zip(TARGET_CATEGORIES, per_class_f1_arr)
    }

    return {
        "val_loss": avg_loss,
        "token_macro_f1": float(macro_f1),
        "token_micro_f1": float(micro_f1),
        "per_class_f1": per_class_f1,
    }


# =====================================================================
# 6. ОСНОВНОЙ ЦИКЛ ОБУЧЕНИЯ
# =====================================================================

def build_optimizer(model, lr_encoder: float = 2.5e-5, lr_head: float = 2e-4, weight_decay: float = 0.01):
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]

    encoder_params_decay = [
        p for n, p in model.rubert.named_parameters() if not any(nd in n for nd in no_decay)
    ]
    encoder_params_no_decay = [
        p for n, p in model.rubert.named_parameters() if any(nd in n for nd in no_decay)
    ]
    head_params = list(model.classifier.parameters()) + [model.layer_weights]

    optimizer_grouped_parameters = [
        {"params": encoder_params_decay, "lr": lr_encoder, "weight_decay": weight_decay},
        {"params": encoder_params_no_decay, "lr": lr_encoder, "weight_decay": 0.0},
        {"params": head_params, "lr": lr_head, "weight_decay": weight_decay},
    ]
    return torch.optim.AdamW(optimizer_grouped_parameters)


def main():
    parser = argparse.ArgumentParser(description="Обучение RuBERT для детекции токсичных спанов")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/ru_paradetox_toxic_spans.jsonl",
        help="Путь к исходному JSONL датасету",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/rubert_toxic_detector",
        help="Директория для сохранения лучшей модели и токенизатора",
    )
    parser.add_argument("--model_name", type=str, default="DeepPavlov/rubert-base-cased")
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr_encoder", type=float, default=2.5e-5)
    parser.add_argument("--lr_head", type=float, default=2e-4)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Используется устройство: {device}")
    if device.type == "cuda":
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")

    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Загрузка, фильтрация harassment_doxxing и склейка дубликатов
    samples = load_and_preprocess_dataset(data_path)

    # Сохраняем очищенный датасет для прозрачности и воспроизводимости
    cleaned_data_path = output_dir / "cleaned_deduplicated_dataset.jsonl"
    with open(cleaned_data_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[Data] Очищенный датасет сохранен в: {cleaned_data_path}")

    # 2. Стратифицированное разбиение на train / val
    train_samples, val_samples = stratified_split(samples, val_size=args.val_size, seed=args.seed)
    print(f"[Split] Train: {len(train_samples)} семплов | Val: {len(val_samples)} семплов")

    # 3. Токенизатор и датасеты
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    train_dataset = ToxicSpanDataset(train_samples, tokenizer, max_len=args.max_len)
    val_dataset = ToxicSpanDataset(val_samples, tokenizer, max_len=args.max_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # 4. Расчет весов классов по тренировочным токенам
    class_weights = compute_damped_class_weights(train_dataset, power=0.35, device=device)

    # 5. Инициализация модели, оптимизатора и расписания LR
    model = RuBertToxicDetector(model_name=args.model_name, num_labels=len(LABEL2ID)).to(device)
    optimizer = build_optimizer(model, lr_encoder=args.lr_encoder, lr_head=args.lr_head)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # 6. Цикл обучения
    best_macro_f1 = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]", leave=False)
        for batch in progress_bar:
            optimizer.zero_grad(set_to_none=True)

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    class_weights=class_weights,
                )
                loss = outputs["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Фиксируем масштаб до шага
            scale_before = scaler.get_scale()

            scaler.step(optimizer)
            scaler.update()

            # Проверяем, состоялся ли реальный шаг оптимизатора
            # Если масштаб не уменьшился (scale_after >= scale_before), значит инфов не было и шаг выполнен
            if scaler.get_scale() >= scale_before:
                scheduler.step()

            train_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / max(len(train_loader), 1)

        # Валидация
        val_metrics = evaluate_model(model, val_loader, class_weights, device)

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            **val_metrics,
        }
        history.append(epoch_record)

        print(
            f"\n--- Эпоха {epoch}/{args.epochs} ---\n"
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_metrics['val_loss']:.4f}\n"
            f"Token Macro F1: {val_metrics['token_macro_f1']:.4f} | "
            f"Token Micro F1: {val_metrics['token_micro_f1']:.4f}"
        )
        print("Per-class Token F1:")
        for cat, score in val_metrics["per_class_f1"].items():
            print(f"  - {cat:20s}: {score:.4f}")

        # Сохраняем лучшую модель по Token Macro F1 (справедливая метрика при редких классах)
        if val_metrics["token_macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["token_macro_f1"]
            best_ckpt_path = output_dir / "best_model.pt"

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "model_name": args.model_name,
                    "max_len": args.max_len,
                    "label2id": LABEL2ID,
                    "id2label": ID2LABEL,
                    "target_categories": TARGET_CATEGORIES,
                    "val_metrics": val_metrics,
                },
                best_ckpt_path,
            )
            tokenizer.save_pretrained(output_dir / "tokenizer")
            print(f"[Checkpoint] Новая лучшая модель сохранена в {best_ckpt_path} (Macro F1: {best_macro_f1:.4f})")

    # Сохраняем историю обучения
    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"\nОбучение завершено! Лучший Token-level Macro F1 на валидации: {best_macro_f1:.4f}")


if __name__ == "__main__":
    main()