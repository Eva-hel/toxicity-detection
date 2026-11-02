"""Pseudo-label Russian toxic spans in s-nlp/ru_paradetox.

Default backend: local Ollama.
Optional backend: OpenRouter.
Output: JSONL, one annotated sample per line.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import httpx
from datasets import load_dataset
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm


# ============================================================
# 1. Настройки
# ============================================================

BACKEND = os.getenv("ANNOTATION_BACKEND", "ollama").strip().lower()

DATASET_NAME = os.getenv("DATASET_NAME", "s-nlp/ru_paradetox")
DATASET_SPLIT = os.getenv("DATASET_SPLIT", "train")

OUTPUT_PATH = Path(
    os.getenv(
        "OUTPUT_PATH",
        "ru_paradetox_toxic_spans.jsonl",
    )
)

DEFAULT_BATCH_SIZE = "1" if BACKEND == "ollama" else "10"
BATCH_SIZE = 5

DEFAULT_RETRIES = "2" if BACKEND == "ollama" else "5"
MAX_RETRIES = max(
    1,
    int(os.getenv("MAX_RETRIES", DEFAULT_RETRIES)),
)

OLLAMA_NUM_CTX = max(
    2048,
    int(os.getenv("OLLAMA_NUM_CTX", "4096")),
)

CATEGORIES = {
    "profanity",
    "insult",
    "derogation_mockery",
    "threat_violence",
    "harassment_doxxing",
    "hate_speech",
}


# ============================================================
# 2. Схемы данных
# ============================================================

class LLMSpan(BaseModel):
    span: str = Field(
        description="Точная подстрока из токсичного комментария"
    )
    category: str


class LLMItem(BaseModel):
    id: int
    spans: list[LLMSpan] = Field(default_factory=list)


class LLMResponse(BaseModel):
    items: list[LLMItem] = Field(default_factory=list)


class Entity(BaseModel):
    span: str
    start: int
    end: int
    category: str


class AnnotatedSample(BaseModel):
    id: int
    text: str
    neutral_reference: str
    entities: list[Entity]


# ============================================================
# 3. Системный промпт
# ============================================================

SYSTEM_PROMPT = """Ты — эксперт по лингвистической разметке
токсичных текстов на русском языке.

Для каждого комментария найди все фрагменты (спаны),
содержащие токсичные языковые паттерны, и отнеси каждый
спан ровно к одной категории.

Нейтральный аналог используй только как дополнительный
контекст. Размечай именно токсичный комментарий.

Категории:

1. profanity — обсценная лексика, мат, междометия и
ругательства общего плана без прямой адресации личности.

2. insult — прямые оскорбления личности, внешности или ума.

3. derogation_mockery — язвительное коверкание слов,
уничижительные ярлыки и насмешки.

4. threat_violence — угрозы расправой или вредом,
пожелания смерти.

5. harassment_doxxing — призывы к деанонимизации, травле
или публикации личных данных.

6. hate_speech — язык вражды, ксенофобия или шовинизм
по расовым, национальным, гендерным и сходным признакам.

Правила:

- Выделяй точную подстроку из токсичного комментария.
- Сохраняй оригинальное написание и регистр.
- Не включай внешнюю пунктуацию, если она не является
  частью самого токсичного выражения.
- Выделяй минимальный достаточный токсичный фрагмент.
- Не придумывай спаны, которых нет в исходном тексте.
- Если токсичных фрагментов нет, верни spans: [].
- Каждый id из входного списка должен появиться в ответе
  ровно один раз.
- Верни только валидный JSON без Markdown и пояснений.

Формат ответа:

{
  "items": [
    {
      "id": 0,
      "spans": [
        {
          "span": "точный фрагмент",
          "category": "profanity"
        }
      ]
    }
  ]
}

Допустимые категории:
profanity, insult, derogation_mockery, threat_violence,
harassment_doxxing, hate_speech.
"""


# ============================================================
# 4. Разбор JSON
# ============================================================

def parse_json_content(raw_content: str) -> dict[str, Any]:
    """Разбирает JSON, даже если модель добавила Markdown."""

    raw_content = (raw_content or "").strip()

    if not raw_content:
        raise ValueError("Модель вернула пустой ответ.")

    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError:
        unfenced = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            raw_content,
            flags=re.IGNORECASE,
        ).strip()

        try:
            parsed = json.loads(unfenced)
        except json.JSONDecodeError:
            start = unfenced.find("{")
            end = unfenced.rfind("}")

            if start < 0 or end <= start:
                raise ValueError(
                    f"В ответе не найден JSON-объект: "
                    f"{raw_content[:300]!r}"
                )

            parsed = json.loads(unfenced[start:end + 1])

    if not isinstance(parsed, dict):
        raise ValueError(
            "Корень JSON-ответа должен быть объектом."
        )

    return parsed


def parse_llm_response(raw_content: str) -> LLMResponse:
    data = parse_json_content(raw_content)
    return LLMResponse.model_validate(data)


# ============================================================
# 5. Определение смещений span
# ============================================================

def align_spans(
    text: str,
    llm_spans: list[LLMSpan],
) -> list[Entity]:
    """Находит точные позиции спанов в исходном тексте."""

    entities: list[Entity] = []
    used_positions: set[tuple[int, int, str]] = set()
    cursor = 0

    for item in llm_spans:
        span = item.span.strip()
        category = item.category.strip()

        if not span or category not in CATEGORIES:
            continue

        chosen = None

        # Сначала ищем с учётом регистра.
        # Если совпадения нет — без учёта регистра.
        for flags in (0, re.IGNORECASE):
            matches = list(
                re.finditer(re.escape(span), text, flags=flags)
            )

            candidates = [
                match
                for match in matches
                if (
                    match.start(),
                    match.end(),
                    category,
                ) not in used_positions
            ]

            if not candidates:
                continue

            after_cursor = [
                match
                for match in candidates
                if match.start() >= cursor
            ]

            chosen = min(
                after_cursor or candidates,
                key=lambda match: match.start(),
            )
            break

        if chosen is None:
            print(
                f"\n[Предупреждение] Спан не найден: {span!r}"
            )
            continue

        start, end = chosen.span()

        used_positions.add((start, end, category))

        entities.append(
            Entity(
                span=text[start:end],
                start=start,
                end=end,
                category=category,
            )
        )

        cursor = max(cursor, end)

    entities.sort(
        key=lambda entity: (
            entity.start,
            entity.end,
            entity.category,
        )
    )

    return entities


# ============================================================
# 6. Настройка подключения к модели
# ============================================================

def create_client() -> tuple[AsyncOpenAI, str]:
    """Создаёт API-клиент для Ollama или OpenRouter."""

    if BACKEND == "ollama":
        base_url = os.getenv(
            "OLLAMA_BASE_URL",
            "http://127.0.0.1:11434/v1",
        )

        model = os.getenv(
            "OLLAMA_MODEL",
            "qwen2.5:14b-instruct",
        )

        # trust_env=False запрещает httpx использовать
        # системные HTTP_PROXY / HTTPS_PROXY.
        #
        # Для localhost это помогает избежать отправки
        # локальных запросов через внешний прокси.
        local_http_client = httpx.AsyncClient(
            trust_env=False,
            timeout=httpx.Timeout(
                180.0,
                connect=10.0,
            ),
        )

        client = AsyncOpenAI(
            base_url=base_url,
            api_key="ollama",
            http_client=local_http_client,
            max_retries=0,
        )

        print(f"Ollama API: {base_url}")
        print("HTTP-прокси: системные настройки игнорируются")
        print(f"Модель: {model}")
        print(f"Размер контекста: {OLLAMA_NUM_CTX}")
        print(f"Размер пачки: {BATCH_SIZE}")

        return client, model

    if BACKEND == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            raise ValueError(
                "Задайте переменную OPENROUTER_API_KEY."
            )

        model = os.getenv(
            "OPENROUTER_MODEL",
            "qwen/qwen-2.5-72b-instruct:free",
        )

        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "Russian Toxic Span Pseudo-labeling",
            },
            timeout=180.0,
            max_retries=0,
        )

        print("Backend: OpenRouter")
        print(f"Модель: {model}")

        return client, model

    raise ValueError(
        f"Неизвестный backend: {BACKEND!r}. "
        "Допустимы ollama и openrouter."
    )


# ============================================================
# 7. Запросы и повторные попытки
# ============================================================


async def completion_with_backoff(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict[str, str]],
) -> Any:
    for attempt in range(MAX_RETRIES):
        try:
            if BACKEND == "openrouter":
                await asyncio.sleep(3.2)

            request_options = {}

            if BACKEND == "ollama":
                request_options["extra_body"] = {
                    "options": {
                        "num_ctx": OLLAMA_NUM_CTX,
                        "repeat_penalty": 1.1,
                        "repeat_last_n": 128,
                        "top_k": 40,
                    }
                }

            return await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                top_p=0.9,
                response_format={"type": "json_object"},
                max_tokens=512,
                **request_options,
            )

        except RateLimitError:
            if attempt == MAX_RETRIES - 1:
                raise

            delay = min(
                60.0,
                5.0 * (2 ** attempt),
            ) + random.uniform(0.2, 1.5)

            print(
                f"\n[Ограничение запросов] "
                f"Попытка {attempt + 1}/{MAX_RETRIES}. "
                f"Пауза {delay:.1f} сек."
            )

            await asyncio.sleep(delay)

        except (
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
        ) as exc:

            status_code = getattr(exc, "status_code", None)

            print(
                f"\n[Сетевая/API-ошибка] "
                f"Тип: {type(exc).__name__}; "
                f"HTTP: {status_code}; "
                f"детали: {exc}"
            )

            if attempt == MAX_RETRIES - 1:
                raise

            delay = min(
                30.0,
                2.0 * (2 ** attempt),
            ) + random.uniform(0.2, 1.0)

            print(
                f"[Повтор] Пауза {delay:.1f} сек."
            )

            await asyncio.sleep(delay)

        except APIError:
            raise

    raise RuntimeError(
        "Не удалось получить ответ после повторных попыток."
    )



# ============================================================
# 8. Сохранение и возобновление
# ============================================================

def load_processed_ids(file_path: Path) -> set[int]:
    """Загружает ID записей, успешно сохранённых ранее."""

    if not file_path.exists():
        return set()

    processed: set[int] = set()

    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
                record_id = record.get("id")

                if isinstance(record_id, int):
                    processed.add(record_id)

            except (json.JSONDecodeError, AttributeError):
                continue

    return processed


def append_jsonl(
    file_path: Path,
    record: dict[str, Any],
) -> None:
    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with file_path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            ) + "\n"
        )
        file.flush()


def save_error(
    sample_id: int,
    error: str,
) -> None:
    error_path = OUTPUT_PATH.with_name(
        f"{OUTPUT_PATH.stem}.errors.jsonl"
    )

    append_jsonl(
        error_path,
        {
            "id": sample_id,
            "error": error,
        },
    )


# ============================================================
# 9. Разметка одной пачки
# ============================================================

async def request_batch(
    client: AsyncOpenAI,
    model_name: str,
    batch: list[dict[str, Any]],
) -> dict[int, list[LLMSpan]]:
    payload = [
        {
            "id": row["id"],
            "toxic_comment": row["text"],
            "neutral_reference": row["neutral_reference"],
        }
        for row in batch
    ]

    user_prompt = (
        "Разметь каждый объект из списка. "
        "Поле toxic_comment — единственный текст, "
        "в котором нужно искать токсичные спаны. "
        "neutral_reference дан только для контекста.\n"
        "Верни JSON с массивом items. Для каждого входного "
        "id верни id и spans.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )

    response = await completion_with_backoff(
        client=client,
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    raw_content = response.choices[0].message.content
    parsed = parse_llm_response(raw_content or "")

    expected_ids = {
        row["id"]
        for row in batch
    }

    by_id: dict[int, list[LLMSpan]] = {}

    for item in parsed.items:
        if item.id in expected_ids:
            by_id[item.id] = item.spans

    return by_id


async def annotate_batch_and_save(
    client: AsyncOpenAI,
    model_name: str,
    batch: list[dict[str, Any]],
) -> int:
    try:
        annotations = await request_batch(
            client,
            model_name,
            batch,
        )

    except InternalServerError as exc:
        status_code = getattr(exc, "status_code", None)

        # Если позже размер пачки будет увеличен, при 503
        # автоматически попробуем разделить пачку на части.
        if (
            BACKEND == "ollama"
            and status_code == 503
            and len(batch) > 1
        ):
            midpoint = len(batch) // 2

            print(
                f"\n[Ollama 503] Разбиваю пачку "
                f"из {len(batch)} записей."
            )

            first = await annotate_batch_and_save(
                client,
                model_name,
                batch[:midpoint],
            )

            second = await annotate_batch_and_save(
                client,
                model_name,
                batch[midpoint:],
            )

            return first + second

        raise

    except (
        ValueError,
        ValidationError,
        json.JSONDecodeError,
    ) as exc:
        if len(batch) > 1:
            print(
                f"\n[Предупреждение] Невалидный ответ: {exc}. "
                "Повторяю по одному комментарию."
            )

            successes = 0

            for row in batch:
                successes += await annotate_batch_and_save(
                    client,
                    model_name,
                    [row],
                )

            return successes

        row = batch[0]

        save_error(
            row["id"],
            f"Некорректный JSON/схема ответа: {exc}",
        )

        print(
            f"\n[Ошибка записи #{row['id']}] {exc}"
        )

        return 0

    successes = 0

    for row in batch:
        sample_id = row["id"]

        if sample_id not in annotations:
            if len(batch) > 1:
                successes += await annotate_batch_and_save(
                    client,
                    model_name,
                    [row],
                )
            else:
                save_error(
                    sample_id,
                    "Модель не вернула объект для этого id.",
                )

                print(
                    f"\n[Ошибка записи #{sample_id}] "
                    "Модель пропустила запись."
                )

            continue

        entities = align_spans(
            row["text"],
            annotations[sample_id],
        )

        record = AnnotatedSample(
            id=sample_id,
            text=row["text"],
            neutral_reference=row["neutral_reference"],
            entities=entities,
        )

        append_jsonl(
            OUTPUT_PATH,
            record.model_dump(),
        )

        successes += 1

    return successes


# ============================================================
# 10. Главная функция
# ============================================================

async def main() -> None:
    client, model_name = create_client()

    print(f"Backend: {BACKEND}; model: {model_name}")
    print(f"Dataset: {DATASET_NAME}, split={DATASET_SPLIT}")
    print(f"Output: {OUTPUT_PATH.resolve()}")

    processed_ids = load_processed_ids(OUTPUT_PATH)

    print(
        f"Ранее успешно размечено: "
        f"{len(processed_ids)} записей"
    )

    print("Загрузка датасета из Hugging Face...")

    dataset = load_dataset(
        DATASET_NAME,
        split=DATASET_SPLIT,
    )

    rows: list[dict[str, Any]] = []

    for idx, row in enumerate(dataset):
        if idx in processed_ids:
            continue

        toxic = row.get("ru_toxic_comment")
        neutral = row.get("ru_neutral_comment")

        toxic_text = (
            toxic if isinstance(toxic, str) else ""
        )

        neutral_text = (
            neutral if isinstance(neutral, str) else ""
        )

        if not toxic_text.strip():
            record = AnnotatedSample(
                id=idx,
                text=toxic_text,
                neutral_reference=neutral_text,
                entities=[],
            )

            append_jsonl(
                OUTPUT_PATH,
                record.model_dump(),
            )

            continue

        rows.append(
            {
                "id": idx,
                "text": toxic_text,
                "neutral_reference": neutral_text,
            }
        )

    # for debugging
    # rows = rows[:1]

    total_pending = len(rows)

    print(
        f"Осталось разметить: {total_pending} "
        f"из {len(dataset)}"
    )

    if total_pending == 0:
        print("Датасет уже полностью размечен.")
        await client.close()
        return

    successful = 0

    try:
        for start in tqdm(
            range(0, len(rows), BATCH_SIZE),
            desc="Разметка",
            unit="batch",
        ):
            batch = rows[start:start + BATCH_SIZE]

            successful += await annotate_batch_and_save(
                client,
                model_name,
                batch,
            )

    except RateLimitError as exc:
        print(
            "\nДостигнут лимит запросов API. "
            "Успешно размеченные записи сохранены."
        )
        print(f"Подробности: {exc}")

    except (
        APIConnectionError,
        APITimeoutError,
    ) as exc:
        print(
            "\nНе удалось подключиться к модели. "
            "Проверьте, что Ollama запущена и доступна."
        )
        print(f"Подробности: {exc}")

    except APIError as exc:
        print(
            "\nAPI вернул ошибку. "
            "Прогресс успешно размеченных записей сохранён."
        )

        print(
            f"HTTP-код: "
            f"{getattr(exc, 'status_code', 'неизвестен')}"
        )

        print(f"Подробности: {exc}")

        body = getattr(exc, "body", None)

        if body:
            print(f"Ответ сервера: {body}")

        response = getattr(exc, "response", None)

        if response is not None:
            try:
                print(
                    f"URL запроса: "
                    f"{response.request.url}"
                )

                response_text = response.text

                if response_text:
                    print(
                        "Тело HTTP-ответа: "
                        f"{response_text[:2000]}"
                    )

            except Exception as diagnostic_exc:
                print(
                    "Не удалось прочитать детали HTTP-ответа: "
                    f"{diagnostic_exc}"
                )

    finally:
        await client.close()

    error_path = OUTPUT_PATH.with_name(
        f"{OUTPUT_PATH.stem}.errors.jsonl"
    )

    print(
        f"\nУспешно записано за этот запуск: {successful}"
    )
    print(f"Результаты: {OUTPUT_PATH.resolve()}")
    print(f"Файл ошибок: {error_path.resolve()}")
    print(
        "Для продолжения запустите скрипт снова. "
        "Уже сохранённые ID будут пропущены."
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())

    except KeyboardInterrupt:
        print(
            "\nОстановлено пользователем. "
            "Успешно сохранённые записи останутся в JSONL."
        )