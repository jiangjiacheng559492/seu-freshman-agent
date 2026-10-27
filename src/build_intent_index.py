"""把意图目录中的示例问题转换为本地向量索引。"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]

CATALOG_PATH = ROOT / "data" / "intents" / "intent_catalog.jsonl"
DOCUMENT_INDEX_PATH = (
    ROOT / "data" / "index" / "freshman_guide_index.json"
)
OUTPUT_PATH = ROOT / "data" / "index" / "intent_index.json"

BATCH_SIZE = 10


def load_catalog() -> list[dict]:
    """读取并检查意图目录。"""
    if not CATALOG_PATH.exists():
        raise RuntimeError(f"找不到意图目录：{CATALOG_PATH}")

    intents = []
    seen_ids = set()

    with CATALOG_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                intent = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"第 {line_number} 行不是合法 JSON：{exc}"
                ) from exc

            required_fields = {
                "intent_id",
                "name",
                "examples",
                "keywords",
                "expected_sections",
            }

            missing_fields = required_fields - intent.keys()

            if missing_fields:
                raise RuntimeError(
                    f"第 {line_number} 行缺少字段："
                    f"{sorted(missing_fields)}"
                )

            intent_id = intent["intent_id"]

            if intent_id in seen_ids:
                raise RuntimeError(
                    f"发现重复的 intent_id：{intent_id}"
                )

            if not intent["examples"]:
                raise RuntimeError(
                    f"意图 {intent_id} 没有示例问题。"
                )

            seen_ids.add(intent_id)
            intents.append(intent)

    return intents


def load_embedding_settings() -> tuple[str, int]:
    """沿用指南知识库的向量模型和维度。"""
    if not DOCUMENT_INDEX_PATH.exists():
        raise RuntimeError(
            "找不到指南向量索引，请先运行 "
            "src/build_local_index.py。"
        )

    index_data = json.loads(
        DOCUMENT_INDEX_PATH.read_text(encoding="utf-8")
    )

    metadata = index_data["metadata"]

    return (
        metadata["embedding_model"],
        metadata["dimensions"],
    )


def build_entries(intents: list[dict]) -> list[dict]:
    """将每个示例问题展开为一条独立记录。"""
    entries = []

    for intent in intents:
        for example_number, example in enumerate(
            intent["examples"],
            start=1,
        ):
            entries.append(
                {
                    "example_id": (
                        f"{intent['intent_id']}-"
                        f"{example_number:02d}"
                    ),
                    "intent_id": intent["intent_id"],
                    "name": intent["name"],
                    "example": example,
                    "keywords": intent["keywords"],
                    "expected_sections": (
                        intent["expected_sections"]
                    ),
                }
            )

    return entries


def embed_entries(
    client: OpenAI,
    model: str,
    dimensions: int,
    entries: list[dict],
) -> None:
    """分批生成示例问题向量。"""
    total = len(entries)

    for start in range(0, total, BATCH_SIZE):
        batch = entries[start : start + BATCH_SIZE]
        inputs = [entry["example"] for entry in batch]

        response = client.embeddings.create(
            model=model,
            input=inputs,
            dimensions=dimensions,
            encoding_format="float",
        )

        embeddings = sorted(
            response.data,
            key=lambda item: item.index,
        )

        for entry, item in zip(batch, embeddings):
            entry["embedding"] = item.embedding

        completed = min(start + len(batch), total)
        print(f"向量化进度：{completed}/{total}")


def main() -> int:
    load_dotenv(ROOT / ".env")

    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    base_url = os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ).strip()

    if not api_key:
        print(
            "配置错误：.env 中没有 DASHSCOPE_API_KEY。",
            file=sys.stderr,
        )
        return 2

    try:
        intents = load_catalog()
        embedding_model, dimensions = (
            load_embedding_settings()
        )
    except RuntimeError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2

    entries = build_entries(intents)

    print(f"意图数量：{len(intents)}")
    print(f"示例问题数量：{len(entries)}")
    print(f"向量模型：{embedding_model}")
    print(f"向量维度：{dimensions}")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    try:
        embed_entries(
            client=client,
            model=embedding_model,
            dimensions=dimensions,
            entries=entries,
        )
    except Exception as exc:
        print(
            f"建立意图索引失败：{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    source_text = CATALOG_PATH.read_text(encoding="utf-8")

    output_data = {
        "metadata": {
            "source": str(CATALOG_PATH),
            "source_sha256": hashlib.sha256(
                source_text.encode("utf-8")
            ).hexdigest(),
            "embedding_model": embedding_model,
            "dimensions": dimensions,
            "intent_count": len(intents),
            "example_count": len(entries),
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
        },
        "entries": entries,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(output_data, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"意图向量索引已保存：{OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())