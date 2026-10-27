"""把新生指南切片、向量化，并保存为本地 JSON 索引。"""

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
SOURCE_PATH = ROOT / "data" / "knowledge" / "freshman_guide.md"
INDEX_PATH = ROOT / "data" / "index" / "freshman_guide_index.json"

MAX_CHARS = 700
OVERLAP_CHARS = 100
BATCH_SIZE = 10
DIMENSIONS = 1024


def parse_paragraphs(text: str) -> list[tuple[str, str]]:
    """读取 Markdown，并记录每段文字所在的章节。"""
    paragraphs: list[tuple[str, str]] = []
    buffer: list[str] = []

    document_title = "新生指南"
    level_two = ""
    level_three = ""

    def current_section() -> str:
        parts = [document_title, level_two, level_three]
        return " / ".join(part for part in parts if part)

    def flush() -> None:
        if buffer:
            paragraph = "\n".join(buffer).strip()
            if paragraph:
                paragraphs.append((current_section(), paragraph))
            buffer.clear()

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if line.startswith("# "):
            flush()
            document_title = line[2:].strip()
            level_two = ""
            level_three = ""
        elif line.startswith("## "):
            flush()
            level_two = line[3:].strip()
            level_three = ""
        elif line.startswith("### "):
            flush()
            level_three = line[4:].strip()
        elif not line:
            flush()
        else:
            buffer.append(line)

    flush()
    return paragraphs


def split_long_text(text: str) -> list[str]:
    """将过长段落按字符长度切分，并保留少量重叠。"""
    if len(text) <= MAX_CHARS:
        return [text]

    pieces: list[str] = []
    start = 0
    step = MAX_CHARS - OVERLAP_CHARS

    while start < len(text):
        piece = text[start : start + MAX_CHARS].strip()
        if piece:
            pieces.append(piece)
        start += step

    return pieces


def build_chunks(text: str) -> list[dict]:
    """把相邻的小段合并为适合检索的文本块。"""
    paragraphs = parse_paragraphs(text)
    chunks: list[dict] = []

    current_section = ""
    current_parts: list[str] = []
    current_length = 0

    def flush() -> None:
        nonlocal current_parts, current_length

        if not current_parts:
            return

        chunks.append(
            {
                "id": f"chunk-{len(chunks) + 1:03d}",
                "section": current_section,
                "text": "\n".join(current_parts),
            }
        )
        current_parts = []
        current_length = 0

    for section, paragraph in paragraphs:
        for piece in split_long_text(paragraph):
            added_length = len(piece) + 1

            if (
                current_parts
                and (
                    section != current_section
                    or current_length + added_length > MAX_CHARS
                )
            ):
                flush()

            if not current_parts:
                current_section = section

            current_parts.append(piece)
            current_length += added_length

    flush()
    return chunks


def embed_chunks(client: OpenAI, model: str, chunks: list[dict]) -> None:
    """分批请求百炼 Embedding 接口。"""
    total = len(chunks)

    for start in range(0, total, BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]

        inputs = [
            f"章节：{chunk['section']}\n内容：{chunk['text']}"
            for chunk in batch
        ]

        response = client.embeddings.create(
            model=model,
            input=inputs,
            dimensions=DIMENSIONS,
            encoding_format="float",
        )

        embeddings = sorted(response.data, key=lambda item: item.index)

        for chunk, item in zip(batch, embeddings):
            chunk["embedding"] = item.embedding

        completed = min(start + len(batch), total)
        print(f"向量化进度：{completed}/{total}")


def main() -> int:
    load_dotenv(ROOT / ".env")

    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    base_url = os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ).strip()
    embedding_model = os.getenv(
        "DASHSCOPE_EMBEDDING_MODEL",
        "text-embedding-v4",
    ).strip()

    if not api_key:
        print(
            "配置错误：.env 中没有 DASHSCOPE_API_KEY。",
            file=sys.stderr,
        )
        return 2

    if not SOURCE_PATH.exists():
        print(f"找不到指南文件：{SOURCE_PATH}", file=sys.stderr)
        return 2

    source_text = SOURCE_PATH.read_text(encoding="utf-8")
    chunks = build_chunks(source_text)

    print(f"指南文件：{SOURCE_PATH}")
    print(f"切片数量：{len(chunks)}")
    print(f"向量模型：{embedding_model}")

    client = OpenAI(api_key=api_key, base_url=base_url)

    try:
        embed_chunks(client, embedding_model, chunks)
    except Exception as exc:
        print(
            f"建立索引失败：{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    index_data = {
        "metadata": {
            "source": str(SOURCE_PATH),
            "source_sha256": hashlib.sha256(
                source_text.encode("utf-8")
            ).hexdigest(),
            "embedding_model": embedding_model,
            "dimensions": DIMENSIONS,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "chunk_count": len(chunks),
        },
        "chunks": chunks,
    }

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(
        json.dumps(index_data, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"本地向量索引已保存：{INDEX_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())