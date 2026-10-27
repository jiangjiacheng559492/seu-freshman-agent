"""使用意图向量索引识别用户问题的可能意图。"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from local_rag_agent import cosine_similarity


ROOT = Path(__file__).resolve().parents[1]
INTENT_INDEX_PATH = (
    ROOT / "data" / "index" / "intent_index.json"
)

ROUTE_MIN_SCORE = 0.62
MULTI_INTENT_MARGIN = 0.05
MAX_INTENTS = 3


def load_intent_index() -> dict:
    if not INTENT_INDEX_PATH.exists():
        raise RuntimeError(
            "找不到意图向量索引，请先运行 "
            "src/build_intent_index.py。"
        )

    return json.loads(
        INTENT_INDEX_PATH.read_text(encoding="utf-8")
    )


def calculate_keyword_score(
    question: str,
    keywords: list[str],
) -> tuple[float, list[str]]:
    """计算问题中出现了多少个领域关键词。"""
    normalized_question = question.lower().replace(" ", "")

    matched_keywords = [
        keyword
        for keyword in keywords
        if keyword.lower().replace(" ", "")
        in normalized_question
    ]

    # 命中两个及以上关键词时，关键词得分达到1。
    keyword_score = min(
        len(matched_keywords) / 2,
        1.0,
    )

    return keyword_score, matched_keywords


def rank_intents(
    question: str,
    query_embedding: list[float],
    entries: list[dict],
) -> list[dict]:
    """将示例问题得分汇总到意图级别。"""
    grouped: dict[str, dict] = {}

    for entry in entries:
        intent_id = entry["intent_id"]

        if intent_id not in grouped:
            grouped[intent_id] = {
                "intent_id": intent_id,
                "name": entry["name"],
                "keywords": entry["keywords"],
                "expected_sections": (
                    entry["expected_sections"]
                ),
                "example_scores": [],
            }

        semantic_score = cosine_similarity(
            query_embedding,
            entry["embedding"],
        )

        grouped[intent_id]["example_scores"].append(
            {
                "example": entry["example"],
                "score": semantic_score,
            }
        )

    ranked_intents = []

    for intent in grouped.values():
        example_scores = sorted(
            intent["example_scores"],
            key=lambda item: item["score"],
            reverse=True,
        )

        best_score = example_scores[0]["score"]

        second_score = (
            example_scores[1]["score"]
            if len(example_scores) > 1
            else best_score
        )

        # 不只依赖单个示例，综合最相似的两个示例。
        semantic_score = (
            best_score * 0.7
            + second_score * 0.3
        )

        keyword_score, matched_keywords = (
            calculate_keyword_score(
                question,
                intent["keywords"],
            )
        )

        # 向量语义占90%，关键词只作为轻量辅助。
        combined_score = (
            semantic_score * 0.9
            + keyword_score * 0.1
        )

        ranked_intents.append(
            {
                "intent_id": intent["intent_id"],
                "name": intent["name"],
                "semantic_score": semantic_score,
                "keyword_score": keyword_score,
                "combined_score": combined_score,
                "matched_keywords": matched_keywords,
                "best_example": example_scores[0][
                    "example"
                ],
                "expected_sections": (
                    intent["expected_sections"]
                ),
            }
        )

    ranked_intents.sort(
        key=lambda item: item["combined_score"],
        reverse=True,
    )

    return ranked_intents


def select_intents(
    ranked_intents: list[dict],
) -> list[dict]:
    """根据绝对阈值和相对差距选择可能意图。"""
    if not ranked_intents:
        return []

    top_score = ranked_intents[0]["combined_score"]

    if top_score < ROUTE_MIN_SCORE:
        return []

    selected = []

    for intent in ranked_intents:
        if len(selected) >= MAX_INTENTS:
            break

        score = intent["combined_score"]

        if score < ROUTE_MIN_SCORE:
            continue

        if top_score - score > MULTI_INTENT_MARGIN:
            continue

        selected.append(intent)

    return selected


def determine_confidence(
    ranked_intents: list[dict],
) -> str:
    """生成便于观察的置信度标签。"""
    if not ranked_intents:
        return "低"

    top_score = ranked_intents[0]["combined_score"]
    second_score = (
        ranked_intents[1]["combined_score"]
        if len(ranked_intents) > 1
        else 0.0
    )

    margin = top_score - second_score

    if top_score >= 0.72 and margin >= 0.04:
        return "高"

    if top_score >= ROUTE_MIN_SCORE:
        return "中"

    return "低"


def get_recommended_sections(
    selected_intents: list[dict],
) -> list[str]:
    """合并多个意图推荐的章节并去重。"""
    sections = []

    for intent in selected_intents:
        for section in intent["expected_sections"]:
            if section not in sections:
                sections.append(section)

    return sections


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
        intent_index = load_intent_index()
    except RuntimeError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2

    question = input("请输入问题：").strip()

    if not question:
        print("问题不能为空。", file=sys.stderr)
        return 2

    metadata = intent_index["metadata"]

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    try:
        response = client.embeddings.create(
            model=metadata["embedding_model"],
            input=question,
            dimensions=metadata["dimensions"],
            encoding_format="float",
        )
    except Exception as exc:
        print(
            f"意图识别失败：{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    query_embedding = response.data[0].embedding

    ranked_intents = rank_intents(
        question=question,
        query_embedding=query_embedding,
        entries=intent_index["entries"],
    )

    selected_intents = select_intents(ranked_intents)
    confidence = determine_confidence(ranked_intents)

    print("\n得分最高的5个意图：")

    for position, intent in enumerate(
        ranked_intents[:5],
        start=1,
    ):
        print(
            f"{position}. {intent['name']} "
            f"(综合={intent['combined_score']:.4f}, "
            f"语义={intent['semantic_score']:.4f}, "
            f"关键词={intent['keyword_score']:.4f})"
        )
        print(
            f"   最相似示例：{intent['best_example']}"
        )
        print(
            f"   命中关键词："
            f"{intent['matched_keywords'] or '无'}"
        )

    print(f"\n路由置信度：{confidence}")

    if selected_intents:
        print("选中的可能意图：")

        for intent in selected_intents:
            print(
                f"- {intent['name']} "
                f"({intent['combined_score']:.4f})"
            )
    else:
        print("没有达到阈值的意图，应该使用普通RAG检索。")

    sections = get_recommended_sections(
        selected_intents
    )

    if sections:
        print("\n建议优先检索的章节：")

        for section in sections:
            print(f"- {section}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())