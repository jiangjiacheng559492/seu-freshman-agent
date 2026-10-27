"""结合意图路由、关键词和文档向量的混合 RAG。"""

from __future__ import annotations

import sys

from openai import OpenAI

from intent_router import (
    determine_confidence,
    get_recommended_sections,
    load_intent_index,
    rank_intents,
    select_intents,
)
from local_rag_agent import (
    cosine_similarity,
    embed_question,
    generate_answer,
    load_index,
    load_settings,
)


TOP_K = 5

# 这两个参数是初始实验值，后续通过评测调整。
SECTION_BOOST = 0.07
MAX_KEYWORD_BOOST = 0.04


def collect_active_keywords(
    selected_intents: list[dict],
) -> list[str]:
    """收集用户问题实际命中的意图关键词。"""
    keywords = []

    for intent in selected_intents:
        for keyword in intent["matched_keywords"]:
            if keyword not in keywords:
                keywords.append(keyword)

    return keywords


def calculate_section_boost(
    section: str,
    recommended_sections: list[str],
) -> float:
    """推荐章节命中时增加少量分数。"""
    for expected_section in recommended_sections:
        if expected_section in section:
            return SECTION_BOOST

    return 0.0


def calculate_document_keyword_boost(
    chunk: dict,
    active_keywords: list[str],
) -> tuple[float, list[str]]:
    """计算问题关键词与指南片段的重合情况。"""
    searchable_text = (
        chunk["section"] + "\n" + chunk["text"]
    ).lower().replace(" ", "")

    matched_keywords = [
        keyword
        for keyword in active_keywords
        if keyword.lower().replace(" ", "")
        in searchable_text
    ]

    if not active_keywords:
        return 0.0, []

    keyword_ratio = (
        len(matched_keywords)
        / len(active_keywords)
    )

    keyword_boost = (
        keyword_ratio * MAX_KEYWORD_BOOST
    )

    return keyword_boost, matched_keywords


def hybrid_retrieve(
    query_embedding: list[float],
    chunks: list[dict],
    selected_intents: list[dict],
    top_k: int = TOP_K,
) -> list[dict]:
    """融合向量、意图章节和关键词进行检索。"""
    recommended_sections = get_recommended_sections(
        selected_intents
    )

    active_keywords = collect_active_keywords(
        selected_intents
    )

    results = []

    for chunk in chunks:
        semantic_score = cosine_similarity(
            query_embedding,
            chunk["embedding"],
        )

        section_boost = calculate_section_boost(
            chunk["section"],
            recommended_sections,
        )

        keyword_boost, matched_keywords = (
            calculate_document_keyword_boost(
                chunk,
                active_keywords,
            )
        )

        final_score = (
            semantic_score
            + section_boost
            + keyword_boost
        )

        results.append(
            {
                "id": chunk["id"],
                "section": chunk["section"],
                "text": chunk["text"],
                "semantic_score": semantic_score,
                "section_boost": section_boost,
                "keyword_boost": keyword_boost,
                "final_score": final_score,
                "matched_keywords": matched_keywords,
            }
        )

    results.sort(
        key=lambda item: item["final_score"],
        reverse=True,
    )

    return results[:top_k]


def main() -> int:
    try:
        api_key, model, base_url = load_settings()
        document_index = load_index()
        intent_index = load_intent_index()
    except RuntimeError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2

    document_metadata = document_index["metadata"]
    intent_metadata = intent_index["metadata"]

    if (
        document_metadata["embedding_model"]
        != intent_metadata["embedding_model"]
        or document_metadata["dimensions"]
        != intent_metadata["dimensions"]
    ):
        print(
            "配置错误：指南索引和意图索引使用的"
            "向量模型或维度不一致。",
            file=sys.stderr,
        )
        return 2

    question = input("请输入问题：").strip()

    if not question:
        print("问题不能为空。", file=sys.stderr)
        return 2

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    try:
        # 一个问题只进行一次向量化，同时用于两个索引。
        query_embedding = embed_question(
            client=client,
            embedding_model=(
                document_metadata["embedding_model"]
            ),
            dimensions=document_metadata["dimensions"],
            question=question,
        )

        ranked_intents = rank_intents(
            question=question,
            query_embedding=query_embedding,
            entries=intent_index["entries"],
        )

        selected_intents = select_intents(
            ranked_intents
        )

        retrieved_chunks = hybrid_retrieve(
            query_embedding=query_embedding,
            chunks=document_index["chunks"],
            selected_intents=selected_intents,
        )

        answer = generate_answer(
            client=client,
            model=model,
            question=question,
            retrieved_chunks=retrieved_chunks,
        )

    except Exception as exc:
        print(
            f"问答失败：{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    confidence = determine_confidence(
        ranked_intents
    )

    print(f"\n意图路由置信度：{confidence}")

    if selected_intents:
        print("识别到的可能意图：")

        for intent in selected_intents:
            print(
                f"- {intent['name']} "
                f"({intent['combined_score']:.4f})"
            )
    else:
        print("未找到可靠意图，使用普通向量检索。")

    print("\n混合检索结果：")

    for position, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):
        print(
            f"{position}. {chunk['section']}"
        )
        print(
            f"   最终={chunk['final_score']:.4f}，"
            f"语义={chunk['semantic_score']:.4f}，"
            f"章节加分={chunk['section_boost']:.4f}，"
            f"关键词加分={chunk['keyword_boost']:.4f}"
        )

    print(f"\n模型：{model}")
    print("\n回答：")
    print(answer)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())