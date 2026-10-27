"""批量评测带意图路由的混合 RAG。"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from hybrid_rag_agent import hybrid_retrieve
from intent_router import (
    determine_confidence,
    load_intent_index,
    rank_intents,
    select_intents,
)
from local_rag_agent import (
    embed_question,
    generate_answer,
    load_index,
    load_settings,
)


ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = ROOT / "evals" / "questions.jsonl"
RESULTS_DIR = ROOT / "results"


def load_questions() -> list[dict]:
    with EVAL_PATH.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def main() -> int:
    api_key, model, base_url = load_settings()
    document_index = load_index()
    intent_index = load_intent_index()
    questions = load_questions()

    document_metadata = document_index["metadata"]
    intent_metadata = intent_index["metadata"]

    if (
        document_metadata["embedding_model"]
        != intent_metadata["embedding_model"]
        or document_metadata["dimensions"]
        != intent_metadata["dimensions"]
    ):
        raise RuntimeError(
            "指南索引和意图索引的模型或维度不一致。"
        )

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = (
        RESULTS_DIR
        / f"hybrid-rag-{timestamp}.jsonl"
    )

    print(f"模型：{model}")
    print(f"题目数量：{len(questions)}")
    print(f"结果文件：{output_path}")

    failed_count = 0

    with output_path.open("w", encoding="utf-8") as output:
        for position, case in enumerate(
            questions,
            start=1,
        ):
            question = case["question"]

            selected_intents = []
            retrieved_chunks = []
            answer = ""
            error = None
            confidence = "低"

            print("\n" + "=" * 70)
            print(
                f"[{position:02d}/{len(questions)}] "
                f"{case['id']}"
            )
            print(f"问题：{question}")

            try:
                query_embedding = embed_question(
                    client=client,
                    embedding_model=(
                        document_metadata[
                            "embedding_model"
                        ]
                    ),
                    dimensions=(
                        document_metadata["dimensions"]
                    ),
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

                confidence = determine_confidence(
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

                print(
                    f"路由置信度：{confidence}"
                )

                if selected_intents:
                    print("识别意图：")

                    for intent in selected_intents:
                        print(
                            f"- {intent['name']} "
                            f"({intent['combined_score']:.4f})"
                        )
                else:
                    print("识别意图：未达到路由阈值")

                print("检索章节：")

                for chunk in retrieved_chunks:
                    print(
                        f"- {chunk['section']} "
                        f"(最终={chunk['final_score']:.4f}, "
                        f"语义={chunk['semantic_score']:.4f})"
                    )

                print(f"\n回答：\n{answer}")

            except Exception as exc:
                failed_count += 1
                error = f"{type(exc).__name__}: {exc}"
                print(f"错误：{error}")

            result = {
                "id": case["id"],
                "category": case["category"],
                "question": question,
                "answer": answer,
                "error": error,
                "model": model,
                "embedding_model": (
                    document_metadata["embedding_model"]
                ),
                "variant": "hybrid_rag_with_intent_router",
                "route_confidence": confidence,
                "selected_intents": [
                    {
                        "intent_id": intent["intent_id"],
                        "name": intent["name"],
                        "combined_score": round(
                            intent["combined_score"],
                            6,
                        ),
                        "matched_keywords": (
                            intent["matched_keywords"]
                        ),
                    }
                    for intent in selected_intents
                ],
                "retrieved_chunks": [
                    {
                        "id": chunk["id"],
                        "section": chunk["section"],
                        "semantic_score": round(
                            chunk["semantic_score"],
                            6,
                        ),
                        "section_boost": round(
                            chunk["section_boost"],
                            6,
                        ),
                        "keyword_boost": round(
                            chunk["keyword_boost"],
                            6,
                        ),
                        "final_score": round(
                            chunk["final_score"],
                            6,
                        ),
                        "text": chunk["text"],
                    }
                    for chunk in retrieved_chunks
                ],
            }

            output.write(
                json.dumps(result, ensure_ascii=False)
                + "\n"
            )
            output.flush()

            time.sleep(0.5)

    print("\n" + "=" * 70)
    print(f"评测结果已保存：{output_path}")
    print(f"成功：{len(questions) - failed_count}")
    print(f"失败：{failed_count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())