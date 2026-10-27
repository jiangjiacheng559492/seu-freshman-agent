"""运行任意 JSONL 评测集，并保存混合 RAG 的完整结果。"""

from __future__ import annotations

import argparse
import json
import sys
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
EVALS_DIR = ROOT / "evals"
RESULTS_DIR = ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行混合 RAG 评测集"
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help=(
            "评测集名称或路径，例如 development、"
            "smoke 或 evals/development.jsonl"
        ),
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="每道题之间等待的秒数，默认0.5",
    )

    return parser.parse_args()


def resolve_dataset_path(value: str) -> Path:
    """将数据集名称转换成实际文件路径。"""
    supplied_path = Path(value)

    if supplied_path.exists():
        return supplied_path.resolve()

    if not value.endswith(".jsonl"):
        value = f"{value}.jsonl"

    candidate = EVALS_DIR / value

    if not candidate.exists():
        raise RuntimeError(
            f"找不到评测集：{candidate}"
        )

    return candidate


def load_cases(path: Path) -> list[dict]:
    cases = []
    seen_ids = set()

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"第{line_number}行不是合法JSON：{exc}"
                ) from exc

            for field in ("id", "question"):
                if field not in case:
                    raise RuntimeError(
                        f"第{line_number}行缺少字段："
                        f"{field}"
                    )

            if case["id"] in seen_ids:
                raise RuntimeError(
                    f"发现重复题目ID：{case['id']}"
                )

            seen_ids.add(case["id"])
            cases.append(case)

    if not cases:
        raise RuntimeError("评测集为空。")

    return cases


def main() -> int:
    args = parse_args()

    try:
        dataset_path = resolve_dataset_path(
            args.dataset
        )
        cases = load_cases(dataset_path)

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
            "配置错误：指南索引与意图索引的"
            "模型或维度不一致。",
            file=sys.stderr,
        )
        return 2

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    RESULTS_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )

    output_path = (
        RESULTS_DIR
        / (
            f"{dataset_path.stem}-hybrid-"
            f"{timestamp}.jsonl"
        )
    )

    print(f"评测集：{dataset_path}")
    print(f"题目数量：{len(cases)}")
    print(f"生成模型：{model}")
    print(f"结果文件：{output_path}")

    succeeded = 0
    failed = 0

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output:
        for position, case in enumerate(
            cases,
            start=1,
        ):
            question = case["question"]

            selected_intents = []
            retrieved_chunks = []
            route_confidence = "低"
            answer = ""
            error = None

            print("\n" + "=" * 70)
            print(
                f"[{position:02d}/{len(cases)}] "
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
                        document_metadata[
                            "dimensions"
                        ]
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

                route_confidence = (
                    determine_confidence(
                        ranked_intents
                    )
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

                succeeded += 1

                intent_names = [
                    intent["name"]
                    for intent in selected_intents
                ]

                print(
                    f"意图："
                    f"{intent_names or ['未达到阈值']}"
                )
                print(
                    f"路由置信度："
                    f"{route_confidence}"
                )
                print(f"回答：\n{answer}")

            except Exception as exc:
                failed += 1
                error = (
                    f"{type(exc).__name__}: {exc}"
                )
                print(f"错误：{error}")

            result = {
                "id": case["id"],
                "category": case.get("category"),
                "question": question,

                # 原始评测标准
                "expected_intent_ids": case.get(
                    "intent_ids",
                    [],
                ),
                "expected_sections": case.get(
                    "expected_sections",
                    [],
                ),
                "required_fact_groups": case.get(
                    "required_fact_groups",
                    [],
                ),
                "forbidden_terms": case.get(
                    "forbidden_terms",
                    [],
                ),
                "expected_behavior": case.get(
                    "expected_behavior",
                    "answer",
                ),

                # Agent运行结果
                "answer": answer,
                "error": error,
                "model": model,
                "embedding_model": (
                    document_metadata[
                        "embedding_model"
                    ]
                ),
                "variant": (
                    "hybrid_rag_with_intent_router"
                ),
                "route_confidence": (
                    route_confidence
                ),
                "selected_intents": [
                    {
                        "intent_id": (
                            intent["intent_id"]
                        ),
                        "name": intent["name"],
                        "combined_score": round(
                            intent[
                                "combined_score"
                            ],
                            6,
                        ),
                        "matched_keywords": (
                            intent[
                                "matched_keywords"
                            ]
                        ),
                    }
                    for intent in selected_intents
                ],
                "retrieved_chunks": [
                    {
                        "id": chunk["id"],
                        "section": (
                            chunk["section"]
                        ),
                        "semantic_score": round(
                            chunk[
                                "semantic_score"
                            ],
                            6,
                        ),
                        "section_boost": round(
                            chunk[
                                "section_boost"
                            ],
                            6,
                        ),
                        "keyword_boost": round(
                            chunk[
                                "keyword_boost"
                            ],
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
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )
            output.flush()

            if args.delay > 0:
                time.sleep(args.delay)

    print("\n" + "=" * 70)
    print(f"结果已保存：{output_path}")
    print(f"成功：{succeeded}")
    print(f"失败：{failed}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())