"""使用15道测试题评测本地 RAG Agent。"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from local_rag_agent import (
    embed_question,
    generate_answer,
    load_index,
    load_settings,
    retrieve,
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
    index_data = load_index()
    questions = load_questions()

    metadata = index_data["metadata"]
    chunks = index_data["chunks"]

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    RESULTS_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = RESULTS_DIR / f"local-rag-{timestamp}.jsonl"

    print(f"模型：{model}")
    print(f"题目数量：{len(questions)}")
    print(f"结果文件：{output_path}")

    with output_path.open("w", encoding="utf-8") as output:
        for position, case in enumerate(questions, start=1):
            question = case["question"]

            print("\n" + "=" * 70)
            print(f"[{position:02d}/{len(questions)}] {case['id']}")
            print(f"问题：{question}")

            try:
                query_embedding = embed_question(
                    client=client,
                    embedding_model=metadata["embedding_model"],
                    dimensions=metadata["dimensions"],
                    question=question,
                )

                retrieved_chunks = retrieve(
                    query_embedding=query_embedding,
                    chunks=chunks,
                )

                answer = generate_answer(
                    client=client,
                    model=model,
                    question=question,
                    retrieved_chunks=retrieved_chunks,
                )

                error = None

                print("检索章节：")
                for chunk in retrieved_chunks:
                    print(
                        f"- {chunk['section']} "
                        f"({chunk['score']:.4f})"
                    )

                print(f"\n回答：\n{answer}")

            except Exception as exc:
                answer = ""
                retrieved_chunks = []
                error = f"{type(exc).__name__}: {exc}"
                print(f"错误：{error}")

            result = {
                "id": case["id"],
                "category": case["category"],
                "question": question,
                "answer": answer,
                "error": error,
                "model": model,
                "embedding_model": metadata["embedding_model"],
                "variant": "local_rag_with_bailian",
                "retrieved_chunks": [
                    {
                        "id": chunk["id"],
                        "section": chunk["section"],
                        "score": round(chunk["score"], 6),
                        "text": chunk["text"],
                    }
                    for chunk in retrieved_chunks
                ],
            }

            output.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )
            output.flush()

            # 稍作间隔，减少短时间连续请求触发限流的可能。
            time.sleep(0.5)

    failed_count = 0

    with output_path.open("r", encoding="utf-8") as file:
        for line in file:
            if json.loads(line).get("error"):
                failed_count += 1

    print("\n" + "=" * 70)
    print(f"评测结果已保存：{output_path}")
    print(f"成功：{len(questions) - failed_count}")
    print(f"失败：{failed_count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())