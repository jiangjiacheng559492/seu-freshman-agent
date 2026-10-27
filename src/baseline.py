"""不使用新生指南知识库的基线问答程序。

用途：先记录模型在没有 RAG 时对评测题的回答，再与后续 file_search
版本比较。真实 API Key 只从项目根目录的 .env 或系统环境变量读取。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = ROOT / "evals" / "questions.jsonl"
RESULTS_DIR = ROOT / "results"

BASELINE_INSTRUCTIONS = """
你是一名面向东南大学生命科学与技术学院研究生新生的问答助手。
请直接回答用户问题；不确定时明确说明不确定，不要编造联系方式、日期、
政策或例外。当前版本没有接入学院指南，因此不要声称自己查阅了学院文件。
回答使用中文，保持简洁。
""".strip()


def load_settings() -> tuple[str, str]:
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "gpt-6-luna").strip()

    if not api_key or api_key == "replace_with_your_new_key":
        raise RuntimeError(
            "未检测到可用的 OPENAI_API_KEY。请复制 .env.example 为 .env，"
            "并在本机编辑器中填入新密钥。不要把密钥发到聊天中。"
        )
    return api_key, model


def ask(client: OpenAI, model: str, question: str) -> str:
    response = client.responses.create(
        model=model,
        instructions=BASELINE_INSTRUCTIONS,
        input=question,
    )
    return response.output_text.strip()


def load_eval_cases() -> list[dict]:
    with EVAL_PATH.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def run_batch(client: OpenAI, model: str) -> Path:
    cases = load_eval_cases()
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = RESULTS_DIR / f"baseline-{timestamp}.jsonl"

    with output_path.open("w", encoding="utf-8") as output:
        for index, case in enumerate(cases, start=1):
            print(f"[{index:02d}/{len(cases)}] {case['id']}: {case['question']}")
            try:
                answer = ask(client, model, case["question"])
                error = None
            except Exception as exc:  # 保留失败样本，方便排查限额或网络问题
                answer = ""
                error = f"{type(exc).__name__}: {exc}"

            result = {
                "id": case["id"],
                "category": case["category"],
                "question": case["question"],
                "answer": answer,
                "error": error,
                "model": model,
                "variant": "baseline_without_knowledge_base",
            }
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行无知识库基线问答")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--question", help="只回答一个问题")
    mode.add_argument("--all", action="store_true", help="运行全部15道评测题")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        api_key, model = load_settings()
    except RuntimeError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2

    client = OpenAI(api_key=api_key)

    if args.all:
        output_path = run_batch(client, model)
        print(f"\n评测结果已保存：{output_path}")
        return 0

    question = args.question or input("请输入问题：").strip()
    if not question:
        print("问题不能为空。", file=sys.stderr)
        return 2

    print(f"\n模型：{model}\n")
    print(ask(client, model, question))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

