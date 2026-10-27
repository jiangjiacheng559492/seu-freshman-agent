"""对阿里云百炼执行一次最小连接测试。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    load_dotenv(ROOT / ".env")

    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    model = os.getenv("DASHSCOPE_MODEL", "qwen3.8-flash").strip()
    base_url = os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ).strip()

    if not api_key:
        print("配置错误：.env 中没有 DASHSCOPE_API_KEY。", file=sys.stderr)
        return 2

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "请只回答：百炼连接成功",
                }
            ],
            max_tokens=30,
        )
        answer = response.choices[0].message.content or ""
    except Exception as exc:
        print(f"连接失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"模型：{model}")
    print(f"回答：{answer.strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
