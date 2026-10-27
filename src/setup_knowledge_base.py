"""上传新生指南，并创建供 file_search 使用的向量库。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
GUIDE_PATH = ROOT / "data" / "knowledge" / "freshman_guide.md"


def save_env_value(name: str, value: str) -> None:
    """在不打印 API Key 的前提下新增或更新一项本地配置。"""
    lines = ENV_PATH.read_text(encoding="utf-8").splitlines()
    prefix = f"{name}="
    replacement = f"{name}={value}"

    for index, line in enumerate(lines):
        if line.startswith(prefix):
            lines[index] = replacement
            break
    else:
        lines.append(replacement)

    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    load_dotenv(ENV_PATH)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    existing_store_id = os.getenv("OPENAI_VECTOR_STORE_ID", "").strip()

    if not api_key or api_key == "replace_with_your_new_key":
        print("配置错误：.env 中没有可用的 OPENAI_API_KEY。", file=sys.stderr)
        return 2

    if existing_store_id:
        print("知识库已经配置，无需重复上传。")
        print(f"Vector Store ID：{existing_store_id}")
        return 0

    if not GUIDE_PATH.exists():
        print(f"找不到指南文件：{GUIDE_PATH}", file=sys.stderr)
        return 2

    client = OpenAI(api_key=api_key)

    print(f"1/3 正在上传：{GUIDE_PATH.name}")
    with GUIDE_PATH.open("rb") as guide_file:
        uploaded_file = client.files.create(
            file=guide_file,
            purpose="assistants",
        )

    print("2/3 正在创建向量库……")
    vector_store = client.vector_stores.create(
        name="SEU 研究生新生指南",
    )

    print("3/3 正在解析并建立检索索引……")
    vector_store_file = client.vector_stores.files.create_and_poll(
        vector_store_id=vector_store.id,
        file_id=uploaded_file.id,
    )

    if vector_store_file.status != "completed":
        print(
            f"知识库处理未完成，当前状态：{vector_store_file.status}",
            file=sys.stderr,
        )
        print(f"File ID：{uploaded_file.id}", file=sys.stderr)
        print(f"Vector Store ID：{vector_store.id}", file=sys.stderr)
        return 1

    save_env_value("OPENAI_VECTOR_STORE_ID", vector_store.id)

    print("\n知识库创建成功！")
    print(f"文件：{GUIDE_PATH.name}")
    print(f"File ID：{uploaded_file.id}")
    print(f"Vector Store ID：{vector_store.id}")
    print("Vector Store ID 已自动保存到 .env。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
