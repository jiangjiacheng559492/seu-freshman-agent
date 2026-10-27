"""修正 D049，并添加领域外问题测试。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent

DATASET_PATH = (
    ROOT
    / "evals"
    / "development.jsonl"
)

BACKUP_PATH = (
    ROOT
    / "evals"
    / "development-before-out-of-domain.jsonl"
)


def load_rows() -> list[dict]:
    return [
        json.loads(line)
        for line in DATASET_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def save_rows(rows: list[dict]) -> None:
    content = "\n".join(
        json.dumps(
            row,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for row in rows
    )

    DATASET_PATH.write_text(
        content + "\n",
        encoding="utf-8",
    )


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"找不到开发集：{DATASET_PATH}"
        )

    shutil.copy2(
        DATASET_PATH,
        BACKUP_PATH,
    )

    rows = load_rows()

    rows_by_id = {
        row["id"]: row
        for row in rows
    }

    # D049 只询问报到地点，
    # 不再强制要求回答报到日期。
    rows_by_id["D049"][
        "required_fact_groups"
    ] = [
        [
            "汇川楼C102",
            "C102",
        ]
    ]

    new_ids = {
        f"D{number:03d}"
        for number in range(55, 61)
    }

    # 允许脚本重复执行。
    rows = [
        row
        for row in rows
        if row["id"] not in new_ids
    ]

    new_cases = [
        {
            "id": "D055",
            "category": "out_of_domain",
            "intent_ids": [],
            "question": (
                "明天南京会不会下雨，"
                "气温大概是多少？"
            ),
            "expected_sections": [],
            "required_fact_groups": [],
            "forbidden_terms": [],
            "expected_behavior": "abstain",
        },
        {
            "id": "D056",
            "category": "out_of_domain",
            "intent_ids": [],
            "question": (
                "帮我用Python写一个冒泡排序程序，"
                "并逐行解释代码。"
            ),
            "expected_sections": [],
            "required_fact_groups": [],
            "forbidden_terms": [],
            "expected_behavior": "abstain",
        },
        {
            "id": "D057",
            "category": "out_of_domain",
            "intent_ids": [],
            "question": (
                "推荐几家南京最值得去的餐厅，"
                "顺便告诉我人均消费。"
            ),
            "expected_sections": [],
            "required_fact_groups": [],
            "forbidden_terms": [],
            "expected_behavior": "abstain",
        },
        {
            "id": "D058",
            "category": "out_of_domain",
            "intent_ids": [],
            "question": (
                "你觉得我现在应该买哪只股票，"
                "下个月哪只股票会涨？"
            ),
            "expected_sections": [],
            "required_fact_groups": [],
            "forbidden_terms": [],
            "expected_behavior": "abstain",
        },
        {
            "id": "D059",
            "category": "out_of_domain",
            "intent_ids": [],
            "question": (
                "东南大学本科生转专业有哪些条件，"
                "大一能申请几次？"
            ),
            "expected_sections": [],
            "required_fact_groups": [],
            "forbidden_terms": [],
            "expected_behavior": "abstain",
        },
        {
            "id": "D060",
            "category": "out_of_domain",
            "intent_ids": [],
            "question": (
                "请给学院所有导师的科研水平"
                "做一个排名，并说明谁最差。"
            ),
            "expected_sections": [],
            "required_fact_groups": [],
            "forbidden_terms": [],
            "expected_behavior": "abstain",
        },
    ]

    rows.extend(new_cases)

    ids = [
        row["id"]
        for row in rows
    ]

    duplicate_ids = sorted(
        {
            case_id
            for case_id in ids
            if ids.count(case_id) > 1
        }
    )

    if duplicate_ids:
        raise RuntimeError(
            f"存在重复ID：{duplicate_ids}"
        )

    save_rows(rows)

    print("领域外测试更新完成。")
    print(f"题目数量：{len(rows)}")
    print(f"最后一个ID：{rows[-1]['id']}")
    print(f"备份文件：{BACKUP_PATH}")


if __name__ == "__main__":
    main()