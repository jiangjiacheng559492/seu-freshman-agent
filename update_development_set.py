"""修正开发评测集，并添加错误前提测试。"""

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
    / "development-before-false-premise.jsonl"
)


def load_rows() -> list[dict]:
    rows = []

    for line in DATASET_PATH.read_text(
        encoding="utf-8"
    ).splitlines():
        if line.strip():
            rows.append(
                json.loads(line)
            )

    return rows


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

    # 修改前保存备份。
    shutil.copy2(
        DATASET_PATH,
        BACKUP_PATH,
    )

    rows = load_rows()

    rows_by_id = {
        row["id"]: row
        for row in rows
    }

    # D027：接受模型生成的等价表达。
    d027 = rows_by_id["D027"]

    d027["required_fact_groups"][2] = [
        "无法直接",
        "不能直接",
        "不能从住宿区直接",
        "车辆不能从住宿区直接",
    ]

    # D032：回答查询入口即可，
    # 不强制要求出现“培养方案”。
    d032 = rows_by_id["D032"]

    d032["required_fact_groups"] = [
        [
            "研究生综合服务系统",
        ]
    ]

    # 如果重复运行脚本，先删除旧的 D043—D048。
    new_ids = {
        f"D{number:03d}"
        for number in range(43, 49)
    }

    rows = [
        row
        for row in rows
        if row["id"] not in new_ids
    ]

    new_cases = [
        {
            "id": "D043",
            "category": "false_premise",
            "intent_ids": [
                "guide_conflicts_and_updates",
            ],
            "question": (
                "报到是在2026年9月15日星期三，"
                "我星期三过去就可以了吧？"
            ),
            "expected_sections": [
                "十、待学院确认的信息",
            ],
            "required_fact_groups": [
                [
                    "星期二",
                    "周二",
                ],
                [
                    "星期三",
                    "周三",
                ],
                [
                    "不一致",
                    "冲突",
                ],
                [
                    "最新通知",
                    "学院确认",
                    "学校通知",
                ],
            ],
            "forbidden_terms": [],
            "expected_behavior": "conflict_warning",
        },
        {
            "id": "D044",
            "category": "false_premise",
            "intent_ids": [
                "dormitory_assignment_and_checkin",
            ],
            "question": (
                "学校会安排9月14日提前入住吧？"
                "我提前一天到应该能直接进宿舍。"
            ),
            "expected_sections": [
                "二、报到前准备 / 2. 查询宿舍",
            ],
            "required_fact_groups": [
                [
                    "9月15日",
                    "九月十五日",
                ],
                [
                    "不安排提前到校住宿",
                    "不能提前入住",
                    "无法提前入住",
                ],
            ],
            "forbidden_terms": [],
            "expected_behavior": "answer",
        },
        {
            "id": "D045",
            "category": "false_premise",
            "intent_ids": [
                "tuition_medical_insurance_payment",
            ],
            "question": (
                "学校医保是强制参加的，"
                "而且每年要交300元，对吗？"
            ),
            "expected_sections": [
                "二、报到前准备 / 3. 缴纳相关费用",
            ],
            "required_fact_groups": [
                [
                    "自愿参加",
                    "自愿参保",
                    "并非强制",
                    "不是强制",
                ],
                [
                    "270元",
                    "270 元",
                ],
                [
                    "9月30日",
                    "九月三十日",
                ],
            ],
            "forbidden_terms": [],
            "expected_behavior": "answer",
        },
        {
            "id": "D046",
            "category": "false_premise",
            "intent_ids": [
                "physical_examination",
            ],
            "question": (
                "新生体检必须空腹，"
                "早上吃东西就不能体检了吧？"
            ),
            "expected_sections": [
                "六、报到后的重要事项 / 2. 新生体检",
            ],
            "required_fact_groups": [
                [
                    "无须空腹",
                    "无需空腹",
                    "不用空腹",
                    "不需要空腹",
                ],
                [
                    "避免大荤",
                    "尽量避免大荤",
                ],
            ],
            "forbidden_terms": [],
            "expected_behavior": "answer",
        },
        {
            "id": "D047",
            "category": "false_premise",
            "intent_ids": [
                "railway_station_transport",
            ],
            "question": (
                "从南京南站是不是坐地铁1号线，"
                "然后在东大成贤学院站下车？"
            ),
            "expected_sections": [
                "三、到校交通与报到地点 / "
                "2. 南京站或南京南站出发",
            ],
            "required_fact_groups": [
                [
                    "地铁3号线",
                    "3号线",
                    "三号线",
                ],
                [
                    "林场",
                ],
                [
                    "东大成贤学院站",
                ],
                [
                    "1号口",
                    "一号口",
                ],
            ],
            "forbidden_terms": [],
            "expected_behavior": "answer",
        },
        {
            "id": "D048",
            "category": "false_premise",
            "intent_ids": [
                "campus_life_and_delivery",
            ],
            "question": (
                "江北校区食堂只能刷校园卡，"
                "不能用微信和支付宝，对吧？"
            ),
            "expected_sections": [
                "五、住宿与江北校区生活 / "
                "2. 食堂与超市",
            ],
            "required_fact_groups": [
                [
                    "微信",
                ],
                [
                    "支付宝",
                ],
                [
                    "暂时无法使用校园一卡通",
                    "不能使用校园一卡通",
                    "无法使用校园一卡通",
                    "不能刷校园卡",
                ],
            ],
            "forbidden_terms": [],
            "expected_behavior": "answer",
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
            f"存在重复 ID：{duplicate_ids}"
        )

    save_rows(rows)

    print("开发集更新完成。")
    print(f"题目数量：{len(rows)}")
    print(f"最后一个ID：{rows[-1]['id']}")
    print(f"备份文件：{BACKUP_PATH}")


if __name__ == "__main__":
    main()