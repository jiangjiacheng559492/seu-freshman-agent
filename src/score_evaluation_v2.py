"""分别统计端到端质量和内部组件质量。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from score_evaluation import (
    REPORTS_DIR,
    ROOT,
    load_rows,
    percentage,
    score_row,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="分别计算端到端和组件评测指标"
    )

    parser.add_argument(
        "--result",
        required=True,
        help="需要评分的JSONL结果文件",
    )

    return parser.parse_args()


def is_end_to_end_passed(item: dict) -> bool:
    """只判断用户最终获得的结果是否合格。"""
    return all(
        [
            item["execution_passed"],
            item["sections"]["passed"],
            item["facts"]["passed"],
            item["forbidden"]["passed"],
            item["behavior"]["passed"],
        ]
    )


def main() -> int:
    args = parse_args()
    result_path = Path(args.result)

    if not result_path.is_absolute():
        result_path = ROOT / result_path

    if not result_path.exists():
        print(
            f"找不到结果文件：{result_path}",
            file=sys.stderr,
        )
        return 2

    try:
        rows = load_rows(result_path)
    except RuntimeError as exc:
        print(f"读取失败：{exc}", file=sys.stderr)
        return 2

    scored_rows = []

    for row in rows:
        scored = score_row(row)

        scored["end_to_end_passed"] = (
            is_end_to_end_passed(scored)
        )

        # 原版passed要求意图、检索和回答全部通过。
        scored["full_pipeline_passed"] = (
            scored["passed"]
        )

        scored_rows.append(scored)

    total = len(scored_rows)

    end_to_end_passed = sum(
        item["end_to_end_passed"]
        for item in scored_rows
    )

    full_pipeline_passed = sum(
        item["full_pipeline_passed"]
        for item in scored_rows
    )

    routing_passed = sum(
        item["intent"]["passed"]
        for item in scored_rows
    )

    retrieval_passed = sum(
        item["sections"]["passed"]
        for item in scored_rows
    )

    answer_fact_passed = sum(
        item["facts"]["passed"]
        for item in scored_rows
    )

    behavior_passed = sum(
        item["behavior"]["passed"]
        for item in scored_rows
    )

    end_to_end_failures = [
        item
        for item in scored_rows
        if not item["end_to_end_passed"]
    ]

    routing_only_failures = [
        item
        for item in scored_rows
        if (
            item["end_to_end_passed"]
            and not item["intent"]["passed"]
        )
    ]

    summary = {
        "result_file": str(result_path),
        "total_cases": total,

        # 用户实际感受到的质量
        "end_to_end_pass_rate": percentage(
            end_to_end_passed,
            total,
        ),
        "end_to_end_passed": end_to_end_passed,
        "end_to_end_failed": (
            total - end_to_end_passed
        ),

        # 内部组件质量
        "intent_accuracy": percentage(
            routing_passed,
            total,
        ),
        "retrieval_case_accuracy": percentage(
            retrieval_passed,
            total,
        ),
        "answer_fact_accuracy": percentage(
            answer_fact_passed,
            total,
        ),
        "behavior_accuracy": percentage(
            behavior_passed,
            total,
        ),

        # 最严格：所有组件和最终答案都必须通过
        "full_pipeline_pass_rate": percentage(
            full_pipeline_passed,
            total,
        ),
        "full_pipeline_passed": (
            full_pipeline_passed
        ),
    }

    report = {
        "summary": summary,
        "end_to_end_failures": (
            end_to_end_failures
        ),
        "routing_only_failures": (
            routing_only_failures
        ),
        "cases": scored_rows,
    }

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        REPORTS_DIR
        / f"{result_path.stem}-score-v2.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n分层评分结果")
    print("=" * 60)
    print(f"题目数量：{total}")
    print(
        "端到端回答通过率："
        f"{summary['end_to_end_pass_rate']}%"
    )
    print(
        "意图识别准确率："
        f"{summary['intent_accuracy']}%"
    )
    print(
        "章节检索整题准确率："
        f"{summary['retrieval_case_accuracy']}%"
    )
    print(
        "答案关键事实准确率："
        f"{summary['answer_fact_accuracy']}%"
    )
    print(
        "回答行为准确率："
        f"{summary['behavior_accuracy']}%"
    )
    print(
        "完整流水线通过率："
        f"{summary['full_pipeline_pass_rate']}%"
    )

    if end_to_end_failures:
        print("\n端到端失败题目：")

        for item in end_to_end_failures:
            print(f"- {item['id']}")

            for reason in item["failure_reasons"]:
                # 意图失败不属于端到端失败原因。
                if not reason.startswith("缺少意图"):
                    print(f"  - {reason}")
    else:
        print("\n没有端到端失败题目。")

    if routing_only_failures:
        print("\n回答正确但意图路由未命中的题目：")

        for item in routing_only_failures:
            print(
                f"- {item['id']}: "
                f"{item['question']}"
            )

    print(f"\n分层评分报告已保存：{report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())