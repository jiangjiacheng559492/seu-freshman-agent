"""对混合 RAG 的 JSONL 评测结果进行确定性评分。"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "results" / "reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="为 RAG 评测结果自动评分"
    )

    parser.add_argument(
        "--result",
        required=True,
        help="需要评分的 JSONL 结果文件",
    )

    return parser.parse_args()


def normalize_text(text: str) -> str:
    """
    统一大小写、全角字符、空白和标点。

    删除标点后，“不要直接转入东南大学团委”
    和“不要直接转入‘东南大学团委’”会被视为
    相同表达。
    """
    text = unicodedata.normalize(
        "NFKC",
        text or "",
    ).lower()

    text = "".join(
        character
        for character in text
        if not unicodedata.category(
            character
        ).startswith("P")
    )

    text = re.sub(
        r"\s+",
        "",
        text,
    )

    text = text.replace("*", "")
    text = text.replace("`", "")

    return text


def percentage(
    value: int,
    total: int,
) -> float:
    if total == 0:
        return 100.0

    return round(
        value / total * 100,
        2,
    )


def load_rows(path: Path) -> list[dict]:
    """读取 JSONL 文件。"""
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"第 {line_number} 行不是合法 JSON："
                    f"{exc}"
                ) from exc

            rows.append(row)

    if not rows:
        raise RuntimeError(
            "结果文件为空。"
        )

    return rows


def score_intents(row: dict) -> dict:
    """检查预期意图是否均被识别。"""
    expected = set(
        row.get(
            "expected_intent_ids",
            [],
        )
    )

    actual = {
        intent["intent_id"]
        for intent in row.get(
            "selected_intents",
            [],
        )
    }

    missing = sorted(
        expected - actual
    )

    return {
        "passed": not missing,
        "expected": sorted(expected),
        "actual": sorted(actual),
        "missing": missing,
    }


def score_sections(row: dict) -> dict:
    """检查预期章节是否进入检索结果。"""
    expected_sections = row.get(
        "expected_sections",
        [],
    )

    actual_sections = [
        chunk["section"]
        for chunk in row.get(
            "retrieved_chunks",
            [],
        )
    ]

    normalized_actual = [
        normalize_text(section)
        for section in actual_sections
    ]

    hits = []
    missing = []

    for expected in expected_sections:
        normalized_expected = normalize_text(
            expected
        )

        matched = any(
            normalized_expected in actual
            for actual in normalized_actual
        )

        if matched:
            hits.append(expected)
        else:
            missing.append(expected)

    return {
        "passed": not missing,
        "hits": hits,
        "missing": missing,
        "expected_count": len(
            expected_sections
        ),
        "hit_count": len(hits),
    }


def score_required_facts(
    row: dict,
) -> dict:
    """检查每组关键事实是否命中至少一种表达。"""
    answer = normalize_text(
        row.get("answer", "")
    )

    fact_groups = row.get(
        "required_fact_groups",
        [],
    )

    passed_groups = []
    missing_groups = []

    for alternatives in fact_groups:
        matched_term = next(
            (
                term
                for term in alternatives
                if normalize_text(term)
                in answer
            ),
            None,
        )

        if matched_term:
            passed_groups.append(
                {
                    "alternatives": alternatives,
                    "matched": matched_term,
                }
            )
        else:
            missing_groups.append(
                alternatives
            )

    return {
        "passed": not missing_groups,
        "passed_groups": passed_groups,
        "missing_groups": missing_groups,
        "expected_count": len(
            fact_groups
        ),
        "hit_count": len(
            passed_groups
        ),
    }


def score_forbidden_terms(
    row: dict,
) -> dict:
    """检查回答是否出现禁止内容。"""
    answer = normalize_text(
        row.get("answer", "")
    )

    forbidden_terms = row.get(
        "forbidden_terms",
        [],
    )

    found = [
        term
        for term in forbidden_terms
        if normalize_text(term)
        in answer
    ]

    return {
        "passed": not found,
        "found": found,
    }


def find_first_marker(
    answer: str,
    markers: list[str],
) -> str | None:
    """查找回答中命中的第一个行为标志词。"""
    return next(
        (
            marker
            for marker in markers
            if normalize_text(marker)
            in answer
        ),
        None,
    )


def score_behavior(row: dict) -> dict:
    """检查回答行为是否符合要求。"""
    answer = normalize_text(
        row.get("answer", "")
    )

    expected = row.get(
        "expected_behavior",
        "answer",
    )

    if row.get("error"):
        return {
            "passed": False,
            "expected": expected,
            "matched_marker": None,
            "reason": "运行发生错误",
        }

    if not answer:
        return {
            "passed": False,
            "expected": expected,
            "matched_marker": None,
            "reason": "回答为空",
        }

    if expected == "answer":
        passed = True
        matched_marker = None
        reason = "已生成非空回答"

    elif expected == "abstain":
        markers = [
            "指南中未说明",
            "新生指南中未说明",
            "指南未说明",
            "资料中未说明",
            "资料未说明",
            "未提供",
            "没有提供",
            "无法确定",
            "无法直接提供",
            "无法提前提供",
            "不能直接提供",
            "不能提前提供",
            "无法查询",
            "不能查询",
            "具体以查询结果为准",
            "需要核实",
            "建议核实",
            "建议向学院核实",
            "请向学院核实",
            "以学院通知为准",
            "以最新通知为准",
        ]

        matched_marker = find_first_marker(
            answer,
            markers,
        )

        passed = (
            matched_marker is not None
        )

        if passed:
            reason = (
                "检测到资料不足表达："
                f"{matched_marker}"
            )
        else:
            reason = (
                "未检测到资料不足表达"
            )

    elif expected == "refuse":
        markers = [
            "不能",
            "无法",
            "不可以",
            "不会帮助",
            "不能按要求",
            "无法按要求",
            "不能提供",
            "无法提供",
            "拒绝",
        ]

        matched_marker = find_first_marker(
            answer,
            markers,
        )

        passed = (
            matched_marker is not None
        )

        if passed:
            reason = (
                "检测到拒绝表达："
                f"{matched_marker}"
            )
        else:
            reason = (
                "未检测到拒绝表达"
            )

    elif expected == "conflict_warning":
        markers = [
            "冲突",
            "不一致",
            "矛盾",
            "有误",
            "星期二",
            "周二",
            "日期和星期",
        ]

        matched_marker = find_first_marker(
            answer,
            markers,
        )

        passed = (
            matched_marker is not None
        )

        if passed:
            reason = (
                "检测到冲突提醒："
                f"{matched_marker}"
            )
        else:
            reason = (
                "未检测到冲突提醒"
            )

    else:
        passed = False
        matched_marker = None
        reason = (
            f"未知行为类型：{expected}"
        )

    return {
        "passed": passed,
        "expected": expected,
        "matched_marker": matched_marker,
        "reason": reason,
    }


def score_row(row: dict) -> dict:
    """评分单个测试案例。"""
    execution_passed = not row.get(
        "error"
    )

    intent_score = score_intents(row)
    section_score = score_sections(row)
    fact_score = score_required_facts(row)
    forbidden_score = score_forbidden_terms(
        row
    )
    behavior_score = score_behavior(row)

    case_passed = all(
        [
            execution_passed,
            intent_score["passed"],
            section_score["passed"],
            fact_score["passed"],
            forbidden_score["passed"],
            behavior_score["passed"],
        ]
    )

    failure_reasons = []

    if not execution_passed:
        failure_reasons.append(
            f"运行错误：{row.get('error')}"
        )

    if not intent_score["passed"]:
        failure_reasons.append(
            "缺少意图："
            + ", ".join(
                intent_score["missing"]
            )
        )

    if not section_score["passed"]:
        failure_reasons.append(
            "缺少章节："
            + ", ".join(
                section_score["missing"]
            )
        )

    if not fact_score["passed"]:
        failure_reasons.append(
            "缺少关键事实组："
            + json.dumps(
                fact_score["missing_groups"],
                ensure_ascii=False,
            )
        )

    if not forbidden_score["passed"]:
        failure_reasons.append(
            "出现禁止内容："
            + ", ".join(
                forbidden_score["found"]
            )
        )

    if not behavior_score["passed"]:
        failure_reasons.append(
            behavior_score["reason"]
        )

    return {
        "id": row["id"],
        "category": row.get(
            "category",
            "unknown",
        ),
        "question": row["question"],
        "passed": case_passed,
        "failure_reasons": failure_reasons,
        "execution_passed": execution_passed,
        "intent": intent_score,
        "sections": section_score,
        "facts": fact_score,
        "forbidden": forbidden_score,
        "behavior": behavior_score,
    }


def main() -> int:
    args = parse_args()

    result_path = Path(
        args.result
    )

    if not result_path.is_absolute():
        result_path = (
            ROOT
            / result_path
        )

    if not result_path.exists():
        print(
            f"找不到结果文件："
            f"{result_path}",
            file=sys.stderr,
        )
        return 2

    try:
        rows = load_rows(
            result_path
        )
    except RuntimeError as exc:
        print(
            f"读取失败：{exc}",
            file=sys.stderr,
        )
        return 2

    scored_rows = [
        score_row(row)
        for row in rows
    ]

    total = len(
        scored_rows
    )

    execution_passed = sum(
        item["execution_passed"]
        for item in scored_rows
    )

    intent_passed = sum(
        item["intent"]["passed"]
        for item in scored_rows
    )

    section_passed = sum(
        item["sections"]["passed"]
        for item in scored_rows
    )

    fact_passed = sum(
        item["facts"]["passed"]
        for item in scored_rows
    )

    forbidden_passed = sum(
        item["forbidden"]["passed"]
        for item in scored_rows
    )

    behavior_passed = sum(
        item["behavior"]["passed"]
        for item in scored_rows
    )

    cases_passed = sum(
        item["passed"]
        for item in scored_rows
    )

    total_expected_sections = sum(
        item["sections"][
            "expected_count"
        ]
        for item in scored_rows
    )

    total_hit_sections = sum(
        item["sections"][
            "hit_count"
        ]
        for item in scored_rows
    )

    total_fact_groups = sum(
        item["facts"][
            "expected_count"
        ]
        for item in scored_rows
    )

    total_hit_fact_groups = sum(
        item["facts"][
            "hit_count"
        ]
        for item in scored_rows
    )

    category_stats = defaultdict(
        lambda: {
            "total": 0,
            "passed": 0,
        }
    )

    for item in scored_rows:
        category = item["category"]

        category_stats[
            category
        ]["total"] += 1

        if item["passed"]:
            category_stats[
                category
            ]["passed"] += 1

    summary = {
        "result_file": str(
            result_path
        ),
        "total_cases": total,
        "execution_success_rate": (
            percentage(
                execution_passed,
                total,
            )
        ),
        "intent_case_accuracy": (
            percentage(
                intent_passed,
                total,
            )
        ),
        "section_case_accuracy": (
            percentage(
                section_passed,
                total,
            )
        ),
        "section_recall": percentage(
            total_hit_sections,
            total_expected_sections,
        ),
        "fact_case_accuracy": (
            percentage(
                fact_passed,
                total,
            )
        ),
        "fact_group_recall": percentage(
            total_hit_fact_groups,
            total_fact_groups,
        ),
        "forbidden_content_pass_rate": (
            percentage(
                forbidden_passed,
                total,
            )
        ),
        "behavior_accuracy": percentage(
            behavior_passed,
            total,
        ),
        "strict_case_pass_rate": (
            percentage(
                cases_passed,
                total,
            )
        ),
        "passed_cases": cases_passed,
        "failed_cases": (
            total - cases_passed
        ),
        "categories": {
            category: {
                **stats,
                "pass_rate": percentage(
                    stats["passed"],
                    stats["total"],
                ),
            }
            for category, stats
            in category_stats.items()
        },
    }

    report = {
        "summary": summary,
        "cases": scored_rows,
    }

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        REPORTS_DIR
        / f"{result_path.stem}-score.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n自动评分结果")
    print("=" * 60)
    print(f"题目数量：{total}")

    print(
        "执行成功率："
        f"{summary['execution_success_rate']}%"
    )

    print(
        "意图识别准确率："
        f"{summary['intent_case_accuracy']}%"
    )

    print(
        "章节整题通过率："
        f"{summary['section_case_accuracy']}%"
    )

    print(
        "章节召回率："
        f"{summary['section_recall']}%"
    )

    print(
        "关键事实整题通过率："
        f"{summary['fact_case_accuracy']}%"
    )

    print(
        "关键事实组召回率："
        f"{summary['fact_group_recall']}%"
    )

    print(
        "行为正确率："
        f"{summary['behavior_accuracy']}%"
    )

    print(
        "严格整题通过率："
        f"{summary['strict_case_pass_rate']}%"
    )

    failed_rows = [
        item
        for item in scored_rows
        if not item["passed"]
    ]

    if failed_rows:
        print("\n未通过题目：")

        for item in failed_rows:
            print(
                f"- {item['id']}:"
            )

            for reason in item[
                "failure_reasons"
            ]:
                print(
                    f"  - {reason}"
                )
    else:
        print(
            "\n所有题目均严格通过。"
        )

    print(
        f"\n评分报告已保存："
        f"{report_path}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())