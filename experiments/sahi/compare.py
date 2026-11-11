"""여러 실험(A/B/C/D 또는 E/F/G)의 summary.json/metrics.json을 모아 비교표를 만든다.

metrics.json이 없는 실험(아직 evaluate.py를 실행하지 않은 경우)은 mAP 관련 칸을
빈 값으로 두고 latency만 표시한다 — B 결과가 없어도 A/C/D만으로 비교표를 만들 수 있다.

2차 실험(E/F/G)을 비교할 때는 반드시 동일한 v2 test set으로 만든 결과만 넘겨야
한다(이 스크립트는 그 전제를 강제하지 않으므로 호출하는 사람이 지켜야 한다).

사용 예:
    python experiments/sahi/compare.py \\
        --result A=experiments/sahi/outputs/A_standard_640 \\
        --result C=experiments/sahi/outputs/C_sahi_640 \\
        --result D=experiments/sahi/outputs/D_hybrid_640 \\
        --output experiments/sahi/outputs/comparison
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 작은 PPE 객체 확인이 이 실험의 핵심이므로 기본적으로 이 class들의 AP50을 별도 컬럼으로 보여준다.
DEFAULT_HIGHLIGHT_CLASSES = ("head", "gloves", "helmet", "safety_vest", "harness_body")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAHI 실험 결과 비교표 생성기")
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        metavar="NAME=OUTPUT_DIR",
        help="예: A=experiments/sahi/outputs/A_standard_640. 여러 번 지정 가능.",
    )
    parser.add_argument(
        "--highlight-classes",
        default=",".join(DEFAULT_HIGHLIGHT_CLASSES),
        help="AP50을 별도 컬럼으로 보여줄 class 이름(쉼표 구분)",
    )
    parser.add_argument("--output", required=True, type=Path, help="comparison.csv/.md를 저장할 경로(확장자 제외)")
    return parser.parse_args()


def _parse_result_args(raw_results: list[str]) -> dict[str, Path]:
    results: dict[str, Path] = {}
    for raw in raw_results:
        if "=" not in raw:
            raise ValueError(f"--result must be NAME=OUTPUT_DIR, got: {raw}")
        name, raw_path = raw.split("=", maxsplit=1)
        name = name.strip()
        path = Path(raw_path.strip())
        if not name:
            raise ValueError(f"--result name must not be empty: {raw}")
        if not path.is_dir():
            raise FileNotFoundError(f"result directory not found: {path}")
        results[name] = path
    return results


def _load_json_or_none(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def _build_row(
    name: str,
    output_dir: Path,
    highlight_classes: list[str],
) -> dict[str, Any]:
    summary = _load_json_or_none(output_dir / "summary.json")
    metrics = _load_json_or_none(output_dir / "metrics.json")

    row: dict[str, Any] = {"experiment": name}

    overall = (metrics or {}).get("overall", {})
    row["map50"] = overall.get("map50")
    row["map50_95"] = overall.get("map50_95")
    row["ap_small"] = overall.get("ap_small")
    row["ap_medium"] = overall.get("ap_medium")
    row["ap_large"] = overall.get("ap_large")

    per_class = (metrics or {}).get("per_class", {})
    for class_name in highlight_classes:
        class_metrics = per_class.get(class_name, {})
        row[f"{class_name}_ap50"] = class_metrics.get("ap50")

    latency = (summary or {}).get("latency", {})
    row["latency_mean_ms"] = latency.get("mean_ms")
    row["latency_p50_ms"] = latency.get("p50_ms")
    row["latency_p95_ms"] = latency.get("p95_ms")

    if metrics is None:
        logger.warning("metrics.json not found for %s (%s); mAP columns left blank", name, output_dir)
    if summary is None:
        logger.warning("summary.json not found for %s (%s); latency columns left blank", name, output_dir)

    return row


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_csv(rows: list[dict[str, Any]], columns: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_format_value(row.get(column)) for column in columns])


def write_markdown(rows: list[dict[str, Any]], columns: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, separator]
    for row in rows:
        lines.append("| " + " | ".join(_format_value(row.get(column)) for column in columns) + " |")

    with output_path.open("w", encoding="utf-8") as md_file:
        md_file.write("\n".join(lines) + "\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    results = _parse_result_args(args.result)
    highlight_classes = [name.strip() for name in args.highlight_classes.split(",") if name.strip()]

    rows = [
        _build_row(name, output_dir, highlight_classes)
        for name, output_dir in results.items()
    ]

    columns = [
        "experiment",
        "map50",
        "map50_95",
        "ap_small",
        "ap_medium",
        "ap_large",
        *[f"{class_name}_ap50" for class_name in highlight_classes],
        "latency_mean_ms",
        "latency_p50_ms",
        "latency_p95_ms",
    ]

    csv_path = args.output.with_suffix(".csv")
    md_path = args.output.with_suffix(".md")
    write_csv(rows, columns, csv_path)
    write_markdown(rows, columns, md_path)

    logger.info("saved: %s", csv_path)
    logger.info("saved: %s", md_path)


if __name__ == "__main__":
    main()
