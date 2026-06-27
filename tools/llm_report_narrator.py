from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import os
from typing import Any, Dict, List, Sequence


def _read_csv(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _to_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def _sort_numeric(rows: Sequence[Dict[str, Any]], key: str, reverse: bool = False) -> List[Dict[str, Any]]:
    return sorted(
        [row for row in rows if _to_float(row.get(key)) is not None],
        key=lambda row: float(row[key]),
        reverse=reverse,
    )


def _table(rows: Sequence[Dict[str, Any]], columns: Sequence[str], limit: int = 8) -> str:
    rows = list(rows[:limit])
    if not rows:
        return "_No rows._\n"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines) + "\n"


def _load_packet(insight_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    return {
        "scorecard": _read_csv(os.path.join(insight_dir, "policy_scorecard.csv")),
        "anomalies": _read_csv(os.path.join(insight_dir, "anomaly_runs.csv")),
        "pareto": _read_csv(os.path.join(insight_dir, "pareto_front.csv")),
        "recommendations": _read_csv(os.path.join(insight_dir, "recommended_next_sweeps.csv")),
    }


def write_narrative(path: str, packet: Dict[str, List[Dict[str, Any]]]) -> None:
    scorecard = packet["scorecard"]
    anomalies = packet["anomalies"]
    pareto = packet["pareto"]
    recommendations = packet["recommendations"]

    best_waf = _sort_numeric(scorecard, "waf_mean")[:5]
    best_wear = _sort_numeric(scorecard, "wear_std_mean")[:5]

    with open(path, "w", encoding="utf-8") as f:
        f.write("# AI-Ready SSD GC Validation Narrative\n\n")
        f.write("## Short Summary\n\n")
        f.write(
            "This packet turns simulator outputs into an engineering narrative: "
            "which policies look efficient, which policies balance wear, which runs look anomalous, "
            "and which follow-up experiments should be run next.\n\n"
        )
        f.write("## Evidence Highlights\n\n")
        f.write("### Lowest Mean WAF\n\n")
        f.write(_table(best_waf, ["scenario", "policy", "runs", "waf_mean", "wear_std_mean", "gc_count_mean"]))
        f.write("\n### Lowest Mean Wear Spread\n\n")
        f.write(_table(best_wear, ["scenario", "policy", "runs", "wear_std_mean", "waf_mean", "gc_count_mean"]))
        f.write("\n### TRIM Activity\n\n")
        trim_rows = [row for row in scorecard if _to_float(row.get("trim_ops_mean")) is not None and float(row.get("trim_ops_mean") or 0.0) > 0.0]
        f.write(_table(trim_rows, ["scenario", "policy", "runs", "trim_ops_mean", "trim_invalidated_pages_mean", "trim_misses_mean", "retrim_count_mean"]))
        f.write("\n### Anomaly Candidates\n\n")
        f.write(_table(anomalies, ["scenario", "policy", "seed", "waf", "wear_std", "gc_count", "anomaly_reasons"]))
        f.write("\n### Pareto Candidates\n\n")
        f.write(_table(pareto, ["scenario", "policy", "seed", "waf", "wear_std", "gc_count", "pareto_metrics"]))
        f.write("\n## Next Experiment Ideas\n\n")
        f.write(_table(recommendations, ["priority", "scenario", "policy", "recommendation", "reason"], limit=10))
        f.write("\n## Caution\n\n")
        f.write("- These are simulator-relative signals, not production SSD performance claims.\n")
        f.write("- A narrative should distinguish observed results from hypotheses about firmware behavior.\n")
        f.write("- TRIM rows should distinguish mapping invalidation from later GC reclamation.\n")
        f.write("- Any LLM-generated explanation should cite the CSV rows used as evidence.\n")


def write_llm_prompt(path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Prompt For LLM Review\n\n")
        f.write("You are reviewing an SSD garbage-collection simulator validation packet.\n")
        f.write("Use only the evidence in the attached CSV/Markdown files. Do not invent measured results.\n\n")
        f.write("Tasks:\n")
        f.write("1. Summarize the strongest policy trade-offs in plain engineering language.\n")
        f.write("2. Separate observations from hypotheses.\n")
        f.write("3. Identify which runs need more seeds or GC event traces.\n")
        f.write("4. Suggest the next validation sweep with a concrete reason.\n")
        f.write("5. Keep the conclusion honest about simulator scope and real SSD limitations.\n\n")
        f.write("Preferred output:\n")
        f.write("- Executive summary\n")
        f.write("- Evidence-backed findings\n")
        f.write("- Open risks and limitations\n")
        f.write("- Recommended next experiments\n")


def write_context_index(path: str, insight_dir: str, packet: Dict[str, List[Dict[str, Any]]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# LLM Context Index\n\n")
        f.write(f"- Insight directory: `{insight_dir}`\n")
        for name, rows in packet.items():
            f.write(f"- {name}: {len(rows)} rows\n")
        f.write("\nFiles to attach or paste into an LLM session:\n\n")
        f.write("- `policy_scorecard.csv`\n")
        f.write("- `anomaly_runs.csv`\n")
        f.write("- `pareto_front.csv`\n")
        f.write("- `recommended_next_sweeps.csv`\n")
        f.write("- `narrative_report.md`\n")
        f.write("- `prompt_for_llm.md`\n")


def run_narrator(insight_dir: str, out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    packet = _load_packet(insight_dir)
    outputs = {
        "narrative_report": os.path.join(out_dir, "narrative_report.md"),
        "prompt_for_llm": os.path.join(out_dir, "prompt_for_llm.md"),
        "context_index": os.path.join(out_dir, "llm_context_index.md"),
    }
    write_narrative(outputs["narrative_report"], packet)
    write_llm_prompt(outputs["prompt_for_llm"])
    write_context_index(outputs["context_index"], insight_dir, packet)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an LLM-ready narrative packet from insight-miner outputs.")
    parser.add_argument("--insight_dir", default="results/insights")
    parser.add_argument("--out_dir", default="results/llm_narrative")
    args = parser.parse_args()

    outputs = run_narrator(args.insight_dir, args.out_dir)
    for label, path in outputs.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
