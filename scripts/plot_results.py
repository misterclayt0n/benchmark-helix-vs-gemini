"""
Generate a simple SVG plot from the benchmark CSV outputs.

The script reads results/helix_results.csv and results/gemini_results.csv,
computes the median latency per concurrency level for each system, and
produces results/latency_plot.svg without requiring external plotting libs.
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
OUTPUT_SVG = RESULTS_DIR / "latency_plot.svg"


def load_results(path: Path, system_name: str) -> Dict[int, List[float]]:
    data: Dict[int, List[float]] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            concurrency = int(row["concurrency"])
            latency = float(row["latency_ms"])
            data.setdefault(concurrency, []).append(latency)
    if not data:
        raise RuntimeError(f"No rows found for {system_name} in {path}")
    return data


def system_summary(system: str, data: Dict[int, List[float]]) -> List[Tuple[int, float]]:
    summary = []
    for concurrency in sorted(data):
        median_latency = statistics.median(data[concurrency])
        summary.append((concurrency, median_latency))
    return summary


def render_svg(series: Dict[str, List[Tuple[int, float]]]) -> str:
    width, height = 800, 500
    margin = 80
    all_x = sorted({x for points in series.values() for x, _ in points})
    all_y = [y for points in series.values() for _, y in points]
    if not all_y:
        raise RuntimeError("No latency values to plot.")
    max_y = max(all_y) * 1.1

    def x_to_px(value: int) -> float:
        idx = all_x.index(value)
        span = len(all_x) - 1 or 1
        return margin + (idx / span) * (width - 2 * margin)

    def y_to_px(value: float) -> float:
        return height - margin - (value / max_y) * (height - 2 * margin)

    colors = {
        "helix": "#2c7fb8",
        "gemini": "#d95f0e",
    }

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<style>text { font-family: "Helvetica", "Arial", sans-serif; font-size: 14px; }</style>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        # axes
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" '
        f'stroke="#000" stroke-width="1.5"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" '
        f'stroke="#000" stroke-width="1.5"/>',
        f'<text x="{width/2}" y="{height - 20}" text-anchor="middle">Concurrency</text>',
        f'<text x="20" y="{height/2}" transform="rotate(-90 20,{height/2})" text-anchor="middle">'
        "Median latency (ms)</text>",
    ]

    # axis ticks
    for idx, x_val in enumerate(all_x):
        px = x_to_px(x_val)
        svg_lines.append(
            f'<line x1="{px}" y1="{height - margin}" x2="{px}" y2="{height - margin + 5}" stroke="#000"/>'
        )
        svg_lines.append(
            f'<text x="{px}" y="{height - margin + 25}" text-anchor="middle">{x_val}</text>'
        )

    for frac in [0.25, 0.5, 0.75, 1.0]:
        y_val = max_y * frac
        py = y_to_px(y_val)
        svg_lines.append(
            f'<line x1="{margin - 5}" y1="{py}" x2="{width - margin}" y2="{py}" '
            f'stroke="#ccc" stroke-dasharray="4 4"/>'
        )
        svg_lines.append(
            f'<text x="{margin - 10}" y="{py + 5}" text-anchor="end">{int(y_val)}</text>'
        )

    # plot lines
    for system, points in series.items():
        color = colors.get(system, "#333333")
        path_cmds = []
        for idx, (x_val, y_val) in enumerate(points):
            px, py = x_to_px(x_val), y_to_px(y_val)
            path_cmds.append(("M" if idx == 0 else "L") + f"{px},{py}")
            svg_lines.append(
                f'<circle cx="{px}" cy="{py}" r="4" fill="{color}" stroke="#ffffff" stroke-width="1"/>'
            )
            svg_lines.append(
                f'<text x="{px}" y="{py - 10}" text-anchor="middle" fill="{color}">{int(y_val)}</text>'
            )
        svg_lines.append(
            f'<path d="{" ".join(path_cmds)}" fill="none" stroke="{color}" stroke-width="2.5"/>'
        )

    # legend
    legend_x = width - margin - 160
    legend_y = margin + 10
    svg_lines.append(f'<rect x="{legend_x - 10}" y="{legend_y - 20}" width="170" height="70" fill="#fff" stroke="#ccc"/>')
    for idx, (system, color) in enumerate(colors.items()):
        svg_lines.append(
            f'<line x1="{legend_x}" y1="{legend_y + idx * 24}" '
            f'x2="{legend_x + 30}" y2="{legend_y + idx * 24}" stroke="{color}" stroke-width="3"/>'
        )
        svg_lines.append(
            f'<text x="{legend_x + 40}" y="{legend_y + idx * 24 + 5}" fill="#333">{system.title()}</text>'
        )

    svg_lines.append("</svg>")
    return "\n".join(svg_lines)


def main() -> None:
    helix_data = load_results(RESULTS_DIR / "helix_results.csv", "helix")
    gemini_data = load_results(RESULTS_DIR / "gemini_results.csv", "gemini")

    helix_summary = system_summary("helix", helix_data)
    gemini_summary = system_summary("gemini", gemini_data)

    svg = render_svg({"helix": helix_summary, "gemini": gemini_summary})
    OUTPUT_SVG.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_SVG.write_text(svg, encoding="utf-8")
    print(f"Saved plot to {OUTPUT_SVG}")


if __name__ == "__main__":
    main()

