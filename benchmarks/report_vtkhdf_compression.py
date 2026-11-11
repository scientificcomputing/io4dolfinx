"""Summarize rank subdirectories produced by vtkhdf_compression.py.

Usage: python benchmarks/report_vtkhdf_compression.py output/vtkhdf-compression
"""

import argparse
import csv
import json
import os
import statistics
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/io4dolfinx-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODES = ["uncompressed", "gzip", "gzip_shuffle"]
LABELS = ["No compression", "Gzip level 4", "Gzip level 4 + shuffle"]
COLORS = ["#697586", "#3178b8", "#c45525"]


def summarize(root: Path):
    reports = [json.loads(path.read_text()) for path in sorted(root.glob("ranks-*/results.json"))]
    if not reports:
        raise ValueError(f"No rank reports in {root}")
    raw = [row for report in reports for row in report["results"]]
    sizes = sorted({row["n"] for row in raw})
    ranks = sorted({row["ranks"] for row in raw})
    rows = []
    for n in sizes:
        for rank in ranks:
            for mode in MODES:
                samples = [
                    r for r in raw if r["n"] == n and r["ranks"] == rank and r["mode"] == mode
                ]
                if not samples:
                    raise ValueError(f"Missing n={n}, ranks={rank}, mode={mode}")
                row = dict(n=n, cells=2 * n * n, ranks=rank, mode=mode, samples=len(samples))
                for metric in (
                    "bytes",
                    "write_seconds",
                    "read_seconds",
                    "sync_seconds",
                    "mesh_write_seconds",
                    "point_write_seconds",
                    "cell_write_seconds",
                    "mesh_read_seconds",
                    "field_read_seconds",
                ):
                    values = sorted(r[metric] for r in samples)
                    row[metric] = statistics.median(values)
                    if metric in ("write_seconds", "read_seconds"):
                        quartiles = statistics.quantiles(values, n=4, method="inclusive")
                        row[f"{metric}_p25"] = quartiles[0]
                        row[f"{metric}_p75"] = quartiles[2]
                        row[f"{metric}_min"] = min(values)
                        row[f"{metric}_max"] = max(values)
                rows.append(row)
    for row in rows:
        baseline = next(
            r
            for r in rows
            if r["n"] == row["n"] and r["ranks"] == row["ranks"] and r["mode"] == "uncompressed"
        )
        row["size_ratio"] = row["bytes"] / baseline["bytes"]
        row["write_time_ratio"] = row["write_seconds"] / baseline["write_seconds"]
        row["read_time_ratio"] = row["read_seconds"] / baseline["read_seconds"]
    with (root / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "all_results.json").write_text(json.dumps(reports, indent=2) + "\n")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(len(sizes), 3, figsize=(15, 3.5 * len(sizes)), squeeze=False)
    metrics = [
        ("write_seconds", "Write time incl. fsync (ms)", 1000),
        ("read_seconds", "Read time, warm cache (ms)", 1000),
        ("bytes", "File size (MB, decimal)", 1e-6),
    ]
    for i, n in enumerate(sizes):
        for j, (metric, label, scale) in enumerate(metrics):
            ax = axes[i, j]
            for mode, legend, color in zip(MODES, LABELS, COLORS):
                selected = [
                    next(
                        r for r in rows if r["n"] == n and r["ranks"] == rank and r["mode"] == mode
                    )
                    for rank in ranks
                ]
                y = [r[metric] * scale for r in selected]
                error = None
                if metric != "bytes":
                    error = [
                        [(r[metric] - r[f"{metric}_p25"]) * scale for r in selected],
                        [(r[f"{metric}_p75"] - r[metric]) * scale for r in selected],
                    ]
                ax.errorbar(ranks, y, yerr=error, marker="o", capsize=3, color=color, label=legend)
            ax.set_xscale("log", base=2)
            ax.set_xticks(ranks, [str(rank) for rank in ranks])
            ax.set_xlabel("MPI ranks")
            ax.set_ylabel(label)
            ax.set_title(f"{2 * n * n:,} triangles ({n} × {n} mesh)")
            ax.grid(True, alpha=0.2)
            ax.set_ylim(bottom=0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("VTKHDF: compression, shuffle, and MPI scaling", fontsize=17, y=0.995)
    fig.text(
        0.5,
        0.005,
        "Median of 5 measured runs; bars show the middle 50%. One warm-up per mode. "
        "Smooth scalar/vector fields; automatic chunks; ext4 storage.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.94))
    fig.savefig(root / "comparison.png", dpi=160)
    fig.savefig(root / "comparison.svg")
    plt.close(fig)
    for row in rows:
        print(
            f"n={row['n']:3} ranks={row['ranks']} {row['mode']:14} "
            f"{row['bytes'] / 1e6:7.3f} MB "
            f"write={row['write_seconds'] * 1000:8.2f} ms "
            f"read={row['read_seconds'] * 1000:8.2f} ms "
            f"write/base={row['write_time_ratio']:.2f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    summarize(parser.parse_args().root)
