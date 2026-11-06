#!/usr/bin/env python3
"""Draw dimensioned central-gap designs from the same polygons used by the study."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import yaml
from electrode_central_gap_geometry import make_central_gap_case, validate_central_gap_spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-config", type=Path,
        default=ROOT / "config/electrode_central_gap_study_12x25mm.yaml")
    parser.add_argument("--out", type=Path,
        default=ROOT / "docs/figures/central_gap_12x25_comparison.png")
    args = parser.parse_args()
    spec = validate_central_gap_spec(yaml.safe_load(args.study_config.read_text()))
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "ecsp-central-gap-study"
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle, Patch

    colors = {"anode": "#cf4545", "cathode": "#356ac3"}
    paper, purple = "#faf8f0", "#784bb4"
    rows = spec["cases"]
    fig, axes = plt.subplots(1, len(rows), figsize=(3.65 * len(rows), 8.2),
                             squeeze=False, sharex=True, sharey=True)
    fig.subplots_adjust(left=.055, right=.985, top=.81, bottom=.17, wspace=.2)
    fig.suptitle("Fixed end gap: central-spacing comparison", fontsize=20, y=.965)
    fig.text(.5, .916, "Propellant: 12 mm wide × 25 mm high   |   Horizontal electrode width: 2 mm   |   End gap: 2 mm",
             ha="center", fontsize=11.5)
    for ax, case in zip(axes[0], rows):
        _, _, meta = make_central_gap_case(case, spec["grid_size"])
        gap = meta["central_gap_mm"]
        ax.add_patch(Rectangle((0, 0), 12, 25, facecolor=paper, edgecolor="#333333", lw=1.4))
        for polarity in ("anode", "cathode"):
            ax.add_patch(Polygon(meta[f"{polarity}_polygon_mm"], closed=True,
                facecolor=colors[polarity], edgecolor=colors[polarity], alpha=.72, lw=1.7))
        ax.axhline(12.5, color="#bbbbbb", ls=":", lw=.75, zorder=0)
        ax.annotate("", xy=(6 + gap / 2, 12.5), xytext=(6 - gap / 2, 12.5),
                    arrowprops={"arrowstyle": "<->", "color": purple, "lw": 1.5})
        ax.text(6, 13.15, f"{gap:g} mm", ha="center", color=purple, weight="bold", fontsize=12,
                bbox={"facecolor": paper, "edgecolor": "none", "pad": 1})
        for x in (5, 7):
            ax.plot([x, x], [25, 26.4], color=purple, lw=.7, ls=":")
        ax.annotate("", xy=(7, 26), xytext=(5, 26),
                    arrowprops={"arrowstyle": "<->", "color": purple, "lw": 1.2})
        ax.text(6, 26.6, "2 mm", ha="center", color=purple, fontsize=10)
        # The bottom-end gap is independently labelled to show both ends fixed.
        ax.annotate("", xy=(7, -.65), xytext=(5, -.65),
                    arrowprops={"arrowstyle": "<->", "color": purple, "lw": 1.2})
        ax.text(6, -1.65, "2 mm", ha="center", color=purple, fontsize=10)
        ax.set_title(f"Central gap = {gap:g} mm\nθ = {meta['inclination_deg']:.3f}° from vertical",
                     fontsize=12, pad=16)
        ax.set_xlim(-.4, 12.4)
        ax.set_ylim(-2.1, 28)
        ax.set_aspect("equal")
        ax.set_xticks([0, 3, 6, 9, 12])
        ax.set_yticks([0, 12.5, 25])
        ax.set_xlabel("x (mm)", fontsize=10)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=3, labelsize=9)
    axes[0][0].set_ylabel("y (mm)", fontsize=10)
    fig.legend(handles=[Patch(facecolor=colors["anode"], alpha=.72, label="Anode (+)"),
                        Patch(facecolor=colors["cathode"], alpha=.72, label="Cathode (−)"),
                        Patch(facecolor=paper, edgecolor="#333333", label="Propellant (also beneath electrodes)")],
               loc="lower center", bbox_to_anchor=(.5, .075), ncol=3, frameon=False, fontsize=11)
    spacing = spec["domain_height_mm"] / spec["grid_size"]
    fig.text(.5, .039, f"Each polarity: 50 mm² in every design. Exact design outlines; solver cell spacing: {spacing:g} mm at Ny = {spec['grid_size']}.",
             ha="center", fontsize=10, color="#555555")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor="white")
    svg_path = args.out.with_suffix(".svg")
    fig.savefig(svg_path, facecolor="white", metadata={"Date": None})
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    plt.close(fig)
    print(args.out)


if __name__ == "__main__":
    main()
