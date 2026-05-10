"""
Animate hard-macro placement snapshots (like the SA GIFs in README assets).

Expects a `.pt` saved with env `MACRO_PLACER_SAVE_HISTORY=1` while running
`submissions/mobo_surrogate/placer.py` (writes `vis/placer_history_<bench>.pt`).

Usage:
    MACRO_PLACER_SAVE_HISTORY=1 uv run evaluate submissions/mobo_surrogate/placer.py -b ibm01
    uv run placement-gif vis/placer_history_ibm01.pt -b ibm01 -o vis/progress_ibm01.gif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

from macro_place.loader import load_benchmark_from_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description="GIF from placer history tensors.")
    ap.add_argument("history_pt", type=str, help="Path to placer_history_<bench>.pt")
    ap.add_argument(
        "-b",
        "--benchmark",
        type=str,
        required=True,
        help="ICCAD04 name (e.g. ibm01) for canvas + macro sizes.",
    )
    ap.add_argument("-o", "--out", type=str, default=None, help="Output .gif path")
    ap.add_argument("--fps", type=float, default=12.0)
    args = ap.parse_args(argv)

    root = Path("external/MacroPlacement/Testcases/ICCAD04") / args.benchmark
    if not root.exists():
        print(f"Benchmark dir not found: {root}", file=sys.stderr)
        sys.exit(1)

    benchmark, _ = load_benchmark_from_dir(str(root))
    data = torch.load(args.history_pt, map_location="cpu", weights_only=False)
    history = data["history"]
    n_hard = int(data.get("n_hard", benchmark.num_hard_macros))

    if not history:
        print("History is empty.", file=sys.stderr)
        sys.exit(1)

    out = Path(args.out or Path(args.history_pt).with_suffix(".gif"))
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, benchmark.canvas_width)
    ax.set_ylim(0, benchmark.canvas_height)
    ax.set_aspect("equal")
    ax.set_title(f"{args.benchmark} — placement search")
    ax.set_xlabel("X (μm)")
    ax.set_ylabel("Y (μm)")
    artists = []

    def draw_frame(pos_np):
        for p in artists:
            p.remove()
        artists.clear()
        ax.add_patch(
            patches.Rectangle(
                (0, 0),
                benchmark.canvas_width,
                benchmark.canvas_height,
                fill=False,
                edgecolor="black",
                linewidth=1.2,
            )
        )
        for i in range(n_hard):
            x, y = float(pos_np[i, 0]), float(pos_np[i, 1])
            w, h = float(benchmark.macro_sizes[i, 0]), float(benchmark.macro_sizes[i, 1])
            color = "darkred" if bool(benchmark.macro_fixed[i]) else "steelblue"
            r = patches.Rectangle(
                (x - w / 2, y - h / 2),
                w,
                h,
                fill=True,
                facecolor=color,
                edgecolor="black",
                linewidth=0.35,
                alpha=0.55,
            )
            ax.add_patch(r)
            artists.append(r)

    # First frame manual
    draw_frame(history[0])
    plt.tight_layout()

    import matplotlib.animation as animation

    frames = [h if isinstance(h, torch.Tensor) else torch.tensor(h) for h in history]

    def animate(k):
        pos = frames[k].numpy()
        draw_frame(pos)
        return artists

    interval_ms = max(40.0, 1000.0 / max(args.fps, 1e-6))
    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=len(frames),
        interval=interval_ms,
        blit=False,
        repeat=True,
    )
    anim.save(str(out), writer="pillow", fps=min(args.fps, 24.0))

    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
