"""Reproduce publication figures from checked-in, verified game recordings."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from qgpu.pixelcore.runtime import verify


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "docs/results")
    parser.add_argument("--out", type=Path, default=root / "docs/media")
    args = parser.parse_args()
    traces = [json.loads((args.data / f).read_text()) for f in ("full-qwen.json", "standalone.json")]
    for trace in traces:
        verify(trace)
    full, small = traces
    assert [r["generated"] for r in full["records"]] == [r["generated"] for r in small["records"]]
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white"})
    times = [[r["seconds"] for r in t["records"][1:]] for t in traces]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [2.6, 1]})
    labels = ["Full Qwen3.5-2B + PixelCore", "Standalone compiled circuit (no language backbone)"]
    colors = ["#30679b", "#b75e30"]
    for values, label, color in zip(times, labels, colors):
        axes[0].plot(range(1, len(values)+1), values, "o-", ms=3.5, lw=1.3, color=color, label=label)
    axes[0].set(xlabel="Game tick", ylabel="Seconds per verified tick",
                title="Same 29-tick game; separate inference modes", ylim=(0, max(times[0])*1.15))
    axes[0].grid(axis="y", alpha=.2)
    axes[0].legend(loc="upper right", fontsize=8)
    means = [np.mean(v) for v in times]
    bars = axes[1].bar(["Full Qwen", "Standalone"], means, color=colors, width=.55)
    axes[1].set(ylabel="Mean seconds / tick", title="Observed mean", ylim=(0, max(means)*1.25))
    for bar, value in zip(bars, means):
        axes[1].text(bar.get_x()+bar.get_width()/2, value+.045, f"{value:.3f} s", ha="center", weight="bold")
    fig.suptitle("PixelCore: measured inference latency on a 24 GB Apple Silicon Mac", fontsize=13, weight="bold")
    fig.text(.01, .015, "bf16 / MPS · one recorded game per mode · initial frame excluded · external button controller", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0,.06,1,.94))
    fig.savefig(args.out/"latency.png", dpi=180)
    fig.savefig(args.out/"latency.svg")
    svg = args.out/"latency.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")
    plt.close(fig)

    cmap = ListedColormap(["#0c140f", "#4b5d43", "#ffb47b", "#c5f888"])
    chosen = [0, 4, 9, 19, 29]
    fig, axes = plt.subplots(1, len(chosen), figsize=(11, 2.9))
    for ax, index in zip(axes, chosen):
        record = full["records"][index]
        ax.imshow(np.array(record["pixels"]).reshape(8,8), cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        ax.set_title(f"Tick {index}\nSaves: {(record['after'] >> 11) & 3}/3", fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle("Actual full-Qwen pixel packets, decoded into frames", fontsize=13, weight="bold")
    fig.legend(handles=[Patch(color=cmap(i),label=name) for i,name in
                        [(1,"Wall"),(2,"Ball"),(3,"Paddle")]], loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0,.14,1,.87))
    fig.savefig(args.out/"game-frames.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,2.6))
    left = 0
    for amount, label, color in [(1,"State\n1 token","#e6a15b"),(16,"16 pixel packets\n4 pixels per packet","#84ac67"),(1,"End\n1 token","#7e9abb")]:
        ax.barh([0], [amount], left=left, color=color, height=.45)
        ax.text(left+amount/2,0,label,ha="center",va="center",fontsize=9)
        left += amount
    ax.set(xlim=(0,18), xticks=range(19), yticks=[], xlabel="Generated token position")
    ax.set_title("One regular game tick = 18 model-generated tokens", weight="bold",pad=15)
    ax.spines[["left","bottom"]].set_visible(False)
    fig.text(.5,.015,"The initial frame emits 17 tokens: its state word is host-supplied.",ha="center",fontsize=9,color="#555555")
    fig.tight_layout(rect=(0,.09,1,1))
    fig.savefig(args.out/"token-budget.png",dpi=180)
    plt.close(fig)
    summary={"verified": True, "ticks": len(times[0]),"modes":[
        {"mode":name,"mean_seconds":float(np.mean(v)),"median_seconds":float(np.median(v)),
         "min_seconds":min(v),"max_seconds":max(v)} for name,v in zip(["full-qwen","standalone"],times)]}
    (args.out/"plot-data.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
