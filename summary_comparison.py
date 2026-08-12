"""Build the final-state comparison figure (summary_comparison.png) across all
five growth models, from the *_ODE_data.json files produced by the individual
simulation scripts.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Model label -> (json file, line color, linestyle, marker)
# Color alone doesn't survive grayscale printing, so each model also gets a
# distinct linestyle and marker (reviewer request).
MODELS = {
    "KDAB": ("KFR_ODE_data.json", "C0", "-", "o"),
    "KOM": ("KOM_ODE_data.json", "C1", "--", "s"),
    "LT": ("LT2_ODE_data.json", "C2", "-.", "^"),
    "GAPK (stress)": ("GCG_ODE_data.json", "C3", ":", "D"),
    "GAPK (strain)": ("GEG_ODE_data.json", "C4", (0, (3, 1, 1, 1)), "v"),
}

PANELS = [
    ("Radial Stress (Cauchy)", "Stress", "radial_stress", (0, 0)),
    ("Radial Stretch", "Stretch", "radial_strain", (0, 1)),
    ("Radial Growth", "Growth", "radial_growth", (0, 2)),
    ("Hoop Stress (Cauchy)", "Stress", "hoop_stress", (1, 0)),
    ("Hoop Stretch", "Stretch", "hoop_strain", (1, 1)),
    ("Hoop Growth", "Growth", "hoop_growth", (1, 2)),
]


def plot_results():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 22,
            "axes.titlesize": 30,
            "axes.labelsize": 28,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "legend.fontsize": 24,
            "legend.title_fontsize": 24,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.fancybox": True,
            "lines.linewidth": 3,
            "grid.alpha": 0.4,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )

    final_states = {}
    for label, (file_name, color, linestyle, marker) in MODELS.items():
        data = json.loads(Path(file_name).read_text())
        R_range = np.array(data["R_range"])
        plot_data = data["plot_data_1d"]
        final_states[label] = {
            "R_range": R_range,
            "color": color,
            "linestyle": linestyle,
            "marker": marker,
            **{key: np.array(plot_data[key][-1]) for _, _, key, _ in PANELS},
        }

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Comparison of Final Simulated States", fontsize=40, fontweight="bold")

    for panel_index, (title, ylabel, key, (row, col)) in enumerate(PANELS):
        ax = axs[row, col]
        for label, state in final_states.items():
            ax.plot(
                state["R_range"],
                state[key],
                color=state["color"],
                linestyle=state["linestyle"],
                marker=state["marker"],
                markevery=8,
                markersize=9,
                label=label,
            )
        ax.set_title(f"({chr(ord('a') + panel_index)}) {title}")
        ax.set_xlabel("$R$")
        ax.set_ylabel(ylabel)
        ax.grid(True)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=len(MODELS),
        framealpha=0.9,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig("summary_comparison.png")
    plt.close(fig)
    print("Saved: summary_comparison.png")


if __name__ == "__main__":
    plot_results()
