from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "images" / "sdf"
BG = "#F8FAFC"
INK = "#0F172A"
BLUE = "#2563EB"
ORANGE = "#EA580C"
SECONDARY = "#64748B"
GRID = "#CBD5E1"
CMAP = LinearSegmentedColormap.from_list("sphere_sdf", [BLUE, BG, ORANGE], N=257)


def sdf(x, y, z):
    return np.sqrt(np.asarray(x) ** 2 + np.asarray(y) ** 2 + np.asarray(z) ** 2) - 1


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "font.size": 16,
        "text.color": INK,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "axes.edgecolor": SECONDARY,
        "axes.labelcolor": INK,
        "xtick.color": SECONDARY,
        "ytick.color": SECONDARY,
        "svg.fonttype": "path",
    })
    coords = np.linspace(-1.7, 1.7, 501)
    x, y = np.meshgrid(coords, coords)
    field = sdf(x, y, 0)
    points = np.array([[-0.4, 0, 0], [0, 1, 0], [1.5, 0, 0]])
    values = sdf(*points.T)
    assert np.allclose(values, [-0.6, 0, 0.5], atol=1e-12)
    assert np.isclose(field.min(), -1)
    angles = np.linspace(0, 2 * np.pi, 361)
    assert np.allclose(sdf(np.cos(angles), np.sin(angles), 0), 0, atol=1e-12)

    fig = plt.figure(figsize=(6.8, 7.0))
    ax = fig.add_axes([0.115, 0.245, 0.82, 0.71])
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1.5)
    levels = np.r_[np.linspace(-1, 0, 33), np.linspace(0, 1.5, 49)[1:]]
    ax.contourf(x, y, field, levels=levels, cmap=CMAP, norm=norm, antialiased=False)
    ax.contour(x, y, field, levels=[-0.5, 0.5, 1], colors=SECONDARY,
               linewidths=0.65, alpha=0.55, linestyles="solid")
    ax.axhline(0, color=GRID, lw=0.7, alpha=0.8)
    ax.axvline(0, color=GRID, lw=0.7, alpha=0.8)
    ax.contour(x, y, field, levels=[0], colors=INK, linewidths=2.5)
    labels = [
        (points[0], (-0.20, -0.43), r"$f(\mathbf{p})<0$", BLUE),
        (points[1], (0.08, 1.38), r"$f(\mathbf{p})=0$", INK),
        (points[2], (1.02, -1.27), r"$f(\mathbf{p})>0$", ORANGE),
    ]
    annotations = []
    for point, location, label, color in labels:
        ax.plot(point[0], point[1], "o", ms=10, mfc=color, mec=BG, mew=1.8, zorder=6)
        annotation = ax.annotate(
            label, xy=point[:2], xytext=location, ha="center", va="center",
            fontsize=20, color=color, zorder=7,
            bbox=dict(facecolor=BG, edgecolor="none", alpha=0.95, pad=4),
            arrowprops=dict(arrowstyle="-", color=color, lw=1.1, shrinkA=6, shrinkB=9),
        )
        annotations.append(annotation)
    ax.text(-1.55, 1.50, "$z=0$", fontsize=14, color=SECONDARY, ha="left", va="center")
    ax.set(xlim=(-1.7, 1.7), ylim=(-1.7, 1.7), aspect="equal",
           xlabel="$x$", ylabel="$y$", xticks=[-1, 0, 1], yticks=[-1, 0, 1])
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=13, length=3)
    ax.set_xlabel("$x$", fontsize=18, labelpad=2)
    ax.set_ylabel("$y$", fontsize=18, rotation=0, labelpad=10)

    cax = fig.add_axes([0.20, 0.105, 0.65, 0.026])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax,
                       orientation="horizontal", ticks=[-1, -0.5, 0, 0.5, 1, 1.5])
    bar.set_ticklabels(["−1", "−0.5", "0", "+0.5", "+1", "+1.5"])
    bar.ax.tick_params(labelsize=12, length=3, color=SECONDARY, labelcolor=SECONDARY)
    bar.outline.set_visible(False)
    bar.solids.set_rasterized(False)
    cax.axvline(0, color=INK, lw=1.8)
    fig.text(0.135, 0.118, "$f$", fontsize=19, ha="center", va="center")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [item.get_bbox_patch().get_window_extent(renderer) for item in annotations]
    for i, box in enumerate(boxes):
        assert ax.bbox.contains(box.x0, box.y0) and ax.bbox.contains(box.x1, box.y1)
        assert all(not box.overlaps(other) for other in boxes[i + 1:])

    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "sphere_sdf_slice.svg"
    fig.savefig(path, facecolor=BG,
                metadata={"Creator": "lab_6 / Python / NumPy / Matplotlib"})
    print(path.relative_to(ROOT))
    plt.close(fig)
    for point, value in zip(points, values):
        print(f"p={tuple(point)}: f(p)={value:.6f}")
    print("SDF values and label layout: OK")


if __name__ == "__main__":
    main()
