from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "images" / "marching_cubes"
BG = "#F8FAFC"
INK = "#0F172A"
BLUE = "#2563EB"
ORANGE = "#EA580C"
TEAL = "#0F766E"
GRAY = "#94A3B8"


def interpolate_zero(start, end, value_start, value_end):
    assert value_start * value_end < 0
    t = -value_start / (value_end - value_start)
    assert 0 < t < 1
    assert np.isclose((1 - t) * value_start + t * value_end, 0, atol=1e-12)
    point = np.asarray(start, dtype=float) + t * (
        np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    )
    return point, t


def bilinear(u, v):
    u, v = np.asarray(u), np.asarray(v)
    return (
        (1 - u) * (1 - v) * (-0.6) + u * (1 - v) * 0.4
        + (1 - u) * v * (-0.2) + u * v * 0.8
    )


def cell_figure(top, bottom, top_fraction, bottom_fraction):
    fig = plt.figure(figsize=(14.4, 6.4))
    ax = fig.add_axes([0.018, 0.075, 0.445, 0.85])
    outline = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    ax.plot(outline[:, 0], outline[:, 1], color=INK, lw=2.6, zorder=1)
    corners = np.array([[0, 1], [1, 1], [0, 0], [1, 0]])
    ax.scatter(corners[:, 0], corners[:, 1], c=[BLUE, ORANGE, BLUE, ORANGE],
               s=290, edgecolors=BG, linewidths=1.6, zorder=5)
    ax.plot([top[0], bottom[0]], [top[1], bottom[1]],
            color=TEAL, lw=5.0, solid_capstyle="round", zorder=3)
    ax.scatter([top[0], bottom[0]], [top[1], bottom[1]], c=INK,
               s=170, edgecolors=BG, linewidths=1.7, zorder=6)
    labels = [
        ax.text(-0.075, 1, "$-0.2$", ha="right", va="center", fontsize=25, color=BLUE),
        ax.text(1.075, 1, "$+0.8$", ha="left", va="center", fontsize=25, color=ORANGE),
        ax.text(-0.075, 0, "$-0.6$", ha="right", va="center", fontsize=25, color=BLUE),
        ax.text(1.075, 0, "$+0.4$", ha="left", va="center", fontsize=25, color=ORANGE),
        ax.text(0, 1.13, "$A$", ha="center", va="center", fontsize=25, color=BLUE),
        ax.text(1, 1.13, "$B$", ha="center", va="center", fontsize=25, color=ORANGE),
        ax.text(0, -0.13, "$C$", ha="center", va="center", fontsize=25, color=BLUE),
        ax.text(1, -0.13, "$D$", ha="center", va="center", fontsize=25, color=ORANGE),
        ax.annotate("$P_1$", xy=top, xytext=(0, 13), textcoords="offset points",
                    ha="center", va="bottom", fontsize=23),
        ax.annotate("$P_2$", xy=bottom, xytext=(0, -13), textcoords="offset points",
                    ha="center", va="top", fontsize=23),
    ]
    ax.set(xlim=(-0.37, 1.37), ylim=(-0.19, 1.23), aspect="equal")
    ax.axis("off")

    fig.add_artist(FancyArrowPatch(
        (0.463, 0.50), (0.516, 0.50), transform=fig.transFigure,
        arrowstyle="-|>", color=GRAY, lw=2.6, mutation_scale=22,
    ))
    box = {"boxstyle": "square,pad=0.40", "facecolor": BG,
           "edgecolor": GRAY, "linewidth": 1.2}
    labels.extend([
        fig.text(0.56, 0.87, r"$t=\dfrac{-f(A)}{f(B)-f(A)}$",
                 ha="left", va="center", fontsize=25, bbox=box),
        fig.text(0.56, 0.69, r"$P=A+t(B-A)$",
                 ha="left", va="center", fontsize=25, bbox=box),
        fig.text(0.56, 0.51, r"$t_1=\dfrac{0.2}{0.8+0.2}="
                 + f"{top_fraction:.1f}" + "$",
                 ha="left", va="center", fontsize=25),
        fig.text(0.56, 0.38, r"$P_1=A+" + f"{top_fraction:.1f}" + r"(B-A)$",
                 ha="left", va="center", fontsize=25, color=TEAL),
        fig.text(0.56, 0.22, r"$t_2=\dfrac{0.6}{0.4+0.6}="
                 + f"{bottom_fraction:.1f}" + "$",
                 ha="left", va="center", fontsize=25),
        fig.text(0.56, 0.09, r"$P_2=C+" + f"{bottom_fraction:.1f}" + r"(D-C)$",
                 ha="left", va="center", fontsize=25, color=TEAL),
    ])

    fig.canvas.draw()
    assert np.isclose(np.linalg.norm(ax.transData.transform((1, 0))
                                    - ax.transData.transform((0, 0))),
                      np.linalg.norm(ax.transData.transform((0, 1))
                                    - ax.transData.transform((0, 0))))
    renderer = fig.canvas.get_renderer()
    boxes = [label.get_window_extent(renderer) for label in labels]
    for i, bounds in enumerate(boxes):
        assert fig.bbox.contains(bounds.x0, bounds.y0)
        assert fig.bbox.contains(bounds.x1, bounds.y1)
        assert all(not bounds.overlaps(other) for other in boxes[i + 1:])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "marching_squares_cell_interpolation.svg"
    fig.savefig(path, facecolor=BG,
                metadata={"Creator": "lab_6 / Python / NumPy / Matplotlib"})
    plt.close(fig)
    print(path.relative_to(ROOT))


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "font.size": 16,
        "text.color": INK,
        "axes.facecolor": BG,
        "figure.facecolor": BG,
        "svg.fonttype": "path",
    })
    top, top_fraction = interpolate_zero([0, 1], [1, 1], -0.2, 0.8)
    bottom, bottom_fraction = interpolate_zero([0, 0], [1, 0], -0.6, 0.4)
    assert np.allclose(top, [0.2, 1])
    assert np.allclose(bottom, [0.6, 0])
    assert np.isclose(top_fraction, 0.2)
    assert np.isclose(bottom_fraction, 0.6)
    assert np.allclose(bilinear([0, 1, 0, 1], [1, 1, 0, 0]), [-0.2, 0.8, -0.6, 0.4])
    t = np.linspace(0, 1, 201)
    segment = top[None, :] * (1 - t[:, None]) + bottom[None, :] * t[:, None]
    assert np.max(np.abs(bilinear(segment[:, 0], segment[:, 1]))) < 1e-12
    cell_figure(top, bottom, top_fraction, bottom_fraction)
    print(f"Interpolation: t1={top_fraction:.1f}, t2={bottom_fraction:.1f}")
    print("Corner values, contour, cell proportions and label layout: OK")


if __name__ == "__main__":
    main()
