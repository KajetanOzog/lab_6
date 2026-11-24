from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Polygon


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "images" / "depth_maps"
BG = "#F8FAFC"
INK = "#0F172A"
BLUE = "#2563EB"
TEAL = "#0F766E"
RED = "#DC2626"
SECONDARY = "#64748B"
TAU = 0.4
SIDE = 3.0
PIXELS = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
POSITIONS = np.column_stack((PIXELS[:, 0], 1 - PIXELS[:, 1])) * SIDE
NAMES = ("00", "10", "01", "11")
FACES = np.array([[0, 2, 1], [1, 2, 3]])
EDGES = ((0, 1), (0, 2), (1, 2), (1, 3), (2, 3))
DEPTHS = (np.array([1.0, 1.1, 1.0, 1.2]), np.array([1.0, 1.1, 1.0, 3.8]))


def number(value):
    return f"{value:.1f}".replace(".", "{,}")


def pale(color, amount):
    return tuple((1 - amount) * np.array(to_rgb(BG)) + amount * np.array(to_rgb(color)))


def back_project(depth):
    fx = fy = 4.0
    cx = cy = 0.5
    points = np.column_stack((
        (PIXELS[:, 0] - cx) * depth / fx,
        (PIXELS[:, 1] - cy) * depth / fy,
        depth,
    ))
    projected = np.column_stack((
        fx * points[:, 0] / points[:, 2] + cx,
        fy * points[:, 1] / points[:, 2] + cy,
    ))
    assert np.allclose(projected, PIXELS)
    return points


def classify(depth):
    points = back_project(depth)
    triangle_depths = points[FACES, 2]
    differences = np.abs(triangle_depths - np.roll(triangle_depths, -1, axis=1))
    maxima = np.max(differences, axis=1)
    assert np.allclose(maxima, np.ptp(triangle_depths, axis=1))
    return maxima <= TAU, maxima


def draw_panel(ax, depth, panel_id):
    accepted, maxima = classify(depth)
    labels = []
    text_positions = ((1.0, 2.18), (2.0, 0.82))
    for i, (face, keep, position) in enumerate(zip(FACES, accepted, text_positions)):
        color = TEAL if keep else RED
        amount = 0.12 if i == 0 else 0.19
        ax.add_patch(Polygon(POSITIONS[face], facecolor=pale(color, amount),
                             edgecolor="none", zorder=1))
        status = "zaakceptowany" if keep else "odrzucony"
        labels.append(ax.text(
            *position, "$T_" + str(i + 1) + "$\n" + status,
            fontsize=16, color=color, ha="center", va="center", linespacing=1.5, zorder=5,
        ))
    for a, b in EDGES:
        color = RED if abs(depth[a] - depth[b]) > TAU else TEAL
        ax.plot(POSITIONS[[a, b], 0], POSITIONS[[a, b], 1],
                color=color, linewidth=2.2 if color == TEAL else 3.0,
                solid_capstyle="round", zorder=2)
    outlier = np.abs(depth - np.median(depth)) > TAU
    ax.scatter(POSITIONS[:, 0], POSITIONS[:, 1], s=115,
               c=np.where(outlier, RED, BLUE), edgecolors=BG, linewidths=1.4, zorder=6)
    offsets = ((-0.18, 0.10), (0.18, 0.10), (-0.18, -0.10), (0.18, -0.10))
    for i, (point, name, offset) in enumerate(zip(POSITIONS, NAMES, offsets)):
        right = PIXELS[i, 0] == 1
        top = PIXELS[i, 1] == 0
        text = "$P_{" + name + "}$\n$z=" + number(depth[i]) + "$"
        labels.append(ax.text(
            *(point + offset), text, fontsize=18, color=RED if outlier[i] else INK,
            ha="left" if right else "right", va="bottom" if top else "top",
            linespacing=1.45, zorder=7,
        ))
    if panel_id == 1:
        labels.extend((
            ax.text(
                3.72, 1.50,
                r"$|z_{10}-z_{11}|$" + "\n$=" + number(abs(depth[1] - depth[3])) + "$",
                fontsize=17, color=RED, ha="center", va="center", linespacing=1.5,
            ),
            ax.text(
                1.50, -0.40,
                r"$|z_{01}-z_{11}|=" + number(abs(depth[2] - depth[3])) + "$",
                fontsize=17, color=RED, ha="center", va="center",
            ),
            ax.text(
                0.70, 0.70, r"$|\Delta z|=" + number(abs(depth[1] - depth[2])) + "$",
                fontsize=13, color=TEAL, ha="center", va="center", rotation=45,
                rotation_mode="anchor",
                bbox={"facecolor": BG, "edgecolor": "none", "pad": 1.2}, zorder=7,
            ),
            ax.text(
                1.65, -1.04, "Duży skok głębi: możliwe różne powierzchnie.",
                fontsize=14, color=SECONDARY, ha="center", va="center",
            ),
        ))
    else:
        labels.append(ax.text(
            1.50, -1.04, "Oba trójkąty zaakceptowane",
            fontsize=15, color=TEAL, ha="center", va="center",
        ))
    ax.set(xlim=(-1.20, 4.60), ylim=(-1.25, 3.85), aspect="equal")
    ax.axis("off")
    print(f"Panel {panel_id + 1}: z={depth.tolist()}, max edge differences={maxima.tolist()}, accepted={accepted.tolist()}", flush=True)
    return labels


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "text.color": INK,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "svg.fonttype": "path",
        "svg.hashsalt": "lab6-depth-triangle-filter",
    })
    assert np.array_equal(classify(DEPTHS[0])[0], [True, True])
    assert np.array_equal(classify(DEPTHS[1])[0], [True, False])
    assert np.allclose(classify(DEPTHS[0])[1], [0.1, 0.2])
    assert np.allclose(classify(DEPTHS[1])[1], [0.1, 2.8])
    fig = plt.figure(figsize=(14.4, 9.4))
    labels = []
    for panel_id, (left, depth, title) in enumerate(zip(
        (0.045, 0.535), DEPTHS, ("Poprawna triangulacja", "Odrzucanie trójkąta"),
    )):
        ax = fig.add_axes([left, 0.27, 0.42, 0.56])
        labels.extend(draw_panel(ax, depth, panel_id))
        labels.append(fig.text(
            left + 0.21, 0.945, title, fontsize=24, ha="center", va="center",
        ))
    labels.extend((
        fig.text(
            0.5, 0.877, "Układ 2×2 pikseli — schemat połączeń odpowiadających im punktów 3D",
            fontsize=15, color=SECONDARY, ha="center", va="center",
        ),
        fig.text(
            0.5, 0.187,
            r"$\max\left(|z_1-z_2|,\ |z_2-z_3|,\ |z_3-z_1|\right)\leq\tau"
            r"\qquad \tau=" + number(TAU) + "$",
            fontsize=25, ha="center", va="center",
        ),
        fig.text(
            0.5, 0.075,
            "Jeśli różnica głębokości pomiędzy wierzchołkami trójkąta jest zbyt duża,\n"
            "trójkąt odrzucamy.",
            fontsize=17, ha="center", va="center", linespacing=1.6,
        ),
    ))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [label.get_window_extent(renderer) for label in labels]
    for i, box in enumerate(boxes):
        assert fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1)
        assert all(not box.overlaps(other) for other in boxes[i + 1:])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "depth_triangle_filter.svg"
    fig.savefig(path, facecolor=BG, metadata={
        "Creator": "lab_6 / NumPy / Matplotlib",
        "Description": (
            "The squares depict 2x2 pixel adjacency, not a metric planar 3D quadrilateral. "
            "Pij are 3D back-projected points. The same diagonal P10-P01 is used in both cases. "
            "Triangle acceptance is computed from all three vertex-depth edge differences. "
            "Left depths: 1.0,1.1,1.0,1.2. Right depths: 1.0,1.1,1.0,3.8. Threshold 0.4. "
            "First panel accepts both triangles; second retains T1 and rejects T2."
        ),
    })
    plt.close(fig)
    print(path.relative_to(ROOT), flush=True)
    print("Real back-projection, per-edge depth differences, triangle decisions and layout: OK")


if __name__ == "__main__":
    main()
