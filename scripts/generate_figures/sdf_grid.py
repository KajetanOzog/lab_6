from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, ConnectionPatch, FancyArrowPatch, Polygon


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "images" / "sdf"
BG = "#F8FAFC"
INK = "#0F172A"
BLUE = "#2563EB"
TEAL = "#0F766E"
ORANGE = "#EA580C"
GRAY = "#94A3B8"
GRID = "#CBD5E1"
EDGES = np.array([
    [0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6],
    [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7],
])


def sdf(points):
    return np.linalg.norm(np.asarray(points), axis=-1) - 1


def basis():
    az, el = np.deg2rad([-55, 26])
    view = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = np.array([-np.sin(az), np.cos(az), 0])
    return np.stack([right, np.cross(view, right), view])


def cube(origin, h):
    offsets = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ])
    return np.asarray(origin) + h * offsets


def sign_colors(values):
    return np.where(np.abs(values) < 1e-12, INK, np.where(values < 0, BLUE, ORANGE))


def draw_cube(ax, projected, linewidth=1.8):
    ax.add_patch(Polygon(projected[[0, 1, 5, 4], :2], fc=BLUE, ec="none", alpha=0.035, zorder=6))
    for edge in EDGES:
        hidden = 3 in edge
        ax.plot(projected[edge, 0], projected[edge, 1], color=INK,
                lw=linewidth, ls=(0, (3, 3)) if hidden else "-", zorder=7)


def sphere_lines(ax, projection, radius=1, color=INK, alpha=0.35, fill=TEAL, fill_alpha=0.065):
    angle = np.linspace(0, 2 * np.pi, 241)
    ax.add_patch(Circle((0, 0), radius, facecolor=fill, edgecolor="none", alpha=fill_alpha, zorder=2))
    ax.add_patch(Circle((0, 0), radius, facecolor="none", edgecolor=color,
                       lw=1.6 if radius == 1 else 0.95, alpha=min(1, alpha * 2), zorder=3))
    for latitude in [-np.pi / 4, 0, np.pi / 4]:
        points = np.column_stack([
            np.cos(latitude) * np.cos(angle), np.cos(latitude) * np.sin(angle),
            np.full_like(angle, np.sin(latitude)),
        ]) * radius
        assert np.allclose(sdf(points), radius - 1, atol=1e-12)
        p = points @ projection.T
        p[p[:, 2] < 0, :2] = np.nan
        ax.plot(p[:, 0], p[:, 1], color=color, lw=0.75, alpha=alpha, zorder=3)
    for longitude in np.linspace(0, np.pi, 4, endpoint=False):
        points = np.column_stack([
            np.cos(angle) * np.cos(longitude),
            np.cos(angle) * np.sin(longitude), np.sin(angle),
        ]) * radius
        assert np.allclose(sdf(points), radius - 1, atol=1e-12)
        p = points @ projection.T
        p[p[:, 2] < 0, :2] = np.nan
        ax.plot(p[:, 0], p[:, 1], color=color, lw=0.75, alpha=alpha, zorder=3)


def continuous_field(fig, projection):
    ax = fig.add_axes([0.035, 0.265, 0.285, 0.64])
    sphere_lines(ax, projection, radius=1.4, color=ORANGE, alpha=0.22, fill_alpha=0)
    sphere_lines(ax, projection, radius=1, color=INK, alpha=0.65, fill=BLUE, fill_alpha=0.08)
    sphere_lines(ax, projection, radius=0.6, color=BLUE, alpha=0.50, fill=BLUE, fill_alpha=0.04)
    ax.annotate("$f=0$", xy=(0.6, 0.8), xytext=(0.97, 1.29),
                fontsize=17, ha="center", va="center",
                bbox=dict(fc=BG, ec="none", alpha=0.94, pad=2),
                arrowprops=dict(arrowstyle="-", color=INK, lw=1))
    ax.annotate("$f=-0.4$", xy=(0.6 / np.sqrt(2), -0.6 / np.sqrt(2)), xytext=(-0.03, 0.12),
                fontsize=14, color=BLUE, ha="center", va="center",
                bbox=dict(fc=BG, ec="none", alpha=0.94, pad=2),
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
    ax.annotate("$f=+0.4$", xy=(-0.84, -1.12), xytext=(-0.76, -1.56),
                fontsize=14, color=ORANGE, ha="center", va="center",
                bbox=dict(fc=BG, ec="none", alpha=0.94, pad=2),
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8))
    ax.set(xlim=(-1.6, 1.6), ylim=(-1.6, 1.6), aspect="equal")
    ax.axis("off")


def sampled_grid(fig, coords, points, values, corners, projection):
    ax = fig.add_axes([0.425, 0.195, 0.355, 0.73])
    segments = []
    for direction in range(3):
        other = [axis for axis in range(3) if axis != direction]
        for a, b in product(coords, repeat=2):
            ends = np.zeros((2, 3))
            ends[:, direction] = coords[[0, -1]]
            ends[:, other[0]] = a
            ends[:, other[1]] = b
            segments.append((ends @ projection.T)[:, :2])
    ax.add_collection(LineCollection(segments, colors=GRID, linewidths=0.65, alpha=0.80, zorder=1))
    sphere_lines(ax, projection)
    p = points @ projection.T
    order = np.argsort(p[:, 2])
    ax.scatter(p[order, 0], p[order, 1], c=sign_colors(values)[order],
               s=13, alpha=0.83, linewidths=0, zorder=4)
    selected = corners @ projection.T
    draw_cube(ax, selected, linewidth=2.1)
    ax.scatter(selected[:, 0], selected[:, 1], c=sign_colors(sdf(corners)),
               s=53, edgecolors=BG, linewidths=1.25, zorder=8)
    ax.set(xlim=(-1.9, 1.9), ylim=(-1.9, 1.9), aspect="equal")
    ax.axis("off")
    fig.text(0.6025, 0.15, r"$f_{ijk}=f(x_i,y_j,z_k)$", ha="center", fontsize=18)
    return ax, selected


def cell_detail(fig, corners, projection):
    ax = fig.add_axes([0.795, 0.30, 0.20, 0.54])
    local = cube([0, 0, 0], 1) @ projection.T
    values = sdf(corners)
    draw_cube(ax, local, linewidth=1.55)
    ax.scatter(local[:, 0], local[:, 1], c=sign_colors(values),
               s=70, edgecolors=BG, linewidths=1.5, zorder=9)
    locations = [
        (-0.48, -0.22), (0.82, -0.58), (1.87, 0.04), (-0.48, 0.33),
        (-0.48, 0.97), (1.87, 0.62), (1.87, 1.13), (0.57, 1.58),
    ]
    annotations = []
    for i, (value, location) in enumerate(zip(values, locations)):
        label = f"{value:+.2f}".replace("-", "−")
        annotations.append(ax.annotate(
            label, xy=local[i, :2], xytext=location, ha="center", va="center",
            fontsize=16, color=BLUE if value < 0 else ORANGE, zorder=10,
            bbox=dict(fc=BG, ec="none", pad=2),
            arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8, shrinkA=3, shrinkB=7),
        ))
    ax.set(xlim=(-0.9, 2.3), ylim=(-0.85, 1.88), aspect="equal")
    ax.axis("off")
    return ax, local, annotations


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
    coords = np.linspace(-1.2, 1.2, 5)
    h = coords[1] - coords[0]
    points = np.stack(np.meshgrid(coords, coords, coords, indexing="ij"), axis=-1).reshape(-1, 3)
    values = sdf(points)
    corners = cube([0.6, -0.6, 0], h)
    corner_values = sdf(corners)
    projection = basis()
    assert np.allclose(np.diff(coords), 0.6)
    assert points.shape == (125, 3)
    assert len(np.unique(corners, axis=0)) == 8
    assert EDGES.shape == (12, 2)
    assert np.allclose(np.linalg.norm(corners[EDGES[:, 0]] - corners[EDGES[:, 1]], axis=1), h)
    assert all(np.any(np.all(np.isclose(points, p), axis=1)) for p in corners)
    assert corner_values.min() < 0 < corner_values.max()
    assert np.sum(corner_values < 0) == 3
    assert np.allclose(sdf([[0, 0, 0], [1, 0, 0], [2, 0, 0]]), [-1, 0, 1])
    assert np.allclose(projection @ projection.T, np.eye(3))

    fig = plt.figure(figsize=(16, 7.6))
    continuous_field(fig, projection)
    grid_ax, selected = sampled_grid(fig, coords, points, values, corners, projection)
    detail_ax, local, annotations = cell_detail(fig, corners, projection)
    canvas = fig.add_axes([0, 0, 1, 1], frameon=False, zorder=0)
    canvas.set(xlim=(0, 1), ylim=(0, 1))
    canvas.axis("off")
    canvas.add_patch(FancyArrowPatch((0.343, 0.57), (0.415, 0.57),
                                    arrowstyle="-|>", mutation_scale=19, lw=1.7, color=INK))
    canvas.text(0.379, 0.62, "sampling", fontsize=19, ha="center")
    connection = ConnectionPatch(
        xyA=selected[[1, 5], :2].mean(axis=0), coordsA="data", axesA=grid_ax,
        xyB=local[0, :2] + [-0.10, 0.02], coordsB="data", axesB=detail_ax,
        arrowstyle="-|>", mutation_scale=12, color=GRAY, lw=1.0,
        linestyle=(0, (3, 3)), shrinkA=5, shrinkB=6,
    )
    fig.add_artist(connection)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [item.get_bbox_patch().get_window_extent(renderer) for item in annotations]
    marker_positions = detail_ax.transData.transform(local[:, :2])
    for i, box in enumerate(boxes):
        assert detail_ax.bbox.contains(box.x0, box.y0) and detail_ax.bbox.contains(box.x1, box.y1)
        assert all(not box.overlaps(other) for other in boxes[i + 1:])
        assert all(not box.padded(7).contains(*point) for point in marker_positions)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "sdf_grid.svg"
    fig.savefig(path, facecolor=BG,
                metadata={"Creator": "lab_6 / Python / NumPy / Matplotlib"})
    print(path.relative_to(ROOT))
    plt.close(fig)
    print("Regular grid: 125 samples, h=0.6")
    print("Selected cube: 8 corners, 12 edges; 3 negative and 5 positive SDF samples")
    print("Corner values:", np.round(corner_values, 6))
    print("Geometry, field values and label layout: OK")


if __name__ == "__main__":
    main()
