from pathlib import Path

import contourpy
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "images" / "marching_cubes"
BG = "#F8FAFC"
INK = "#0F172A"
BLUE = "#2563EB"
TEAL = "#0F766E"
ORANGE = "#EA580C"
GRID = "#CBD5E1"


def sdf(x, y):
    return np.sqrt(np.asarray(x) ** 2 + np.asarray(y) ** 2) - 1


def extract_contour(coords, field):
    generator = contourpy.contour_generator(
        x=coords, y=coords, z=field, name="serial",
        line_type="Separate", corner_mask=False, quad_as_tri=False,
        z_interp="Linear",
    )
    contours = generator.lines(0.0)
    assert len(contours) == 1
    return contours[0]


def validate_contour(coords, field, contour):
    h = coords[1] - coords[0]
    assert field.shape == (len(coords), len(coords))
    assert np.allclose(np.diff(coords), h)
    assert h > 0
    assert np.all(np.isfinite(contour))
    assert np.allclose(contour[0], contour[-1], atol=1e-12)
    assert len(np.unique(contour[:-1], axis=0)) == len(contour) - 1
    assert np.all(np.abs(field) > 1e-12)
    assert np.allclose(sdf([0, 1, 2], [0, 0, 0]), [-1, 0, 1])

    for x, y in contour:
        u, v = (np.array([x, y]) - coords[0]) / h
        vertical = np.isclose(u, np.rint(u), atol=1e-10)
        horizontal = np.isclose(v, np.rint(v), atol=1e-10)
        assert vertical != horizontal
        if vertical:
            i, j = int(np.rint(u)), int(np.floor(v))
            a, b, t = field[j, i], field[j + 1, i], v - j
        else:
            i, j = int(np.floor(u)), int(np.rint(v))
            a, b, t = field[j, i], field[j, i + 1], u - i
        assert a * b < 0
        assert 0 < t < 1
        assert abs((1 - t) * a + t * b) < 1e-11

    starts = contour[:-1]
    edges = np.diff(contour, axis=0)
    lengths2 = np.sum(edges * edges, axis=1)
    assert np.all(lengths2 > 1e-12)
    centers = starts + edges / 2
    cells = np.floor((centers - coords[0]) / h).astype(int)
    assert np.all((cells >= 0) & (cells < len(coords) - 1))
    quad_values = np.stack([
        field[:-1, :-1], field[:-1, 1:], field[1:, 1:], field[1:, :-1],
    ], axis=-1)
    active = (quad_values.min(axis=-1) < 0) & (quad_values.max(axis=-1) > 0)
    assert np.all(active[cells[:, 1], cells[:, 0]])
    assert len(np.unique(cells, axis=0)) == len(edges) == int(active.sum())

    t = np.clip(-np.sum(starts * edges, axis=1) / lengths2, 0, 1)
    nearest = starts + t[:, None] * edges
    radii_min = np.linalg.norm(nearest, axis=1)
    radii_vertices = np.linalg.norm(contour, axis=1)
    assert radii_vertices.max() <= 1 + 1e-12
    vertex_error = np.max(np.abs(radii_vertices - 1))
    segment_error = np.max(1 - radii_min)
    assert segment_error < h * h / 4
    area = abs(np.sum(starts[:, 0] * contour[1:, 1] - contour[1:, 0] * starts[:, 1])) / 2
    winding = np.sum(np.arctan2(
        starts[:, 0] * contour[1:, 1] - starts[:, 1] * contour[1:, 0],
        np.sum(starts * contour[1:], axis=1),
    ))
    assert np.isclose(abs(winding), 2 * np.pi)
    assert np.pi * (1 - segment_error) ** 2 - 1e-12 <= area <= np.pi + 1e-12
    return len(edges), vertex_error, segment_error, area


def draw_samples(ax, coords, field):
    segments = [
        [[coords[0], position], [coords[-1], position]] for position in coords
    ] + [
        [[position, coords[0]], [position, coords[-1]]] for position in coords
    ]
    ax.add_collection(LineCollection(segments, colors=GRID, linewidths=0.85, zorder=1))
    x, y = np.meshgrid(coords, coords)
    colors = np.where(field.ravel() < 0, BLUE, ORANGE)
    ax.scatter(x.ravel(), y.ravel(), c=colors, s=260, edgecolors=BG,
               linewidths=1.0, zorder=4)
    limits = (coords[0] - 0.14, coords[-1] + 0.14)
    ax.set(xlim=limits, ylim=limits, aspect="equal")
    ax.axis("off")


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "font.size": 16,
        "text.color": INK,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "svg.fonttype": "path",
        "path.simplify": False,
    })
    coords = np.linspace(-1.65, 1.65, 7)
    x, y = np.meshgrid(coords, coords)
    field = sdf(x, y)
    contour = extract_contour(coords, field)
    count, vertex_error, segment_error, area = validate_contour(coords, field, contour)

    fig = plt.figure(figsize=(14, 6.8))
    left = fig.add_axes([0.035, 0.18, 0.405, 0.78])
    right = fig.add_axes([0.56, 0.18, 0.405, 0.78])
    for ax in [left, right]:
        draw_samples(ax, coords, field)
    right.plot(
        contour[:, 0], contour[:, 1], color=TEAL, lw=5.2,
        marker="o", ms=4.2, mfc=TEAL, mec=TEAL, mew=0,
        solid_joinstyle="miter", solid_capstyle="butt", zorder=3,
    )
    fig.text(0.2375, 0.075, "Próbki pola skalarnego",
             ha="center", va="center", fontsize=23)
    fig.text(0.7625, 0.075, "Odtworzony kontur",
             ha="center", va="center", fontsize=23)
    arrow_ax = fig.add_axes([0, 0, 1, 1], frameon=False, zorder=0)
    arrow_ax.set(xlim=(0, 1), ylim=(0, 1))
    arrow_ax.axis("off")
    arrow_ax.add_patch(FancyArrowPatch(
        (0.463, 0.56), (0.537, 0.56), arrowstyle="-|>",
        mutation_scale=25, lw=3.2, color=INK,
    ))
    fig.canvas.draw()
    assert left.get_xlim() == right.get_xlim()
    assert left.get_ylim() == right.get_ylim()
    assert np.allclose(left.bbox.size, right.bbox.size)
    assert np.array_equal(left.collections[1].get_offsets(), right.collections[1].get_offsets())
    assert np.array_equal(left.collections[1].get_facecolors(), right.collections[1].get_facecolors())

    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "marching_squares_intro.svg"
    fig.savefig(path, facecolor=BG,
                metadata={"Creator": "lab_6 / Python / NumPy / Matplotlib / ContourPy"})
    print(path.relative_to(ROOT))
    plt.close(fig)
    n = len(coords)
    print(f"Grid: {n-1} x {n-1} cells; {n} x {n} identical samples in both panels")
    print(f"Closed contour: {count} straight segments")
    print(f"Maximum vertex radial error: {vertex_error:.8f}")
    print(f"Maximum full-segment radial error: {segment_error:.8f}")
    print(f"Contour polygon area: {area:.8f}; exact circle area: {np.pi:.8f}")
    print("Grid-edge intersections, active cells, closure and geometry: OK")


if __name__ == "__main__":
    main()
