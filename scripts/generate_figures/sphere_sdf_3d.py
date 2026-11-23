from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "images" / "sdf"
BG = "#F8FAFC"
INK = "#0F172A"
BLUE = "#2563EB"
TEAL = "#0F766E"
ORANGE = "#EA580C"
GRAY = "#94A3B8"
SECONDARY = "#64748B"


def sdf(points):
    return np.linalg.norm(np.asarray(points), axis=-1) - 1


def camera_basis():
    azimuth, elevation = np.deg2rad([-58, 24])
    direction = np.array([
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation),
    ])
    right = np.array([-np.sin(azimuth), np.cos(azimuth), 0.0])
    up = np.cross(direction, right)
    return np.stack([right, up, direction])


def hemisphere(nrings=24, ntheta=96):
    phi = np.linspace(0, np.pi / 2, nrings + 1)[1:]
    theta = np.arange(ntheta) * 2 * np.pi / ntheta
    p, t = np.meshgrid(phi, theta, indexing="ij")
    rings = np.stack([
        np.sin(p) * np.cos(t), np.cos(p), np.sin(p) * np.sin(t),
    ], axis=-1).reshape(-1, 3)
    vertices = np.vstack([[0, 1, 0], rings])
    faces = [[0, 1 + (j + 1) % ntheta, 1 + j] for j in range(ntheta)]
    for i in range(nrings - 1):
        for j in range(ntheta):
            a = 1 + i * ntheta + j
            b = 1 + (i + 1) * ntheta + j
            c = 1 + (i + 1) * ntheta + (j + 1) % ntheta
            d = 1 + i * ntheta + (j + 1) % ntheta
            faces.extend([[a, c, b], [a, d, c]])
    return vertices, np.asarray(faces)


def arrow(ax, start, end, color, double=False, width=1.4, scale=13, zorder=5):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="<->" if double else "-|>",
        color=color, linewidth=width, mutation_scale=scale,
        shrinkA=0, shrinkB=0, zorder=zorder,
    ))


def convex_hull(points):
    points = sorted(map(tuple, np.unique(points, axis=0)))

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1])


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "font.size": 16,
        "text.color": INK,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "svg.fonttype": "path",
    })
    points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    values = sdf(points)
    vertices, faces = hemisphere()
    basis = camera_basis()
    assert np.allclose(values, [-1, 0, 1], atol=1e-12)
    assert np.allclose(sdf(vertices), 0, atol=1e-12)
    assert np.all(vertices[:, 1] >= -1e-12)
    assert np.allclose(basis @ basis.T, np.eye(3), atol=1e-12)
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    assert np.all(np.linalg.norm(normals, axis=1) > 1e-10)
    assert np.all(np.einsum("ij,ij->i", normals, triangles.mean(axis=1)) > 0)
    projected = vertices @ basis.T
    markers = points @ basis.T

    fig = plt.figure(figsize=(11.0, 6.5))
    ax = fig.add_axes([0.035, 0.08, 0.93, 0.88])
    silhouette = convex_hull(projected[:, :2])
    ax.add_patch(Polygon(
        silhouette, facecolor=TEAL, edgecolor=TEAL,
        linewidth=1.0, alpha=0.16, zorder=1,
    ))
    angle = np.linspace(0, 2 * np.pi, 361)
    section_3d = np.column_stack([np.cos(angle), np.zeros_like(angle), np.sin(angle)])
    assert np.allclose(sdf(section_3d), 0, atol=1e-12)
    section = section_3d @ basis.T
    ax.add_patch(Polygon(section[:, :2], facecolor=BLUE, edgecolor="none", alpha=0.055, zorder=3))

    theta = np.linspace(0, np.pi, 201)
    for latitude in [-np.pi / 4, 0, np.pi / 4]:
        line = np.column_stack([
            np.cos(latitude) * np.cos(theta),
            np.cos(latitude) * np.sin(theta),
            np.full_like(theta, np.sin(latitude)),
        ]) @ basis.T
        ax.plot(line[:, 0], line[:, 1], color=TEAL, lw=0.85, alpha=0.55, zorder=2)
    latitude = np.linspace(-np.pi / 2, np.pi / 2, 201)
    for longitude in [np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
        line = np.column_stack([
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        ]) @ basis.T
        ax.plot(line[:, 0], line[:, 1], color=TEAL, lw=0.85, alpha=0.55, zorder=2)
    ax.plot(section[:, 0], section[:, 1], color=INK, lw=2.0, zorder=4)

    axis_start = np.array([-0.95, 0, 0]) @ basis.T
    axis_end = np.array([2.85, 0, 0]) @ basis.T
    arrow(ax, axis_start[:2], axis_end[:2], GRAY, width=0.85, scale=11, zorder=5)
    ax.text(axis_end[0] + 0.09, axis_end[1] - 0.01, "$x$", fontsize=18, va="center")

    for i, color in enumerate([BLUE, INK, ORANGE]):
        ax.plot(*markers[i, :2], "o", ms=11, mfc=color, mec=BG, mew=1.8, zorder=9)
    annotations = []
    for i, (name, location, color) in enumerate([
        ("A", (-0.36, 1.30), BLUE),
        ("B", (1.36, 1.03), INK),
        ("C", (2.20, 0.24), ORANGE),
    ]):
        coordinates = ",".join(str(int(v)) for v in points[i])
        value = f"+{int(values[i])}" if values[i] > 0 else str(int(values[i]))
        label = "$" + f"{name}=({coordinates})" + "$\n$" + f"f({name})={value}" + "$"
        annotations.append(ax.annotate(
            label, xy=markers[i, :2], xytext=location, ha="center", va="center",
            fontsize=19, color=color, linespacing=1.55, zorder=10,
            bbox=dict(facecolor=BG, edgecolor="none", alpha=0.96, pad=4),
            arrowprops=dict(arrowstyle="-", color=color, lw=1.1, shrinkA=6, shrinkB=10),
        ))
    for i, color, label in [(0, BLUE, "$r=1$"), (1, ORANGE, "$d=1$")]:
        start = markers[i, :2] + [0, -0.23]
        end = markers[i + 1, :2] + [0, -0.23]
        ax.plot([markers[i, 0], start[0]], [markers[i, 1] - 0.04, start[1] - 0.02],
                color=color, lw=0.7, zorder=6)
        ax.plot([markers[i + 1, 0], end[0]], [markers[i + 1, 1] - 0.04, end[1] - 0.02],
                color=color, lw=0.7, zorder=6)
        offset = (end - start) * 0.025
        arrow(ax, start + offset, end - offset, color, double=True, zorder=7)
        center = (start + end) / 2
        ax.text(center[0], center[1] - 0.20, label, ha="center", va="center",
                fontsize=17, color=color, zorder=8)

    ax.text(-0.83, -1.18, r"przekrój $y=0$", color=SECONDARY, fontsize=13, ha="center")
    ax.set(xlim=(-1.23, 2.77), ylim=(-1.32, 1.63), aspect="equal")
    ax.axis("off")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [item.get_bbox_patch().get_window_extent(renderer) for item in annotations]
    for i, box in enumerate(boxes):
        assert ax.bbox.contains(box.x0, box.y0) and ax.bbox.contains(box.x1, box.y1)
        assert all(not box.overlaps(other) for other in boxes[i + 1:])

    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "sphere_sdf_3d.svg"
    fig.savefig(path, facecolor=BG,
                metadata={"Creator": "lab_6 / Python / NumPy / Matplotlib"})
    print(path.relative_to(ROOT))
    plt.close(fig)
    print("SDF(A)=-1; SDF(B)=0; SDF(C)=+1")
    print("Unit radius, surface geometry, projection and label layout: OK")


if __name__ == "__main__":
    main()
