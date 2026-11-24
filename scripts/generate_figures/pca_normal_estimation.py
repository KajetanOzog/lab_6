from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyArrowPatch, Polygon


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "images" / "point_clouds"
BG = "#F8FAFC"
INK = "#0F172A"
BLUE = "#2563EB"
ORANGE = "#EA580C"
TEAL = "#0F766E"
GRAY = "#94A3B8"
NEIGHBORS = 24


def make_cloud():
    coordinate = np.arange(-7, 8) * 0.16
    x, y = np.meshgrid(coordinate, coordinate)
    mask = x * x + y * y <= 1.05 ** 2
    x, y = x[mask], y[mask]
    points = np.column_stack((x, y, 0.24 * (x * x - y * y)))
    angle_x, angle_y = np.deg2rad([-12, 18])
    rotation_x = np.array([
        [1, 0, 0],
        [0, np.cos(angle_x), -np.sin(angle_x)],
        [0, np.sin(angle_x), np.cos(angle_x)],
    ])
    rotation_y = np.array([
        [np.cos(angle_y), 0, np.sin(angle_y)],
        [0, 1, 0],
        [-np.sin(angle_y), 0, np.cos(angle_y)],
    ])
    rotation = rotation_y @ rotation_x
    assert np.allclose(rotation @ rotation.T, np.eye(3))
    points = points @ rotation.T
    query_id = int(np.argmin(np.linalg.norm(points, axis=1)))
    return points, query_id


def fit_pca(points, query_id):
    squared_distance = np.sum((points - points[query_id]) ** 2, axis=1)
    order = np.argsort(squared_distance, kind="stable")
    neighbors = order[order != query_id][:NEIGHBORS]
    local = points[neighbors]
    center = local.mean(axis=0)
    centered = local - center
    covariance = centered.T @ centered / len(local)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normal = eigenvectors[:, 0]
    if normal[2] < 0:
        normal = -normal
    tangents = eigenvectors[:, 1:]
    local_uv = centered @ tangents
    lower = local_uv.min(axis=0) - 0.10
    upper = local_uv.max(axis=0) + 0.10
    uv = np.array([
        [lower[0], lower[1]], [upper[0], lower[1]],
        [upper[0], upper[1]], [lower[0], upper[1]],
    ])
    plane = center + uv @ tangents.T
    assert len(neighbors) == NEIGHBORS and query_id not in neighbors
    assert np.all(np.diff(eigenvalues) >= 0)
    assert np.isclose(np.linalg.norm(normal), 1)
    assert np.allclose(tangents.T @ normal, 0, atol=1e-14)
    assert np.allclose((plane - center) @ normal, 0, atol=1e-14)
    assert np.allclose(covariance @ normal, eigenvalues[0] * normal, atol=1e-14)
    assert np.isclose(np.mean((centered @ normal) ** 2), eigenvalues[0])
    assert abs((points[query_id] - center) @ normal) < 1e-14
    print(f"Cloud: {len(points)} points; selected neighborhood: {len(neighbors)}", flush=True)
    print("PCA eigenvalues:", np.round(eigenvalues, 8), flush=True)
    print("Unit normal:", np.round(normal, 8), flush=True)
    return neighbors, center, plane, normal


def camera_basis():
    azimuth, elevation = np.deg2rad([-58, 28])
    direction = np.array([
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation),
    ])
    right = np.array([-np.sin(azimuth), np.cos(azimuth), 0])
    up = np.cross(direction, right)
    basis = np.stack((right, up, direction))
    assert np.allclose(basis @ basis.T, np.eye(3))
    return basis


def portable_svg_opacity(path):
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    tree = ET.parse(path)
    for element in tree.iter():
        properties = [
            part.strip() for part in element.get("style", "").split(";") if part.strip()
        ]
        remaining = []
        for property_text in properties:
            name, value = property_text.split(":", 1)
            if name.strip() in ("fill-opacity", "stroke-opacity", "opacity"):
                element.set(name.strip(), value.strip())
            else:
                remaining.append(property_text)
        if properties:
            element.set("style", "; ".join(remaining))
    tree.write(path, encoding="utf-8", xml_declaration=True)


def render(points, query_id, neighbors, center, plane, normal):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "text.color": INK,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "svg.fonttype": "path",
        "svg.hashsalt": "lab6-pca-normal-estimation",
    })
    basis = camera_basis()
    projected = points @ basis.T
    plane_2d = (plane @ basis.T)[:, :2]
    query = points[query_id]
    point = projected[query_id, :2]
    endpoint = ((query + 0.90 * normal) @ basis.T)[:2]
    fig = plt.figure(figsize=(9.0, 6.4))
    ax = fig.add_axes([0.025, 0.025, 0.95, 0.95])
    selected = np.zeros(len(points), dtype=bool)
    selected[neighbors] = True
    selected[query_id] = True
    background = ~selected
    context_color = 0.35 * np.array(to_rgb(BLUE)) + 0.65 * np.array(to_rgb(BG))
    ax.scatter(projected[background, 0], projected[background, 1], s=28,
               color=context_color, linewidths=0, zorder=1)
    signed_distance = (points[neighbors] - center) @ normal
    behind = neighbors[signed_distance < -1e-12]
    front = neighbors[signed_distance >= -1e-12]
    ax.scatter(projected[behind, 0], projected[behind, 1], s=64, color=BLUE,
               edgecolors=BG, linewidths=0.75, zorder=2)
    ax.add_patch(Polygon(
        plane_2d, closed=True, facecolor=(*to_rgb(TEAL), 0.16),
        edgecolor=TEAL, linewidth=1.5, zorder=3,
    ))
    ax.scatter(projected[front, 0], projected[front, 1], s=64, color=BLUE,
               edgecolors=BG, linewidths=0.75, zorder=4)
    ax.scatter(*point, s=165, color=ORANGE, edgecolors=BG, linewidths=1.5, zorder=7)
    ax.add_patch(FancyArrowPatch(
        point, endpoint, arrowstyle="-|>", mutation_scale=26,
        color=INK, linewidth=3.0, shrinkA=8, shrinkB=0, zorder=8,
    ))
    tangent = np.cross(normal, basis[2])
    tangent /= np.linalg.norm(tangent)
    if (tangent @ basis.T)[0] < 0:
        tangent = -tangent
    marker = query + 0.105 * np.array([tangent, tangent + normal, normal])
    marker = marker @ basis.T
    assert np.isclose(tangent @ normal, 0, atol=1e-14)
    assert np.isclose((tangent @ basis.T)[:2] @ (normal @ basis.T)[:2], 0, atol=1e-14)
    ax.plot(marker[:, 0], marker[:, 1], color=INK, linewidth=1.2, zorder=6)
    labels = [
        ax.text(point[0] - 0.13, point[1] - 0.15, "$p$", color=ORANGE,
                fontsize=23, ha="center", va="center", zorder=10),
        ax.text(endpoint[0] + 0.08, endpoint[1] + 0.04, r"$\mathbf{n}$",
                fontsize=25, ha="left", va="center", zorder=10),
    ]
    neighbor_anchor = projected[neighbors[np.argmin(projected[neighbors, 0])], :2]
    labels.append(ax.annotate(
        r"$\mathcal{N}_{24}(p)$", xy=neighbor_anchor, xytext=(-1.10, -0.45),
        fontsize=21, color=BLUE, ha="center", va="center",
        arrowprops={"arrowstyle": "-", "color": BLUE, "lw": 1.0, "shrinkB": 7},
        bbox={"facecolor": BG, "edgecolor": "none", "pad": 2}, zorder=10,
    ))
    corner_id = np.argmax(plane_2d[:, 0] - 0.5 * plane_2d[:, 1])
    labels.append(ax.annotate(
        "płaszczyzna PCA", xy=plane_2d[corner_id], xytext=(0.90, -0.50),
        fontsize=18, color=TEAL, ha="center", va="center",
        arrowprops={"arrowstyle": "-", "color": TEAL, "lw": 1.0, "shrinkB": 4},
        bbox={"facecolor": BG, "edgecolor": "none", "pad": 2}, zorder=10,
    ))
    ax.set(xlim=(-1.50, 1.50), ylim=(-0.85, 1.15), aspect="equal")
    ax.axis("off")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [label.get_window_extent(renderer) for label in labels]
    for i, box in enumerate(boxes):
        assert fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1)
        assert all(not box.overlaps(other) for other in boxes[i + 1:])
    assert np.allclose((query + 0.90 * normal - query), 0.90 * normal)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "pca_normal_estimation.svg"
    fig.savefig(path, facecolor=BG, metadata={
        "Creator": "lab_6 / NumPy PCA / Matplotlib",
        "Description": (
            "Synthetic curved saddle point cloud; 24 nearest neighbors by Euclidean distance. "
            "Plane through neighborhood centroid; normal is the smallest-eigenvalue eigenvector "
            "of the actual 3D neighborhood covariance. PCA sign fixed by positive world Z."
        ),
    })
    portable_svg_opacity(path)
    plt.close(fig)
    print(path.relative_to(ROOT), flush=True)
    print("PCA plane, perpendicular unit normal and label layout: OK", flush=True)


def main():
    points, query_id = make_cloud()
    neighbors, center, plane, normal = fit_pca(points, query_id)
    render(points, query_id, neighbors, center, plane, normal)


if __name__ == "__main__":
    main()
