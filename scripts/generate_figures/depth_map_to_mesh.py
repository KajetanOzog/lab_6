from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "images" / "depth_maps"
BG = "#F8FAFC"
INK = "#0F172A"
BLUE = "#2563EB"
TEAL = "#0F766E"
ORANGE = "#EA580C"
RED = "#DC2626"
SECONDARY = "#64748B"
WIDTH, HEIGHT = 96, 78
FX = FY = 96.0
CX, CY = (WIDTH - 1) / 2, (HEIGHT - 1) / 2
SPHERE_CENTER = np.array([0.0, 0.0, 3.0])
SPHERE_RADIUS = 0.8
WALL_Z = 5.0
DEPTH_THRESHOLD = 0.4


def generate_depth():
    u, v = np.meshgrid(np.arange(WIDTH), np.arange(HEIGHT))
    rays = np.stack(((u - CX) / FX, (v - CY) / FY, np.ones_like(u)), axis=-1)
    a = np.einsum("...i,...i->...", rays, rays)
    b = -2 * np.einsum("...i,i->...", rays, SPHERE_CENTER)
    c = SPHERE_CENTER @ SPHERE_CENTER - SPHERE_RADIUS ** 2
    discriminant = b * b - 4 * a * c
    sphere_z = (-b - np.sqrt(np.maximum(discriminant, 0))) / (2 * a)
    sphere_mask = (discriminant >= 0) & (sphere_z > 0) & (sphere_z < WALL_Z)
    depth = np.where(sphere_mask, sphere_z, WALL_Z)
    assert depth.shape == (HEIGHT, WIDTH) and np.all(depth > 0)
    assert np.any(sphere_mask) and np.any(~sphere_mask)
    print(
        f"Depth image {WIDTH} x {HEIGHT}; sphere={np.count_nonzero(sphere_mask)} pixels; "
        f"wall={np.count_nonzero(~sphere_mask)} pixels",
        flush=True,
    )
    return depth, sphere_mask


def back_project(depth):
    u, v = np.meshgrid(np.arange(WIDTH), np.arange(HEIGHT))
    points = np.stack((
        (u - CX) * depth / FX,
        (v - CY) * depth / FY,
        depth,
    ), axis=-1).reshape(-1, 3)
    assert np.allclose(CX + FX * points[:, 0] / points[:, 2], u.ravel())
    assert np.allclose(CY + FY * points[:, 1] / points[:, 2], v.ravel())
    return points


def triangulate(points, sphere_mask):
    row, column = np.meshgrid(np.arange(HEIGHT - 1), np.arange(WIDTH - 1), indexing="ij")
    a = (row * WIDTH + column).ravel()
    b, d = a + 1, a + WIDTH
    c = d + 1
    faces = np.stack((
        np.column_stack((a, d, b)),
        np.column_stack((b, d, c)),
    ), axis=1).reshape(-1, 3)
    assert len(faces) == 2 * (WIDTH - 1) * (HEIGHT - 1)
    triangles = points[faces]
    depth_range = np.ptp(triangles[:, :, 2], axis=1)
    discarded = depth_range > DEPTH_THRESHOLD
    labels = sphere_mask.ravel()[faces]
    crosses_objects = np.any(labels, axis=1) & ~np.all(labels, axis=1)
    assert np.array_equal(discarded, crosses_objects)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas2 = np.linalg.norm(normals, axis=1)
    assert np.all(areas2 > 1e-12)
    normals /= areas2[:, None]
    print(
        f"Raw triangles={len(faces)}, rejected={np.count_nonzero(discarded)}, "
        f"retained={np.count_nonzero(~discarded)}; threshold={DEPTH_THRESHOLD}",
        flush=True,
    )
    print(
        f"Within-surface max delta Z={depth_range[~crosses_objects].max():.6f}; "
        f"cross-surface min delta Z={depth_range[crosses_objects].min():.6f}",
        flush=True,
    )
    return faces, normals, discarded


def camera_basis():
    direction = np.array([0.65, -0.25, -1.3])
    direction /= np.linalg.norm(direction)
    right = np.cross(np.array([0, -1, 0]), direction)
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    basis = np.stack((right, up, direction))
    assert np.allclose(basis @ basis.T, np.eye(3))
    return basis


def render(depth, sphere_mask, points, faces, normals, discarded):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "text.color": INK,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "svg.fonttype": "path",
        "svg.hashsalt": "lab6-single-depth-to-mesh",
    })
    basis = camera_basis()
    target = np.array([0, 0, 3.7])
    projected = (points - target) @ basis.T
    lower = projected[:, :2].min(axis=0)
    upper = projected[:, :2].max(axis=0)
    center = (lower + upper) / 2
    half = (upper - lower) * 0.55
    xlim = (center[0] - half[0], center[0] + half[0])
    ylim = (center[1] - half[1], center[1] + half[1])
    fig = plt.figure(figsize=(13.6, 10.5))
    positions = ((0.04, 0.63), (0.54, 0.63), (0.04, 0.13), (0.54, 0.13))
    width, height = 0.42, 0.32
    labels = []
    axes = []
    depth_ax = fig.add_axes([*positions[0], width, height])
    cmap = LinearSegmentedColormap.from_list("depth_blue_orange", [BLUE, BG, ORANGE], N=256)
    depth_min = SPHERE_CENTER[2] - SPHERE_RADIUS
    image = depth_ax.imshow(
        depth, cmap=cmap, norm=Normalize(depth_min, WALL_Z),
        origin="upper", interpolation="nearest",
    )
    depth_ax.axis("off")
    color_ax = fig.add_axes([positions[0][0] + 0.05, 0.579, width - 0.10, 0.014])
    bar = fig.colorbar(image, cax=color_ax, orientation="horizontal",
                       ticks=[depth_min, (depth_min + WALL_Z) / 2, WALL_Z])
    bar.set_ticklabels(["2,2", "3,6", "5,0"])
    bar.ax.tick_params(labelsize=12, length=3, colors=SECONDARY)
    bar.outline.set_visible(False)
    color_ax.text(-0.075, 0.5, "$Z$", transform=color_ax.transAxes,
                  fontsize=16, ha="right", va="center")
    for left, bottom in positions[1:]:
        ax = fig.add_axes([left, bottom, width, height])
        ax.set(xlim=xlim, ylim=ylim, aspect="equal")
        ax.axis("off")
        axes.append(ax)
    light = basis[2] + np.array([0, -0.35, 0])
    light /= np.linalg.norm(light)
    point_normals = np.tile(np.array([0, 0, -1.0]), (len(points), 1))
    point_normals[sphere_mask.ravel()] = (
        points[sphere_mask.ravel()] - SPHERE_CENTER
    ) / SPHERE_RADIUS
    point_shade = 0.78 + 0.22 * np.maximum(point_normals @ light, 0)
    order = np.argsort(projected[:, 2], kind="stable")
    axes[0].scatter(
        projected[order, 0], projected[order, 1], s=4.0 * (64 / WIDTH) ** 2,
        c=np.array(to_rgb(BLUE))[None, :] * point_shade[order, None],
        edgecolors="none", linewidths=0,
    )
    projected_triangles = projected[faces]
    shade = 0.74 + 0.26 * np.maximum(normals @ light, 0)
    colors = np.array(to_rgb(TEAL))[None, :] * shade[:, None]
    raw_colors = colors.copy()
    raw_colors[discarded] = np.array(to_rgb(RED)) * shade[discarded, None]
    for ax, mask, face_colors in [
        (axes[1], np.ones(len(faces), bool), raw_colors),
        (axes[2], ~discarded, colors),
    ]:
        indices = np.where(mask)[0]
        indices = indices[np.argsort(projected_triangles[indices, :, 2].mean(axis=1), kind="stable")]
        edge_colors = face_colors[indices].copy()
        if ax is axes[1]:
            edge_colors[discarded[indices]] = np.array(to_rgb("#991B1B")) * shade[indices[discarded[indices]], None]
        ax.add_collection(PolyCollection(
            projected_triangles[indices, :, :2], facecolors=face_colors[indices],
            edgecolors=edge_colors, linewidths=0.40, antialiaseds=False, joinstyle="round",
        ))
    for (left, bottom), title in zip(positions, (
        "Mapa głębi", "Chmura punktów", "Mesh bez filtracji", "Mesh po filtracji",
    )):
        caption_y = 0.515 if bottom > 0.5 else 0.075
        labels.append(fig.text(left + width / 2, caption_y, title, fontsize=21,
                               ha="center", va="center"))
    labels.extend((
        fig.text(positions[2][0] + width / 2, 0.025, "czerwone: błędne połączenia",
                 fontsize=13, color=RED, ha="center", va="center"),
        fig.text(positions[3][0] + width / 2, 0.025, r"próg różnicy $Z$: 0,4",
                 fontsize=13, color=SECONDARY, ha="center", va="center"),
    ))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = [label.get_window_extent(renderer) for label in labels]
    for i, box in enumerate(bounds):
        assert fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1)
        assert all(not box.overlaps(other) for other in bounds[i + 1:])
    scales = [
        (np.linalg.norm(ax.transData.transform((1, 0)) - ax.transData.transform((0, 0))),
         np.linalg.norm(ax.transData.transform((0, 1)) - ax.transData.transform((0, 0))))
        for ax in axes
    ]
    assert np.allclose(scales, scales[0][0])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "depth_map_to_mesh.svg"
    fig.savefig(path, facecolor=BG, metadata={
        "Creator": "lab_6 / NumPy / Matplotlib",
        "Description": (
            f"Synthetic pinhole camera: {WIDTH}x{HEIGHT}, fx=fy={FX}, cx={CX}, cy={CY}, "
            f"positive Y down, positive Z forward. Nearest analytic ray-sphere hit for "
            f"sphere(center={SPHERE_CENTER.tolist()}, radius={SPHERE_RADIUS}) and plane Z={WALL_Z}. "
            f"Depth stores optical Z, not ray length. Back-projection uses only the depth map. "
            f"Two triangles per pixel quad. Red faces connect different original objects. "
            f"Filtering uses only max(Z)-min(Z) <= {DEPTH_THRESHOLD}, not object labels. "
            f"Observed surfaces only; no hidden sphere or occluded wall geometry is inserted. "
            f"All 3D panels share an orthographic camera, light and physical scale."
        ),
    })
    plt.close(fig)
    print(path.relative_to(ROOT), flush=True)


def main():
    depth, sphere_mask = generate_depth()
    points = back_project(depth)
    assert np.max(np.abs(np.linalg.norm(points[sphere_mask.ravel()] - SPHERE_CENTER, axis=1) - SPHERE_RADIUS)) < 1e-12
    assert np.allclose(points[~sphere_mask.ravel(), 2], WALL_Z)
    faces, normals, discarded = triangulate(points, sphere_mask)
    render(depth, sphere_mask, points, faces, normals, discarded)
    print("Analytic visibility, true back-projection, raw pixel triangles, depth-only filter and shared view: OK")


if __name__ == "__main__":
    main()
