from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgb
from skimage.measure import marching_cubes


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "images" / "marching_cubes"
BG = "#F8FAFC"
INK = "#0F172A"
TEAL = "#0F766E"
EDGE = "#084740"
SECONDARY = "#64748B"
RESOLUTIONS = (8, 16, 32, 64, 128)
DOMAIN = (-1.5, 1.5)
SEED = 1400
LOWER_Z, LOWER_RADIUS = -0.72, 0.52
UPPER_Z, UPPER_RADIUS = 0.72, 0.48
NECK_RADIUS = 0.18
COLLAR_RADIUS = 0.34
COLLAR_HALF_HEIGHT = 0.035
LOWER_JOIN = LOWER_Z + np.sqrt(LOWER_RADIUS ** 2 - NECK_RADIUS ** 2)
UPPER_JOIN = UPPER_Z - np.sqrt(UPPER_RADIUS ** 2 - NECK_RADIUS ** 2)
PROFILE_SEGMENTS = np.array([
    [[NECK_RADIUS, LOWER_JOIN], [NECK_RADIUS, -COLLAR_HALF_HEIGHT]],
    [[NECK_RADIUS, -COLLAR_HALF_HEIGHT], [COLLAR_RADIUS, -COLLAR_HALF_HEIGHT]],
    [[COLLAR_RADIUS, -COLLAR_HALF_HEIGHT], [COLLAR_RADIUS, COLLAR_HALF_HEIGHT]],
    [[COLLAR_RADIUS, COLLAR_HALF_HEIGHT], [NECK_RADIUS, COLLAR_HALF_HEIGHT]],
    [[NECK_RADIUS, COLLAR_HALF_HEIGHT], [NECK_RADIUS, UPPER_JOIN]],
])
ARC_LIMITS = (
    (LOWER_Z, LOWER_RADIUS, -np.pi / 2, np.arccos(NECK_RADIUS / LOWER_RADIUS)),
    (UPPER_Z, UPPER_RADIUS, -np.arccos(NECK_RADIUS / UPPER_RADIUS), np.pi / 2),
)


def exact_sdf_xyz(x, y, z):
    rho = np.hypot(x, y)
    result = np.full(np.broadcast_shapes(rho.shape, np.shape(z)), np.inf)
    for center, radius, start, end in ARC_LIMITS:
        angle = np.clip(np.arctan2(z - center, rho), start, end)
        distance = np.hypot(rho - radius * np.cos(angle), z - center - radius * np.sin(angle))
        np.minimum(result, distance, out=result)
    for a, b in PROFILE_SEGMENTS:
        direction = b - a
        t = np.clip(
            ((rho - a[0]) * direction[0] + (z - a[1]) * direction[1])
            / np.dot(direction, direction), 0, 1,
        )
        distance = np.hypot(rho - a[0] - t * direction[0], z - a[1] - t * direction[1])
        np.minimum(result, distance, out=result)
    inside = (
        (rho ** 2 + (z - LOWER_Z) ** 2 < LOWER_RADIUS ** 2)
        | (rho ** 2 + (z - UPPER_Z) ** 2 < UPPER_RADIUS ** 2)
        | ((rho < NECK_RADIUS) & (z >= LOWER_JOIN) & (z <= UPPER_JOIN))
        | ((rho < COLLAR_RADIUS) & (np.abs(z) < COLLAR_HALF_HEIGHT))
    )
    return np.where(inside, -result, result)


def exact_sdf(points):
    points = np.asarray(points, dtype=float)
    return exact_sdf_xyz(points[..., 0], points[..., 1], points[..., 2])


def validate_reference():
    profile = []
    for center, radius, low, high in ARC_LIMITS:
        angle = np.linspace(low, high, 701)
        profile.append(np.column_stack((radius * np.cos(angle), center + radius * np.sin(angle))))
    for a, b in PROFILE_SEGMENTS:
        t = np.linspace(0, 1, 101)[:, None]
        profile.append((1 - t) * a + t * b)
    profile = np.concatenate(profile)
    angles = np.arange(len(profile)) * np.pi * (3 - np.sqrt(5))
    surface = np.column_stack((
        profile[:, 0] * np.cos(angles),
        profile[:, 0] * np.sin(angles),
        profile[:, 1],
    ))
    assert np.max(np.abs(exact_sdf(surface))) < 2e-15
    probes = np.array([
        [0, 0, LOWER_Z], [0, 0, UPPER_Z], [0, 0, 0.12],
        [0.26, 0, 0], [0.40, 0, 0],
    ])
    assert np.allclose(exact_sdf(probes), [-LOWER_RADIUS, -UPPER_RADIUS, -NECK_RADIUS, -0.035, 0.06])
    points = np.random.default_rng(SEED).uniform(*DOMAIN, size=(1000, 3))
    epsilon = 1e-6
    gradient = np.column_stack([
        (exact_sdf(points + np.eye(3)[i] * epsilon)
         - exact_sdf(points - np.eye(3)[i] * epsilon)) / (2 * epsilon)
        for i in range(3)
    ])
    assert np.max(np.abs(np.linalg.norm(gradient, axis=1) - 1)) < 2e-4


def sample_field(resolution):
    coordinates = np.linspace(*DOMAIN, resolution)
    x, y, z = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij", sparse=True)
    return exact_sdf_xyz(x, y, z).astype(np.float32)


def extract_mesh(field, spacing):
    return marching_cubes(
        field, level=0, spacing=(spacing,) * 3, gradient_direction="ascent",
        step_size=1, allow_degenerate=False, method="lewiner",
    )


def build_mesh(resolution):
    spacing = (DOMAIN[1] - DOMAIN[0]) / (resolution - 1)
    field = sample_field(resolution)
    assert field.shape == (resolution,) * 3
    for dimension in range(3):
        assert np.all(np.take(field, [0, resolution - 1], axis=dimension) > 0)
    vertices, faces, _, _ = extract_mesh(field, spacing)
    vertices = vertices.astype(float) + DOMAIN[0]
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if mesh.volume < 0:
        faces = faces[:, ::-1].copy()
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert mesh.body_count >= 1 and mesh.euler_number == 2 * mesh.body_count
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    twice_area = np.linalg.norm(normals, axis=1)
    assert np.all(twice_area > 1e-12)
    normals /= twice_area[:, None]
    row = {
        "samples_per_axis": resolution,
        "vertices": len(vertices),
        "triangles": len(faces),
        "components": mesh.body_count,
    }
    print(f"{resolution}^3: V={len(vertices)}, F={len(faces)}, components={mesh.body_count}", flush=True)
    return {"row": row, "triangles": triangles, "normals": normals}


def camera_basis():
    azimuth, elevation, roll = np.deg2rad([-58, 13, -28])
    direction = np.array([
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation),
    ])
    right = np.array([-np.sin(azimuth), np.cos(azimuth), 0])
    up = np.cross(direction, right)
    right, up = np.cos(roll) * right + np.sin(roll) * up, -np.sin(roll) * right + np.cos(roll) * up
    basis = np.stack((right, up, direction))
    assert np.allclose(basis @ basis.T, np.eye(3))
    return basis


def configure_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "text.color": INK,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "svg.fonttype": "path",
        "svg.hashsalt": "lab6-marching-cubes-resolution",
    })


def comparison(results):
    fig = plt.figure(figsize=(16, 5.5))
    basis = camera_basis()
    light = np.array([-0.35, -0.55, 0.83])
    light /= np.linalg.norm(light)
    labels = []
    axes = []
    gap = 0.016
    panel_width = (0.964 - (len(results) - 1) * gap) / len(results)
    for index, result in enumerate(results):
        left = 0.018 + index * (panel_width + gap)
        ax = fig.add_axes([left, 0.24, panel_width, 0.73])
        axes.append(ax)
        projected = result["triangles"] @ basis.T
        visible = result["normals"] @ basis[2] > 0
        order = np.where(visible)[0]
        order = order[np.argsort(projected[order, :, 2].mean(axis=1), kind="stable")]
        shade = 0.80 + 0.20 * np.maximum(result["normals"] @ light, 0)
        colors = np.array(to_rgb(TEAL))[None, :] * shade[:, None]
        surface = PolyCollection(
            projected[order, :, :2], facecolors=colors[order],
            edgecolors=EDGE, linewidths=0.24, antialiaseds=True,
        )
        ax.add_collection(surface)
        ax.set(xlim=(-1.06, 1.06), ylim=(-1.36, 1.36), aspect="equal")
        ax.axis("off")
        row = result["row"]
        resolution = row["samples_per_axis"]
        count = f"{row['triangles']:,}".replace(",", " ")
        labels.extend((
            fig.text(left + panel_width / 2, 0.163, "$" + str(resolution) + r"^3$ próbek",
                     ha="center", va="center", fontsize=23),
            fig.text(left + panel_width / 2, 0.087, count + " trójkątów",
                     ha="center", va="center", fontsize=18, color=SECONDARY),
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
    svg = OUTPUT / "marching_cubes_resolution_wireframe.svg"
    fig.savefig(svg, facecolor=BG, metadata={
        "Creator": "lab_6 / NumPy / scikit-image / Matplotlib",
        "Description": "Exact SDF; same domain, camera and physical scale. Raw Lewiner Marching Cubes meshes.",
    })
    assert svg.stat().st_size < 12 * 1024 * 1024
    print(svg.relative_to(ROOT), flush=True)
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    configure_style()
    validate_reference()
    results = [build_mesh(resolution) for resolution in RESOLUTIONS]
    comparison(results)
    print("Exact SDF, topology, identical view/scale and layout: OK")


if __name__ == "__main__":
    main()
