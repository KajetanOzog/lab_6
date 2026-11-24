from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgb
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "images" / "point_clouds"
BG = "#F8FAFC"
INK = "#0F172A"
BLUE = "#2563EB"
TEAL = "#0F766E"
SEED = 1810
MAJOR_RADIUS = 1.0
MINOR_RADIUS = 0.36
THETA_SAMPLES = 128
PHI_SAMPLES = 64
GAP_THETA = np.deg2rad(-58)
GAP_PHI = np.deg2rad(58)
GAP_HALF_WIDTH = 0.36
GAP_HALF_HEIGHT = 0.23
RADIUS_MULTIPLIERS = np.array([1.4, 2.0, 2.6])
POISSON_DEPTH = 8
RECONSTRUCTION_DOCUMENTATION = "https://www.open3d.org/docs/latest/tutorial/geometry/surface_reconstruction.html"


def wrapped_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def torus_sdf(points):
    points = np.asarray(points)
    return np.hypot(np.hypot(points[:, 0], points[:, 1]) - MAJOR_RADIUS, points[:, 2]) - MINOR_RADIUS


def reference_normals(points):
    radial = np.hypot(points[:, 0], points[:, 1])
    centerline = np.column_stack((
        MAJOR_RADIUS * points[:, 0] / radial,
        MAJOR_RADIUS * points[:, 1] / radial,
        np.zeros(len(points)),
    ))
    normal = points - centerline
    return normal / np.linalg.norm(normal, axis=1)[:, None]


def generate_points():
    rng = np.random.default_rng(SEED)
    i, j = np.meshgrid(np.arange(THETA_SAMPLES), np.arange(PHI_SAMPLES), indexing="ij")
    theta = (i.ravel() + 0.5 + rng.uniform(-0.25, 0.25, i.size)) * (2 * np.pi / THETA_SAMPLES)
    fraction = (j.ravel() + 0.5 + rng.uniform(-0.25, 0.25, j.size)) / PHI_SAMPLES
    target = 2 * np.pi * fraction
    phi = target.copy()
    for _ in range(9):
        phi -= (
            phi + (MINOR_RADIUS / MAJOR_RADIUS) * np.sin(phi) - target
        ) / (1 + (MINOR_RADIUS / MAJOR_RADIUS) * np.cos(phi))
    assert np.max(np.abs(phi + (MINOR_RADIUS / MAJOR_RADIUS) * np.sin(phi) - target)) < 1e-12
    radial = MAJOR_RADIUS + MINOR_RADIUS * np.cos(phi)
    points = np.column_stack((
        radial * np.cos(theta), radial * np.sin(theta), MINOR_RADIUS * np.sin(phi),
    ))
    theta_distance = (MAJOR_RADIUS + MINOR_RADIUS * np.cos(GAP_PHI)) * wrapped_angle(theta - GAP_THETA)
    phi_distance = MINOR_RADIUS * wrapped_angle(phi - GAP_PHI)
    gap = (theta_distance / GAP_HALF_WIDTH) ** 2 + (phi_distance / GAP_HALF_HEIGHT) ** 2 < 1
    retained = points[~gap]
    assert np.max(np.abs(torus_sdf(retained))) < 1e-14
    print(f"Analytic torus: {len(points)} samples; removed {np.count_nonzero(gap)}; retained {len(retained)}", flush=True)
    return retained


def oriented_cloud(points):
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=7, workers=1)
    spacing = float(np.median(distances[:, 6]))
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.6 * spacing, max_nn=40),
    )
    cloud.orient_normals_consistent_tangent_plane(k=35)
    normals = np.asarray(cloud.normals).copy()
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    if np.mean(np.einsum("ij,ij->i", points, normals)) < 0:
        normals = -normals
    cloud.normals = o3d.utility.Vector3dVector(normals)
    agreement = np.einsum("ij,ij->i", normals, reference_normals(points))
    assert np.min(agreement) > 0.90
    assert np.quantile(agreement, 0.01) > 0.97
    assert np.allclose(np.linalg.norm(normals, axis=1), 1)
    print(
        f"Estimated normals, consistent tangent plane orientation; "
        f"min cosine to analytic normals={agreement.min():.5f}; spacing(k=6)={spacing:.5f}",
        flush=True,
    )
    return cloud, spacing


def mesh_arrays(mesh, name):
    vertices = np.asarray(mesh.vertices).copy()
    faces = np.asarray(mesh.triangles).copy()
    assert len(faces) > 0 and np.all(np.isfinite(vertices))
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    length = np.linalg.norm(normals, axis=1)
    assert np.all(length > 1e-12)
    normals /= length[:, None]
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    unique_edges, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    boundary = int(np.count_nonzero(counts == 1))
    assert np.all(counts <= 2)
    boundary_vertices, boundary_degrees = np.unique(unique_edges[counts == 1], return_counts=True)
    assert len(boundary_vertices) == 0 or np.all(boundary_degrees == 2)
    print(
        f"{name}: V={len(vertices)}, F={len(faces)}, boundary edges={boundary}, "
        f"Euler={len(np.unique(faces)) - len(unique_edges) + len(faces)}",
        flush=True,
    )
    return {
        "name": name, "vertices": vertices, "faces": faces,
        "triangles": triangles, "normals": normals, "boundary_edges": boundary,
    }


def reconstruct(cloud, spacing):
    radii = spacing * RADIUS_MULTIPLIERS
    print("Ball Pivoting radii:", np.round(radii, 5), flush=True)
    bpa = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        cloud, o3d.utility.DoubleVector(radii),
    )
    poisson, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud, depth=POISSON_DEPTH, scale=1.1, linear_fit=True, n_threads=1,
    )
    assert len(densities) == len(poisson.vertices)
    bpa_result = mesh_arrays(bpa, "Ball Pivoting")
    poisson_result = mesh_arrays(poisson, "Poisson")
    used = len(np.unique(bpa_result["faces"]))
    coverage = used / len(cloud.points)
    assert coverage > 0.99
    assert np.allclose(bpa_result["vertices"], np.asarray(cloud.points))
    print(f"BPA input-vertex coverage: {coverage:.2%}", flush=True)
    print("Poisson shown without density trimming, cropping, smoothing or hole filling", flush=True)
    return bpa_result, poisson_result, radii


def camera_basis():
    azimuth, elevation = np.deg2rad([-58, 32])
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


def render(cloud, meshes, radii):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "text.color": INK,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "svg.fonttype": "path",
        "svg.hashsalt": "lab6-ball-pivoting-vs-poisson",
    })
    basis = camera_basis()
    light = basis[2] + np.array([0, 0, 0.40])
    light /= np.linalg.norm(light)
    points = np.asarray(cloud.points)
    cloud_projection = points @ basis.T
    mesh_projections = [result["triangles"] @ basis.T for result in meshes]
    all_projected = np.concatenate((
        cloud_projection,
        *(result["vertices"] @ basis.T for result in meshes),
    ))
    horizontal = 1.12 * np.max(np.abs(all_projected[:, 0]))
    vertical = 1.17 * np.max(np.abs(all_projected[:, 1]))
    fig = plt.figure(figsize=(15, 5.3))
    axes, labels = [], []
    for left, title in zip((0.02, 0.35, 0.68), ("Chmura punktów", "Ball Pivoting", "Poisson")):
        ax = fig.add_axes([left, 0.22, 0.30, 0.74])
        axes.append(ax)
        ax.set(xlim=(-horizontal, horizontal), ylim=(-vertical, vertical), aspect="equal")
        ax.axis("off")
        labels.append(fig.text(left + 0.15, 0.105, title, fontsize=23,
                               ha="center", va="center"))
    normals = np.asarray(cloud.normals)
    point_shade = 0.72 + 0.28 * np.maximum(normals @ light, 0)
    point_colors = np.array(to_rgb(BLUE))[None, :] * point_shade[:, None]
    back_facing = normals @ basis[2] < 0
    point_colors[back_facing] = (
        0.22 * point_colors[back_facing] + 0.78 * np.array(to_rgb(BG))
    )
    order = np.argsort(cloud_projection[:, 2], kind="stable")
    axes[0].scatter(
        cloud_projection[order, 0], cloud_projection[order, 1], s=2.1,
        c=point_colors[order],
        edgecolors="none", linewidths=0,
    )
    for ax, result, projected in zip(axes[1:], meshes, mesh_projections):
        order = np.argsort(projected[:, :, 2].mean(axis=1), kind="stable")
        shade = 0.72 + 0.28 * np.maximum(result["normals"] @ light, 0)
        colors = np.array(to_rgb(TEAL))[None, :] * shade[:, None]
        ax.add_collection(PolyCollection(
            projected[order, :, :2], facecolors=colors[order],
            edgecolors="face", linewidths=0.60, joinstyle="round", antialiaseds=False,
        ))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [label.get_window_extent(renderer) for label in labels]
    for i, box in enumerate(boxes):
        assert fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1)
        assert all(not box.overlaps(other) for other in boxes[i + 1:])
    scales = [
        (np.linalg.norm(ax.transData.transform((1, 0)) - ax.transData.transform((0, 0))),
         np.linalg.norm(ax.transData.transform((0, 1)) - ax.transData.transform((0, 0))))
        for ax in axes
    ]
    assert np.allclose(scales, scales[0][0])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "ball_pivoting_vs_poisson.svg"
    fig.savefig(path, facecolor=BG, metadata={
        "Creator": "lab_6 / NumPy / Open3D / Matplotlib",
        "Description": (
            f"Same observed analytic torus samples and estimated oriented normals for both methods. "
            f"R={MAJOR_RADIUS}, r={MINOR_RADIUS}. Local elliptical gap at theta={GAP_THETA}, "
            f"phi={GAP_PHI}, surface-parameter half-width={GAP_HALF_WIDTH}, half-height={GAP_HALF_HEIGHT}. "
            f"BPA radii={radii.tolist()}; Poisson depth={POISSON_DEPTH}, scale=1.1, "
            f"linear_fit=True. Raw algorithm output: no density trimming, no cropping, no smoothing, "
            f"no manual hole filling. Same orthographic camera, scale and directional light."
            f" All input samples are shown; back-facing samples use faded blue for depth context."
        ),
    })
    plt.close(fig)
    print(path.relative_to(ROOT), flush=True)


def main():
    o3d.utility.random.seed(SEED)
    points = generate_points()
    cloud, spacing = oriented_cloud(points)
    bpa, poisson, radii = reconstruct(cloud, spacing)
    render(cloud, (bpa, poisson), radii)
    print("Oriented input, actual reconstructions, full output and matched view/lighting: OK")


if __name__ == "__main__":
    main()
