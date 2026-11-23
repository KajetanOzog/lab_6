from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.patches import Circle


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "images" / "sdf"
BG = "#F8FAFC"
INK = "#0F172A"
BLUE = "#2563EB"
TEAL = "#0F766E"
ORANGE = "#EA580C"
GRAY = "#94A3B8"
SECONDARY = "#64748B"
SDF_CMAP = LinearSegmentedColormap.from_list("sdf", [BLUE, BG, ORANGE], N=257)


def configure_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 16,
        "mathtext.fontset": "dejavusans",
        "text.color": INK,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.dpi": 300,
    })


def camera_basis():
    az, el = np.deg2rad([-55, 24])
    toward_camera = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = np.array([-np.sin(az), np.cos(az), 0.0])
    up = np.cross(toward_camera, right)
    return np.stack([right, up, toward_camera])


def sphere_points(count=720):
    index = np.arange(count)
    z = 1 - 2 * (index + 0.5) / count
    angle = index * np.pi * (3 - np.sqrt(5))
    r = np.sqrt(1 - z * z)
    return np.column_stack([r * np.cos(angle), r * np.sin(angle), z])


def axes_for_panel(fig, left):
    ax = fig.add_axes([left, 0.285, 0.275, 0.65])
    ax.set(xlim=(-1.4, 1.4), ylim=(-1.4, 1.4), aspect="equal")
    ax.axis("off")
    return ax


def draw_cloud(ax, points, basis):
    projected = points @ basis.T
    back = projected[:, 2] < 0
    ax.scatter(projected[back, 0], projected[back, 1], s=6.0, color=BLUE,
               alpha=0.24, edgecolors="none")
    ax.scatter(projected[~back, 0], projected[~back, 1], s=10.5, color=BLUE,
               alpha=0.95, edgecolors="none")


def draw_sdf(ax):
    coord = np.linspace(-1.4, 1.4, 401)
    x, y = np.meshgrid(coord, coord)
    f = np.hypot(x, y) - 1
    ax.contourf(x, y, f, levels=np.linspace(-1, 1, 65),
                cmap=SDF_CMAP, norm=Normalize(-1, 1), antialiased=False)
    ax.add_patch(Circle((0, 0), 1, facecolor="none", edgecolor=INK, linewidth=2.4))
    ax.contour(x, y, f, levels=[-0.5, 0.5], colors=SECONDARY,
               linewidths=0.65, alpha=0.65, linestyles="solid")
    label_box = dict(facecolor=BG, edgecolor="none", pad=2.5, alpha=0.95)
    ax.text(0, 0, "$f<0$", ha="center", va="center", fontsize=20, color=BLUE, bbox=label_box)
    ax.annotate("$f=0$", xy=(0.60, 0.80), xytext=(0.97, 1.15),
                ha="center", va="center", fontsize=16, color=INK, bbox=label_box,
                arrowprops=dict(arrowstyle="-", color=INK, linewidth=1))
    ax.text(-0.96, -1.17, "$f>0$", ha="center", va="center", fontsize=16, color=ORANGE, bbox=label_box)


def draw_mesh(ax, mesh, basis):
    vertices = np.asarray(mesh.vertices) @ basis.T
    normals = np.asarray(mesh.face_normals)
    visible = normals @ basis[2] > 0
    faces = np.asarray(mesh.faces)[visible]
    order = np.argsort(vertices[faces, 2].mean(axis=1))
    light = np.array([-0.3, -0.5, 0.8])
    light /= np.linalg.norm(light)
    tones = np.digitize(normals[visible] @ light, [-0.35, 0.15, 0.65])
    teal, background = np.array(to_rgb(TEAL)), np.array(to_rgb(BG))
    palette = np.array([(1 - a) * teal + a * background for a in [0.02, 0.12, 0.25, 0.38]])
    ax.add_collection(PolyCollection(
        vertices[faces[order], :2], facecolors=palette[tones[order]],
        edgecolors=INK, linewidths=0.45, antialiaseds=True,
    ))


def main():
    configure_style()
    points = sphere_points()
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    basis = camera_basis()
    fig = plt.figure(figsize=(14.4, 6.1))
    lefts = [0.035, 0.3625, 0.69]
    axes = [axes_for_panel(fig, left) for left in lefts]
    draw_cloud(axes[0], points, basis)
    draw_sdf(axes[1])
    draw_mesh(axes[2], mesh, basis)
    for left, label in zip(lefts, ["Point Cloud", "SDF", "Mesh"]):
        fig.text(left + 0.1375, 0.09, label, ha="center", va="center", fontsize=27)
    axes[1].text(0.02, 0.96, "$z=0$", transform=axes[1].transAxes,
                 ha="left", va="top", fontsize=13, color=SECONDARY)
    cax = fig.add_axes([0.3975, 0.213, 0.205, 0.023])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-1, 1), cmap=SDF_CMAP),
                       cax=cax, orientation="horizontal", ticks=[-1, 0, 1])
    bar.set_ticklabels(["−1", "0", "+1"])
    bar.ax.tick_params(labelsize=13, length=3, color=GRAY, labelcolor=SECONDARY)
    bar.outline.set_visible(False)
    cax.axvline(0, color=INK, linewidth=1.6)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "representations_3d.png"
    fig.savefig(path, dpi=300, facecolor=BG,
                metadata={"Creator": "lab_6 / Python / NumPy / Matplotlib / Trimesh"})
    print(path.relative_to(ROOT))
    plt.close(fig)


if __name__ == "__main__":
    main()
