"""Geometria kamery i wspólna oprawa demonstracji rekonstrukcji."""

from base64 import b64encode

import numpy as np
import open3d as o3d
import plotly.graph_objects as go
from plotly.subplots import make_subplots


BG, INK = "#F8FAFC", "#0F172A"
BLUE, ORANGE, TEAL = "#2563EB", "#EA580C", "#0F766E"
GRID = "#CBD5E1"


def pixel_directions(intrinsic):
    """Promienie (x/z, y/z, 1) przez środki pikseli w układzie kamery."""
    fx, fy = intrinsic.get_focal_length()
    cx, cy = intrinsic.get_principal_point()
    u, v = np.meshgrid(np.arange(intrinsic.width), np.arange(intrinsic.height))
    return np.stack(((u - cx) / fx, (v - cy) / fy, np.ones_like(u)), axis=-1)


def world_to_camera(eye, target=(0, 0, 0)):
    """Macierz świata do kamery z osiami w prawo, w dół i do przodu."""
    eye = np.asarray(eye, dtype=float)
    forward = np.asarray(target) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack((right, down, forward))
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rotation
    extrinsic[:3, 3] = -rotation @ eye
    return extrinsic


def render_depth(scene, intrinsic, extrinsic):
    """Zwraca osiowe z w metrach i zero dla promieni bez trafienia."""
    rotation, translation = extrinsic[:3, :3], extrinsic[:3, 3]
    eye = -rotation.T @ translation
    directions = pixel_directions(intrinsic) @ rotation
    origins = np.broadcast_to(eye, directions.shape)
    rays = np.concatenate((origins, directions), axis=-1).astype(np.float32)
    hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    # Kierunek w układzie kamery ma składową z = 1, więc t_hit jest głębokością osiową.
    return np.where(np.isfinite(hit), hit, 0)


def point_trace(points):
    x, y, z = np.asarray(points, dtype=np.float32).T
    return go.Scatter3d(
        x=x, y=y, z=z, mode="markers", marker=dict(size=1.6, color=BLUE),
        hoverinfo="skip", showlegend=False,
    )


def mesh_trace(vertices, faces, flat=True):
    x, y, z = np.asarray(vertices, dtype=np.float32).T
    i, j, k = np.asarray(faces, dtype=np.min_scalar_type(len(vertices))).T
    return go.Mesh3d(
        x=x, y=y, z=z, i=i, j=j, k=k, color=TEAL,
        flatshading=flat, hoverinfo="skip", showlegend=False,
        lighting=dict(ambient=0.63, diffuse=0.65, specular=0, roughness=1),
    )


def mesh_index_update(faces):
    """Zapisuje tablice indeksów dla Plotly.restyle bez rozwijania ich do list liczb."""
    return {
        axis: [dict(dtype="u4", bdata=b64encode(indices.astype("<u4").tobytes()).decode("ascii"))]
        for axis, indices in zip("ijk", faces.T)
    }


def style_figure(figure, height):
    figure.update_layout(
        height=height, margin=dict(l=12, r=12, t=36, b=12),
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family="DejaVu Sans, sans-serif", size=14, color=INK),
    )


def scene_layout(bounds, eye=(1.10, -1.45, 1.18), up=(0, 0, 1)):
    return dict(
        aspectmode="data", bgcolor=BG, uirevision="camera",
        camera=dict(eye=dict(zip("xyz", eye)), up=dict(zip("xyz", up)),
                    projection=dict(type="orthographic")),
        **{f"{axis}axis": dict(title=axis, range=limits, backgroundcolor=BG,
                              gridcolor=GRID, showticklabels=False, zeroline=False)
           for axis, limits in zip("xyz", bounds)},
    )


def surface_panels(panels, columns, bounds, height):
    """Panele o tym samym widoku początkowym i skali."""
    rows = len(panels) // columns
    figure = make_subplots(
        rows=rows, cols=columns, specs=[[{"type": "scene"}] * columns for _ in range(rows)],
        subplot_titles=[title for title, _ in panels],
        horizontal_spacing=0.03, vertical_spacing=0.09,
    )
    for index, (_, trace) in enumerate(panels):
        row, column = divmod(index, columns)
        figure.add_trace(trace, row=row + 1, col=column + 1)
    figure.update_scenes(**scene_layout(bounds))
    style_figure(figure, height)
    return figure
