"""Reprodukcja wag do sekcji NeRF. Wymaga PyTorch oraz Open3D."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from torch import nn
from torch.nn import functional as F

from scripts.utils.surface_reconstruction import pixel_directions, world_to_camera


def make_training_scene():
    # Ta sama geometria co torus na początku notebooka 02.
    mesh = o3d.geometry.TriangleMesh.create_torus(
        torus_radius=0.89, tube_radius=0.28, radial_resolution=96, tubular_resolution=40
    )
    vertices = np.asarray(mesh.vertices)
    x, y, z = vertices.T.copy()
    angle = np.arctan2(y, x)
    tube_scale = 1 + 0.18 * np.cos(3 * angle + 0.4)
    radius = 0.89 + (np.hypot(x, y) - 0.89) * tube_scale
    vertices[:] = np.column_stack((radius*np.cos(angle), radius*np.sin(angle),
                                   z*tube_scale + 0.14*np.sin(3*angle)))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def training_rays(device, azimuth_offset=0.0):
    scene = make_training_scene()
    intrinsic = o3d.camera.PinholeCameraIntrinsic(64, 64, 77, 77, 31.5, 31.5)
    directions = pixel_directions(intrinsic).reshape(-1, 3)
    all_rays, all_rgb = [], []
    for elevation in np.deg2rad([-64, -40, -17, 17, 40, 64]):
        for azimuth in np.linspace(0, 2*np.pi, 12, endpoint=False) + azimuth_offset:
            eye = 3.45*np.array([np.cos(elevation)*np.cos(azimuth),
                                 np.cos(elevation)*np.sin(azimuth), np.sin(elevation)])
            world_dirs = directions @ world_to_camera(eye)[:3, :3]
            world_dirs /= np.linalg.norm(world_dirs, axis=1, keepdims=True)
            rays = np.column_stack((np.broadcast_to(eye, world_dirs.shape), world_dirs)).astype(np.float32)
            distance = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
            visible = np.isfinite(distance)
            point = rays[visible, :3] + distance[visible, None]*rays[visible, 3:]
            rgb = np.full((len(rays), 3), .96, dtype=np.float32)
            rgb[visible] = np.array([.38, .49, .61]) + point*np.array([.16, .16, .27])
            all_rays.append(rays)
            all_rgb.append(rgb)
    return (torch.as_tensor(np.concatenate(all_rays), device=device),
            torch.as_tensor(np.concatenate(all_rgb), device=device))


class RadianceField(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("frequencies", torch.tensor([1, 2, 4])*torch.pi)
        self.layers = nn.Sequential(nn.Linear(21, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                                    nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 4))

    def forward(self, xyz):
        angles = xyz[..., None]*self.frequencies
        features = torch.cat((xyz, angles.sin().flatten(-2), angles.cos().flatten(-2)), dim=-1)
        raw = self.layers(features)
        return F.softplus(raw[..., 0] - 2), torch.sigmoid(raw[..., 1:])


def render(model, rays, jitter):
    count = 48
    step = 3.0/count
    device = rays.device
    offsets = torch.rand((len(rays), count), device=device) if jitter else 0.5
    depths = 1.95 + step*(torch.arange(count, device=device) + offsets)
    points = rays[:, None, :3] + depths[..., None]*rays[:, None, 3:]
    sigma, rgb = model(points)
    survival = torch.exp(-sigma*step)
    transparency = torch.cumprod(torch.cat((torch.ones_like(survival[:, :1]), survival), dim=-1), dim=-1)
    weights = transparency[:, :-1]*(1-survival)
    return (weights[..., None]*rgb).sum(1) + transparency[:, -1:]*.96


def main():
    parser = ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--output", type=Path, default=Path("data/torus_nerf_weights.npz"))
    args = parser.parse_args()
    torch.manual_seed(17)
    torch.set_num_threads(3)
    rays, colors = training_rays(args.device)
    model = RadianceField().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    for step in range(args.steps):
        ids = torch.randint(len(rays), (448,), device=args.device)
        loss = F.mse_loss(render(model, rays[ids], True), colors[ids])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step+1) % 500 == 0:
            print(step+1, float(loss.detach()), flush=True)
    with torch.no_grad():
        test_rays, test_colors = training_rays(args.device, azimuth_offset=np.pi/12)
        predicted = torch.cat([render(model, batch, False) for batch in test_rays[::35].split(512)])
        error = float(F.mse_loss(predicted, test_colors[::35]))
        print("RGB MSE for held-out views:", error, flush=True)
    arrays = {}
    for i, layer in enumerate(model.layers[::2]):
        arrays[f"w{i}"] = layer.weight.detach().cpu().numpy()
        arrays[f"b{i}"] = layer.bias.detach().cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)


if __name__ == "__main__":
    main()
