from pathlib import Path

import contourpy
import matplotlib.font_manager
import numpy as np
from PIL import GifImagePlugin, Image, ImageChops, ImageDraw, ImageFont

from scripts.utils.marching_squares import cell_segments


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "images" / "marching_cubes"
BG = "#F8FAFC"
INK = "#0F172A"
BLUE = "#2563EB"
ORANGE = "#EA580C"
TEAL = "#0F766E"
GRID = "#CBD5E1"
GRAY = "#94A3B8"
HIGHLIGHT = "#E2E8F0"
LIMIT = 1.6
SIZE = 960
SCALE = 2
MARGIN = 84
START_MS = 800
END_MS = 3500


def boundary():
    theta = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    radius = 1 + 0.26 * np.cos(3 * theta) + 0.10 * np.sin(2 * theta)
    return np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))


def signed_distance(points, polygon):
    points = np.asarray(points, dtype=float)
    shape = points.shape[:-1]
    flat = points.reshape(-1, 2)
    start = polygon
    end = np.roll(polygon, -1, axis=0)
    edges = end - start
    lengths2 = np.einsum("ij,ij->i", edges, edges)
    distances = []
    for chunk_start in range(0, len(flat), 512):
        chunk = flat[chunk_start:chunk_start + 512]
        delta = chunk[:, None, :] - start[None, :, :]
        projection = np.clip(np.einsum("nki,ki->nk", delta, edges) / lengths2, 0, 1)
        residual = delta - projection[:, :, None] * edges
        distance = np.sqrt(np.min(np.einsum("nki,nki->nk", residual, residual), axis=1))
        px, py = chunk[:, 0, None], chunk[:, 1, None]
        ax, ay = start[:, 0], start[:, 1]
        bx, by = end[:, 0], end[:, 1]
        crossing = (ay > py) != (by > py)
        divisor = np.where(by != ay, by - ay, 1)
        intersection_x = ax + (py - ay) * (bx - ax) / divisor
        inside = np.count_nonzero(crossing & (px < intersection_x), axis=1) % 2 == 1
        distances.append(np.where(inside, -distance, distance))
    return np.concatenate(distances).reshape(shape)


def segment_key(segment):
    return tuple(sorted(tuple(np.round(point, 10)) for point in segment))


def sample_grid(resolution, polygon):
    coordinates = np.linspace(-LIMIT, LIMIT, resolution + 1)
    x, y = np.meshgrid(coordinates, coordinates)
    values = signed_distance(np.stack((x, y), axis=-1), polygon)
    assert np.min(np.abs(values)) > 1e-8
    assert np.all(values[[0, -1], :] > 0) and np.all(values[:, [0, -1]] > 0)
    cells, segments = [], []
    for row in range(resolution - 1, -1, -1):
        for column in range(resolution):
            corners = np.array([
                [coordinates[column], coordinates[row]],
                [coordinates[column + 1], coordinates[row]],
                [coordinates[column + 1], coordinates[row + 1]],
                [coordinates[column], coordinates[row + 1]],
            ])
            corner_values = values[
                [row, row, row + 1, row + 1],
                [column, column + 1, column + 1, column],
            ]
            local = cell_segments(corners, corner_values)
            cells.append((row, column, local))
            segments.extend(local)
    reference = contourpy.contour_generator(
        x=coordinates, y=coordinates, z=values, name="serial", corner_mask=False
    ).lines(0)
    assert len(reference) == 1 and np.allclose(reference[0][0], reference[0][-1])
    expected = {
        segment_key(np.array([a, b]))
        for line in reference for a, b in zip(line[:-1], line[1:])
    }
    assert {segment_key(segment) for segment in segments} == expected
    degrees = {}
    for segment in segments:
        for point in segment_key(segment):
            degrees[point] = degrees.get(point, 0) + 1
    assert all(degree == 2 for degree in degrees.values())
    fractions = np.linspace(0, 1, 13)
    probes = np.concatenate([
        (1 - fractions[:, None]) * segment[0] + fractions[:, None] * segment[1]
        for segment in segments
    ])
    error = float(np.max(np.abs(signed_distance(probes, polygon))))
    return {
        "resolution": resolution, "coordinates": coordinates, "x": x, "y": y,
        "values": values, "cells": cells, "segments": segments, "error": error,
    }


def screen(points):
    points = np.asarray(points, dtype=float)
    return np.stack((
        MARGIN + (points[..., 0] + LIMIT) / (2 * LIMIT) * (SIZE - 2 * MARGIN),
        SIZE - MARGIN - (points[..., 1] + LIMIT) / (2 * LIMIT) * (SIZE - 2 * MARGIN),
    ), axis=-1) * SCALE


def make_palette():
    background = np.array(Image.new("RGB", (1, 1), BG).getpixel((0, 0)))
    colors = [tuple(background)]
    for color in [INK, BLUE, ORANGE, TEAL, GRID, GRAY, HIGHLIGHT]:
        rgb = np.array(Image.new("RGB", (1, 1), color).getpixel((0, 0)))
        for amount in np.linspace(1 / 32, 1, 32):
            colors.append(tuple(np.rint((1 - amount) * background + amount * rgb).astype(int)))
    colors.extend([tuple(background)] * (256 - len(colors)))
    palette = Image.new("P", (1, 1))
    palette.putpalette([int(channel) for color in colors for channel in color])
    return palette


def static_canvas(data):
    canvas = Image.new("RGB", (SIZE * SCALE, SIZE * SCALE), BG)
    draw = ImageDraw.Draw(canvas)
    coordinates = data["coordinates"]
    for value in coordinates:
        for endpoints in [
            [[value, -LIMIT], [value, LIMIT]],
            [[-LIMIT, value], [LIMIT, value]],
        ]:
            draw.line([tuple(point) for point in screen(endpoints)], fill=GRID, width=2 * SCALE)
    radius = 6.2 * SCALE
    for point, value in zip(
        screen(np.stack((data["x"], data["y"]), axis=-1)).reshape(-1, 2),
        data["values"].ravel(),
    ):
        px, py = point
        draw.ellipse((px - radius, py - radius, px + radius, py + radius),
                     fill=BLUE if value < 0 else ORANGE)
    font_path = matplotlib.font_manager.findfont("DejaVu Sans")
    font = ImageFont.truetype(font_path, 28 * SCALE)
    resolution = data["resolution"]
    draw.text((MARGIN * SCALE, 24 * SCALE), f"{resolution} × {resolution}",
              font=font, fill=INK)
    return canvas


def render_frame(base, data, segments, palette, active=None):
    canvas = base.copy()
    draw = ImageDraw.Draw(canvas)
    if active is not None:
        row, column = active
        coordinates = data["coordinates"]
        bounds = screen([
            [coordinates[column], coordinates[row + 1]],
            [coordinates[column + 1], coordinates[row]],
        ]).ravel()
        inset = 2.5 * SCALE
        bounds += np.array([inset, inset, -inset, -inset])
        draw.rectangle(tuple(bounds), fill=HIGHLIGHT, outline=GRAY, width=2 * SCALE)
    width = 6 * SCALE
    radius = width / 2
    for segment in segments:
        points = screen(segment)
        draw.line([tuple(point) for point in points], fill=TEAL, width=width)
        for px, py in points:
            draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=TEAL)
    return canvas.resize((SIZE, SIZE), Image.Resampling.LANCZOS).quantize(
        palette=palette, dither=Image.Dither.NONE
    )


def write_animation(data, stem):
    palette = make_palette()
    base = static_canvas(data)
    previous = render_frame(base, data, [], palette)
    path = OUTPUT / (stem + ".gif")
    duration = START_MS + END_MS
    with path.open("wb") as stream:
        header, _ = GifImagePlugin.getheader(previous, info={"loop": 0, "optimize": False})
        for block in header:
            stream.write(block)
        for block in GifImagePlugin.getdata(previous, duration=START_MS, disposal=1):
            stream.write(block)
        accumulated = []
        for index, (row, column, local) in enumerate(data["cells"]):
            accumulated.extend(local)
            current = render_frame(base, data, accumulated, palette, (row, column))
            box = ImageChops.difference(previous, current).getbbox()
            assert box is not None
            delay = 60 + (100 if local else 0)
            for block in GifImagePlugin.getdata(
                current.crop(box), offset=box[:2], duration=delay, disposal=1
            ):
                stream.write(block)
            duration += delay
            previous = current
            if (index + 1) % 150 == 0:
                print(f"{stem}: {index + 1}/{len(data['cells'])}", flush=True)
        current = render_frame(base, data, accumulated, palette)
        box = ImageChops.difference(previous, current).getbbox()
        for block in GifImagePlugin.getdata(
            current.crop(box), offset=box[:2], duration=END_MS, disposal=1
        ):
            stream.write(block)
        stream.write(b";")
    with Image.open(path) as animation:
        assert animation.size == (SIZE, SIZE)
        assert animation.n_frames == data["resolution"] ** 2 + 2
        assert animation.info["loop"] == 0
        observed_duration = 0
        checkpoints = {0, len(data["cells"]) // 2, len(data["cells"]) + 1}
        teal_counts = []
        for frame_index in range(animation.n_frames):
            animation.seek(frame_index)
            animation.load()
            observed_duration += animation.info["duration"]
            if frame_index in checkpoints:
                pixels = np.asarray(animation.convert("RGB")).astype(int)
                target = np.array(Image.new("RGB", (1, 1), TEAL).getpixel((0, 0)))
                teal_counts.append(np.count_nonzero(np.max(np.abs(pixels - target), axis=2) < 18))
        assert observed_duration == duration
        assert teal_counts[0] == 0 < teal_counts[1] < teal_counts[2]
    data["frames"] = data["resolution"] ** 2 + 2
    data["duration"] = duration / 1000
    print(f"{path.name}: {data['frames']} frames, {data['duration']:.1f} s, verified", flush=True)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    polygon = boundary()
    assert np.max(np.abs(signed_distance(polygon, polygon))) < 1e-12
    signs = signed_distance(np.array([[0, 0], [LIMIT, LIMIT]]), polygon)
    assert signs[0] < 0 < signs[1]
    unit_cell = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    for mask in range(16):
        values = np.array([-1.0 if mask & (1 << i) else 1.0 for i in range(4)])
        local = cell_segments(unit_cell, values)
        expected = 0 if mask in (0, 15) else 4 if mask in (5, 10) else 1
        assert len(local) == expected
    coarse = sample_grid(12, polygon)
    write_animation(coarse, "marching_squares_coarse")
    print(f"Sampled contour error: {coarse['error']:.6f}")
    print("Animation decoded; progressive contour and numerical checks: OK")


if __name__ == "__main__":
    main()
