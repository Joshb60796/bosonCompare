"""Unsigned 16-bit thermal cubes with a known heating window.

Three cubes share a scene and differ in spot strength, spot position, and a
second region whose heating starts later. That second region is what gives
principal component 2 a spatial shape.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from bosoncompare.analysis import (
    FPS,
    HEAT_OFF,
    HEAT_ON,
    N_FRAMES,
    heat_envelope,
)

NY = 96
NX = 128

# Distinct seeds so each set's noise is stable from run to run.
SET_SPECS = (
    {
        "name": "set1",
        "seed": 7,
        "gain": 1.00,
        "main_center": (48, 78),
        "main_sigma": (10.0, 14.0),
        "main_amp": 2200.0,
        "late_center": (34, 46),
        "late_sigma": (6.0, 7.0),
        "late_amp": 900.0,
        "late_on": 80,
        "therm_gain": (18.0, 11.0),
    },
    {
        "name": "set2",
        "seed": 11,
        "gain": 1.04,
        "main_center": (51, 84),
        "main_sigma": (11.0, 13.0),
        "main_amp": 2700.0,
        "late_center": (36, 50),
        "late_sigma": (5.5, 6.5),
        "late_amp": 500.0,
        "late_on": 100,
        "therm_gain": (20.5, 12.0),
    },
    {
        "name": "set3",
        "seed": 19,
        "gain": 0.96,
        "main_center": (45, 72),
        "main_sigma": (16.0, 18.0),
        "main_amp": 1600.0,
        "late_center": (40, 58),
        "late_sigma": (8.0, 8.0),
        "late_amp": 1400.0,
        "late_on": 60,
        "therm_gain": (15.0, 13.5),
    },
)


def _blob(yy, xx, center, sigma):
    cy, cx = center
    sy, sx = sigma
    return np.exp(-0.5 * (((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2)).astype(np.float32)


def make_cube(spec, n_frames=N_FRAMES, ny=NY, nx=NX):
    """Return one ``uint16`` cube of shape (y, x, frame)."""
    yy, xx = np.mgrid[0:ny, 0:nx]
    yy = yy.astype(np.float32)
    xx = xx.astype(np.float32)
    rng = np.random.default_rng(spec["seed"])
    baseline = 9000.0 + 380.0 * (xx / nx) + 70.0 * np.sin(yy / 7.0)
    baseline[28:68, 36:104] += 550.0
    baseline[40:46, 20:108] -= 180.0
    baseline += rng.normal(0.0, 8.0, size=(nx,)).astype(np.float32)[None, :]
    baseline = (baseline * spec["gain"]).astype(np.float32)

    main = spec["main_amp"] * _blob(yy, xx, spec["main_center"], spec["main_sigma"])
    late = spec["late_amp"] * _blob(yy, xx, spec["late_center"], spec["late_sigma"])
    env_main = heat_envelope(n_frames, HEAT_ON, HEAT_OFF, decay_frames=90.0).astype(np.float32)
    env_late = heat_envelope(
        n_frames, spec["late_on"], HEAT_OFF, decay_frames=150.0).astype(np.float32)

    stack = (baseline[:, :, None]
             + main[:, :, None] * env_main
             + late[:, :, None] * env_late)
    stack += rng.normal(0.0, 6.0, size=stack.shape).astype(np.float32)
    return np.clip(np.rint(stack), 0, 65535).astype(np.uint16)


def make_log(spec, n_frames=N_FRAMES, fps=FPS):
    """Thermistor rows spanning the same camera interval."""
    duration = (n_frames - 1) / float(fps)
    t = np.arange(0.0, duration + 1e-9, 0.05, dtype=np.float64)
    frame = np.clip(np.rint(t * fps).astype(int), 0, n_frames - 1)
    env = heat_envelope(n_frames, HEAT_ON, HEAT_OFF, decay_frames=90.0)
    env_t = env[frame]
    g1, g2 = spec["therm_gain"]
    rng = np.random.default_rng(spec["seed"] + 100)
    therm = 25.0 + g1 * env_t + rng.normal(0.0, 0.03, size=t.size)
    therm2 = 24.6 + g2 * env_t + rng.normal(0.0, 0.03, size=t.size)
    return t, therm, therm2


def write_log(path, t, therm, therm2):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("# synthetic thermistor log for Boson Compare\n")
        handle.write("t_rel_s,therm_c,therm2_c\n")
        for tv, a, b in zip(t, therm, therm2):
            handle.write(f"{tv:.4f},{a:.4f},{b:.4f}\n")


def synthetic_dir(root=None):
    if root is None:
        root = Path(__file__).resolve().parents[1]
    return Path(root) / "synthetic_data"


def cube_paths(root=None):
    folder = synthetic_dir(root)
    return [folder / f"{spec['name']}_cube.npy" for spec in SET_SPECS]


def log_paths(root=None):
    folder = synthetic_dir(root)
    return [folder / f"{spec['name']}_thermistor.csv" for spec in SET_SPECS]


def save_npy(path, array):
    """Write an array, replacing ``path`` only after the file is complete.

    ``np.save`` opens the destination with truncation. On Windows that raises
    ``OSError: [Errno 22] Invalid argument`` when the destination is still
    memory-mapped, and it also leaves a truncated file if the write fails.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.partial.npy")
    np.save(temporary, array)
    os.replace(temporary, path)


def generate(root=None, n_frames=N_FRAMES):
    """Write three cubes and three logs. Returns ``(cube_paths, log_paths)``."""
    folder = synthetic_dir(root)
    folder.mkdir(parents=True, exist_ok=True)
    cubes, logs = [], []
    for spec in SET_SPECS:
        cube_path = folder / f"{spec['name']}_cube.npy"
        log_path = folder / f"{spec['name']}_thermistor.csv"
        cube = make_cube(spec, n_frames=n_frames)
        save_npy(cube_path, cube)
        t, therm, therm2 = make_log(spec, n_frames=n_frames)
        write_log(log_path, t, therm, therm2)
        cubes.append(cube_path)
        logs.append(log_path)
    return cubes, logs


def ensure(root=None, n_frames=N_FRAMES):
    """Generate the set when any file is missing or the wrong length."""
    cubes = cube_paths(root)
    logs = log_paths(root)
    ready = all(path.is_file() for path in cubes + logs)
    if ready:
        try:
            probe = np.load(cubes[0], mmap_mode="r")
            ready = probe.ndim == 3 and probe.shape[2] == n_frames and probe.dtype == np.uint16
        except (OSError, ValueError):
            ready = False
    if not ready:
        return generate(root, n_frames=n_frames)
    return cubes, logs


def describe(root=None):
    folder = synthetic_dir(root)
    return os.fspath(folder)
