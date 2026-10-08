"""Cube math shared by the viewer and the tests.

Cube axis order is (y, x, frame). A frame window is inclusive on both ends
and is a frame index, applied to every cube and clipped to that cube.
"""

from __future__ import annotations

import csv
import os

import numpy as np

# Defaults the window starts from. The view can change frame rate and box size.
FPS = 60.0
T0_S = 0.0
BOX = 1
SMOOTH = 1
TLINEAR = None  # None = raw counts; "high" = 0.04 K/count; "low" = 0.4
DISPLAY = "std"
STRIDE = 1
NUDGE = 0.1

# Synthetic heating window. Frames [0, HEAT_ON) are baseline only.
HEAT_ON = 4
HEAT_OFF = 240
N_FRAMES = 1000

TLINEAR_RES = {"high": 0.04, "low": 0.4}

# Logarithmic polynomial used by Thermographic Signal Reconstruction.
# Shepard's Thermal Wave Imaging papers fit a low order, commonly 5 to 7.
TSR_DEGREE = 5

MODES = (
    ("std", "STD"),
    ("var", "VAR"),
    ("ptp", "PTP"),
    ("mean", "MEAN"),
    ("delta", "Δ FRAME"),
    ("first", "FRAME"),
    ("pca1", "PC1"),
    ("pca2", "PC2"),
    ("pca3", "PC3"),
    ("tsr1", "TSR1"),
    ("tsr2", "TSR2"),
)

# Component index passed to the singular-vector routine. 0 is PC1.
PCA_COMPONENTS = {"pca1": 0, "pca2": 1, "pca3": 2}

# Temporal reductions, the principal components, and the TSR fit use the
# dragged range. FRAME, TSR1, and TSR2 also follow the frame slider.
RANGE_MODES = frozenset({
    "std", "var", "ptp", "mean", "delta", "pca1", "pca2", "pca3", "tsr1", "tsr2",
})
SLIDER_MODES = frozenset({"first", "tsr1", "tsr2"})
DIVERGING_MODES = frozenset({
    "delta", "pca1", "pca2", "pca3", "tsr1", "tsr2",
})

MODE_LABEL = {
    "std": "standard deviation of counts",
    "var": "variance of counts",
    "ptp": "peak-to-peak counts",
    "mean": "mean counts",
    "delta": "orange frame minus green frame",
    "first": "selected frame counts",
    "pca1": "principal component 1",
    "pca2": "principal component 2",
    "pca3": "principal component 3",
    "tsr1": "1st logarithmic derivative",
    "tsr2": "2nd logarithmic derivative",
}


def load_log(path):
    """Read t_rel_s, therm_c, and therm2_c from a pulse log.

    Comment lines (#) and the text header are skipped. Rows that leave a
    thermistor field empty are dropped.
    """
    header = None
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parsed = next(csv.reader([line]))
            if header is None:
                header = [name.strip().lower() for name in parsed]
                continue
            rows.append(parsed)
    if header is None:
        raise ValueError(f"{os.path.basename(path)} has no header")
    index = {name: i for i, name in enumerate(header)}

    def column(*candidates):
        for cand in candidates:
            if cand in index:
                return index[cand]
        raise KeyError(
            f"{os.path.basename(path)} needs t_rel_s plus therm_c and therm2_c; "
            f"got {header}")

    it = column("t_rel_s", "t", "time_s", "time")
    i1 = column("therm_c", "t1", "therm1_c", "therm1")
    i2 = column("therm2_c", "t2", "therm2")
    t, t1, t2 = [], [], []
    for row in rows:
        if max(it, i1, i2) >= len(row):
            continue
        try:
            tv = float(row[it])
            a = float(row[i1])
            b = float(row[i2])
        except ValueError:
            continue
        t.append(tv)
        t1.append(a)
        t2.append(b)
    if not t:
        raise ValueError(f"no thermistor samples in {os.path.basename(path)}")
    return np.asarray(t, float), np.asarray(t1, float), np.asarray(t2, float)


def frame_bounds(n_frames, f0, f1):
    """Inclusive frame indexes clipped to a cube, ordered low to high."""
    if n_frames <= 0:
        return 0, 0
    last = n_frames - 1
    lo = int(np.clip(int(f0), 0, last))
    hi = int(np.clip(int(f1), 0, last))
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi


def heat_envelope(n_frames=N_FRAMES, heat_on=HEAT_ON, heat_off=HEAT_OFF,
                  decay_frames=90.0):
    """Source term versus frame.

    Frames before ``heat_on`` stay at 0. The source ramps from just above 0
    at ``heat_on`` to 1 at ``heat_off``, then shuts off and the tail decays.
    """
    if n_frames <= 0:
        return np.zeros(0, dtype=np.float64)
    frame = np.arange(n_frames, dtype=np.float64)
    env = np.zeros(n_frames, dtype=np.float64)
    heat_on = int(heat_on)
    heat_off = int(min(heat_off, n_frames - 1))
    if heat_off < heat_on:
        return env
    rising = (frame >= heat_on) & (frame <= heat_off)
    span = max(heat_off - heat_on, 1)
    phase = np.clip((frame[rising] - heat_on) / span, 0.0, 1.0)
    smooth = phase * phase * (3.0 - 2.0 * phase)
    if smooth.size:
        smooth = np.maximum(smooth, 1e-3)
        smooth[-1] = 1.0
    env[rising] = smooth
    cooling = frame > heat_off
    env[cooling] = np.exp(-(frame[cooling] - heat_off) / float(decay_frames))
    return env


def display_image(cube, mode, frame=0, subtract_first=False, f0=0, f1=None,
                  stride=STRIDE):
    """One 2-D map for ``mode``.

    ``std``, ``var``, ``ptp``, ``mean``, ``pca1``, ``pca2``, ``pca3``,
    ``delta``, ``tsr1``, and ``tsr2`` use the inclusive frame window.
    ``delta`` is the high frame
    minus the low frame, the orange line minus the green line. ``first``
    uses the single ``frame``. ``tsr1`` and ``tsr2`` evaluate the logarithmic
    derivatives at ``frame``, clamped into the fit window.
    """
    if cube.ndim != 3:
        raise ValueError(f"expected a 3-D cube, got {cube.shape}")
    nf = int(cube.shape[2])
    if nf <= 0:
        raise ValueError("cube has no frames")
    if f1 is None:
        f1 = nf - 1
    lo, hi = frame_bounds(nf, f0, f1)
    frame = int(np.clip(int(frame), 0, nf - 1))
    step = max(int(stride), 1)
    base = np.asarray(cube[:, :, 0], dtype=np.float32) if subtract_first else None

    if mode == "first":
        img = np.asarray(cube[:, :, frame], dtype=np.float32)
        return img if base is None else img - base
    if mode == "delta":
        # High edge is the orange line, low edge is the green line.
        return (np.asarray(cube[:, :, hi], dtype=np.float32)
                - np.asarray(cube[:, :, lo], dtype=np.float32))
    if mode in PCA_COMPONENTS:
        return pca_component(
            cube, lo, hi, component=PCA_COMPONENTS[mode],
            subtract_first=subtract_first, stride=step)
    if mode in ("tsr1", "tsr2"):
        fit = fit_log_polynomial(
            cube, lo, hi, subtract_first=subtract_first, stride=step)
        return log_derivative_image(fit, frame, 1 if mode == "tsr1" else 2)

    sub = np.asarray(cube[:, :, lo:hi + 1:step], dtype=np.float32)
    if subtract_first:
        sub = sub - base[:, :, None]
    if mode == "std":
        return sub.std(axis=2)
    if mode == "var":
        return sub.var(axis=2)
    if mode == "ptp":
        return sub.max(axis=2) - sub.min(axis=2)
    if mode == "mean":
        return sub.mean(axis=2)
    raise ValueError(f"unknown display mode {mode!r}")


def pca_component(cube, f0, f1, component=1, subtract_first=False, stride=1):
    """Spatial scores of one principal component over an inclusive frame window.

    This is principal component thermography. Each pixel is one observation
    and the frames are the variables. The temporal mean of each pixel is
    removed, which is the Rajic centering: the decomposition describes the
    shape of the transient, not the absolute level. The score of component
    ``k`` (0 is the first) is the pixel projection onto that temporal
    singular vector, scaled by its singular value. The sign is then chosen
    so the map correlates positively with the temporal standard deviation.
    """
    nf = int(cube.shape[2])
    lo, hi = frame_bounds(nf, f0, f1)
    step = max(int(stride), 1)
    sub = np.asarray(cube[:, :, lo:hi + 1:step], dtype=np.float32)
    ny, nx, nt = sub.shape
    if subtract_first:
        base = np.asarray(cube[:, :, 0], dtype=np.float32)
        sub -= base[:, :, None]
    if nt < 2 or component >= nt:
        return np.zeros((ny, nx), dtype=np.float32)

    activity = sub.std(axis=2)
    sub -= sub.mean(axis=2, keepdims=True)
    flat = sub.reshape(-1, nt)
    gram = np.zeros((nt, nt), dtype=np.float64)
    step_rows = 8192
    # The Gram matrix is (time x time). Chunking holds peak memory to one
    # block even when a real Boson cube is hundreds of thousands of pixels.
    for start in range(0, flat.shape[0], step_rows):
        block = flat[start:start + step_rows].astype(np.float64, copy=False)
        gram += block.T @ block
    evals, evecs = np.linalg.eigh(gram)
    order = np.argsort(evals)[::-1]
    vec = evecs[:, order[int(component)]].astype(np.float32)
    scores = np.empty(flat.shape[0], dtype=np.float32)
    for start in range(0, flat.shape[0], step_rows):
        scores[start:start + step_rows] = flat[start:start + step_rows] @ vec
    weights = activity.reshape(-1).astype(np.float64, copy=False)
    if float(scores @ weights) < 0.0:
        scores = -scores
    return scores.reshape(ny, nx)


class _LogFit:
    """Polynomial for ln(temperature rise) against normalized ln(time)."""

    def __init__(self, coeff, mu, sig, f0, t_min, t_max, ny, nx):
        self.coeff = coeff
        self.mu = mu
        self.sig = sig
        self.f0 = f0
        self.t_min = t_min
        self.t_max = t_max
        self.ny = ny
        self.nx = nx


_FIT_CACHE = {}
_FIT_CACHE_LIMIT = 8


def polynomial_derivative(coeff, x, order):
    """Evaluate the ``order`` derivative of ``sum coeff[k] x**k``.

    ``coeff`` is ``(degree + 1,)`` or ``(degree + 1, n_series)``, low degree
    first. ``x`` is one-dimensional. The result has shape ``(n_series, x.size)``.
    """
    coeff = np.asarray(coeff, dtype=np.float64)
    if coeff.ndim == 1:
        coeff = coeff[:, None]
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    series = coeff
    for _ in range(int(order)):
        if series.shape[0] <= 1:
            return np.zeros((coeff.shape[1], x.size), dtype=np.float64)
        scales = np.arange(1, series.shape[0], dtype=np.float64)[:, None]
        series = series[1:] * scales
    return np.asarray(np.polynomial.polynomial.polyval(x, series), dtype=np.float64)


def fit_log_polynomial(cube, f0, f1, subtract_first=False, stride=1,
                       degree=TSR_DEGREE):
    """Fit the Thermal Wave Imaging logarithmic polynomial on one window.

    Time is measured in frames from the low edge of the window, the green
    line. The sample at that edge is the time origin and is left out of the
    fit, because ln(t) is undefined there. Temperature rise is the frame
    minus the pre-window frame, or minus frame 0 when ``subtract_first`` is
    set. The fit is ``ln(ΔT) = q(u)`` with ``u`` the normalized ``ln(t)``.
    Repeated calls on the same cube and window reuse the coefficients.
    """
    nf = int(cube.shape[2])
    lo, hi = frame_bounds(nf, f0, f1)
    step = max(int(stride), 1)
    key = (
        cube.__array_interface__["data"][0],
        cube.shape,
        cube.strides,
        lo,
        hi,
        step,
        bool(subtract_first),
        int(degree),
    )
    cached = _FIT_CACHE.get(key)
    if cached is not None:
        return cached
    fit = _fit_log_polynomial(cube, lo, hi, step, subtract_first, degree)
    _FIT_CACHE[key] = fit
    while len(_FIT_CACHE) > _FIT_CACHE_LIMIT:
        _FIT_CACHE.pop(next(iter(_FIT_CACHE)))
    return fit


def _fit_log_polynomial(cube, lo, hi, step, subtract_first, degree):
    ny, nx, _nf = cube.shape
    frames = np.arange(lo + step, hi + 1, step, dtype=np.int64)
    empty = _LogFit(None, 0.0, 1.0, lo, 1.0, 1.0, ny, nx)
    if frames.size < 4:
        return empty
    t = (frames - lo).astype(np.float64)
    log_t = np.log(t)
    sig = float(log_t.std())
    if not np.isfinite(sig) or sig <= 0.0:
        return empty
    mu = float(log_t.mean())
    deg = min(int(degree), int(frames.size) - 2)
    if deg < 1:
        return empty
    u = (log_t - mu) / sig
    vandermonde = np.polynomial.polynomial.polyvander(u, deg)
    if subtract_first or lo == 0:
        base_index = 0
    else:
        base_index = lo - 1
    base = np.asarray(cube[:, :, base_index], dtype=np.float64)
    delta = np.asarray(cube[:, :, frames], dtype=np.float64) - base[:, :, None]
    np.maximum(delta, 1.0, out=delta)
    np.log(delta, out=delta)
    flat = np.ascontiguousarray(delta.reshape(-1, frames.size))
    coeff = np.empty((deg + 1, flat.shape[0]), dtype=np.float64)
    for start in range(0, flat.shape[0], 8192):
        block = np.ascontiguousarray(flat[start:start + 8192].T)
        solved, *_residual = np.linalg.lstsq(vandermonde, block, rcond=None)
        coeff[:, start:start + solved.shape[1]] = solved
    return _LogFit(coeff, mu, sig, lo, float(t[0]), float(t[-1]), ny, nx)


def log_derivative_image(fit, frame, order):
    """Logarithmic derivative of a fitted pixel, at one frame.

    ``order`` 1 is ``d ln(ΔT) / d ln(t)`` and ``order`` 2 is the second
    derivative with respect to ``ln(t)``. The frame is clamped into the
    fitted times. A constant scale on ``ΔT`` cancels, which is the thermal
    wave property that heating nonuniformity drops out of the derivatives.
    """
    if fit.coeff is None:
        return np.zeros((fit.ny, fit.nx), dtype=np.float32)
    t = float(frame) - float(fit.f0)
    t = float(np.clip(t, fit.t_min, fit.t_max))
    u = (np.log(t) - fit.mu) / fit.sig
    slope = polynomial_derivative(fit.coeff, np.array([u]), order)
    scale = float(fit.sig) ** int(order)
    image = (slope[:, 0] / scale).astype(np.float32)
    return image.reshape(fit.ny, fit.nx)


def smooth(y, n):
    n = int(n)
    if n <= 1 or n >= y.size:
        return y
    return np.convolve(y, np.ones(n) / n, mode="same")


def extract_trace(cube, px, py, half, subtract_first=False, smooth_n=SMOOTH,
                  tlinear=TLINEAR):
    """Mean of an odd box around one pixel, one value per frame."""
    ny, nx, _nf = cube.shape
    px, py = int(px), int(py)
    half = max(0, int(half))
    y0, y1 = max(0, py - half), min(ny, py + half + 1)
    x0, x1 = max(0, px - half), min(nx, px + half + 1)
    block = np.asarray(cube[y0:y1, x0:x1, :], dtype=np.float32)
    series = block.mean(axis=(0, 1))
    if subtract_first:
        series = series - series[0]
    if tlinear is not None:
        series = series * TLINEAR_RES[tlinear] - 273.15
    return smooth(series, smooth_n)


def percentile_limits(images, low=1.0, high=99.0):
    """Shared display limits from finite pixels of one or more maps."""
    parts = []
    for img in images:
        if img is None:
            continue
        finite = np.asarray(img)[np.isfinite(img)]
        if finite.size:
            parts.append(finite.ravel())
    if not parts:
        return 0.0, 1.0
    finite = parts[0] if len(parts) == 1 else np.concatenate(parts)
    lo, hi = np.percentile(finite, [low, high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
        if hi <= lo:
            hi = lo + 1.0
    return float(lo), float(hi)


def symmetric_limit(image):
    """Half-width of a diverging scale, from the 99th percentile of |image|."""
    if image is None:
        return 1.0
    finite = np.abs(np.asarray(image)[np.isfinite(image)])
    if finite.size == 0:
        return 1.0
    span = float(np.percentile(finite, 99))
    if not np.isfinite(span) or span <= 0:
        span = float(finite.max()) if finite.size else 1.0
    return span if span > 0 else 1.0


def left_axis_is_difference(show_difference, any_thermistor):
    """Left axis policy.

    Checked "A − B" forces the difference scale. With the box clear, the
    difference scale is still used when no thermistor log is loaded, because
    an empty thermistor axis has nothing to show.
    """
    if show_difference:
        return True
    return not any_thermistor


def pad_limits(lo, hi, fraction=0.08):
    lo, hi = float(lo), float(hi)
    span = hi - lo
    pad = fraction * span if span > 0 else max(abs(lo) * 0.05, 1.0)
    return lo - pad, hi + pad


def mode_caption(mode, frame, f0, f1):
    if mode == "first":
        return f"frame {frame} counts"
    if mode == "delta":
        return f"orange frame {f1} minus green frame {f0}"
    if mode in PCA_COMPONENTS:
        return f"PC{PCA_COMPONENTS[mode] + 1}, frames {f0}–{f1}"
    if mode == "tsr1":
        shown = int(np.clip(int(frame), int(f0) + 1, max(int(f1), int(f0) + 1)))
        return f"1st logarithmic derivative at frame {shown}, fit {f0}–{f1}"
    if mode == "tsr2":
        shown = int(np.clip(int(frame), int(f0) + 1, max(int(f1), int(f0) + 1)))
        return f"2nd logarithmic derivative at frame {shown}, fit {f0}–{f1}"
    label = MODE_LABEL.get(mode, mode)
    if mode in RANGE_MODES:
        return f"{label}, frames {f0}–{f1}"
    return label
