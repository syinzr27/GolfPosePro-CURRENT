# smoothing with one euro filter

import math
import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

# one euro filter implementation
class _LowPassFilter:
    """Simple first-order exponential low-pass filter."""

    def __init__(self, alpha: float = 1.0):
        self._alpha = alpha
        self._y = None
        self._initialized = False

    def __call__(self, value: float, alpha: float = None) -> float:
        if alpha is not None:
            self._alpha = alpha
        if not self._initialized:
            self._y = value
            self._initialized = True
        else:
            self._y = self._alpha * value + (1.0 - self._alpha) * self._y
        return self._y
    @property
    def last(self) -> float:
        return self._y if self._y is not None else 0.0

    def reset(self):
        self._y = None
        self._initialized = False

class OneEuroFilter:
    def __init__(
        self,
        freq: float = 30.0,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self._x_filter = _LowPassFilter()
        self._dx_filter = _LowPassFilter()
        self._last_time = None
    @staticmethod
    def _alpha(cutoff: float, freq: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float, timestamp: float = None) -> float:
        if self._last_time is not None and timestamp is not None:
            dt = timestamp - self._last_time
            self.freq = 1.0 / max(dt, 1e-9)
        self._last_time = timestamp

        # derivative of signal
        prev = self._x_filter.last
        if prev is None or not self._x_filter._initialized:
            dx = 0.0
        else:
            dx = (x - prev) * self.freq

        # smooth the derivative
        edx = self._dx_filter(dx, alpha=self._alpha(self.d_cutoff, self.freq))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self._x_filter(x, alpha=self._alpha(cutoff, self.freq))

    def reset(self):
        """Reset internal state (call between separate sequences)."""
        self._x_filter.reset()
        self._dx_filter.reset()
        self._last_time = None

# more smoothing :)
def smooth_series(
    data: np.ndarray,
    fps: float = 30.0,
    min_cutoff: float = 1.0,
    beta: float = 0.007,
    d_cutoff: float = 1.0,
) -> np.ndarray:
    # one-euro filter for 1D time-series
    if len(data) == 0:
        return data.copy()
    filt = OneEuroFilter(freq=fps, min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
    out = np.empty_like(data, dtype=float)
    for i, val in enumerate(data):
        if np.isnan(val):
            out[i] = val
        else:
            out[i] = filt(float(val), timestamp=i / fps)
    return out

def smooth_landmarks(
    frame_data: list,
    landmark_keys: list = None,
    fps: float = 30.0,
    min_cutoff: float = 1.0,
    beta: float = 0.007,
    d_cutoff: float = 1.0,
) -> list:
    # one-euro filter for all landmark coordinate series
    if not frame_data:
        return frame_data
    if landmark_keys is None:
        sample = frame_data[0]
        landmark_keys = [k for k in sample if k != "frame_idx" and
                         (k.endswith("_x") or k.endswith("_y"))]
    if not landmark_keys:
        return frame_data
    n = len(frame_data)
    smoothed_arrays = {}
    for key in landmark_keys:
        raw = np.array(
            [f.get(key) if f.get(key) is not None else np.nan for f in frame_data],
            dtype=float,
        )
        nans = np.isnan(raw)
        if nans.any() and not nans.all():
            idx = np.where(~nans, np.arange(n), 0)
            np.maximum.accumulate(idx, out=idx)
            raw = raw[idx]
        smoothed_arrays[key] = smooth_series(
            raw, fps=fps, min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff
        )
    # reconstruct frame_data with smoothed values
    result = []
    for i, frame in enumerate(frame_data):
        new_frame = {"frame_idx": frame["frame_idx"]}
        for key in landmark_keys:
            new_frame[key] = float(smoothed_arrays[key][i])
        # Preserve any extra keys not in landmark_keys
        for k, v in frame.items():
            if k not in new_frame:
                new_frame[k] = v
        result.append(new_frame)
    return result

def smooth_landmarks_3d(
    frames_3d: list,
    fps: float = 30.0,
    min_cutoff: float = 1.0,
    beta: float = 0.007,
    d_cutoff: float = 1.0,
) -> list:
    # one-euro filter for all 3D landmark coordinate series
    if not frames_3d:
        return frames_3d
    n = len(frames_3d)
    by_idx = {}  # landmark idx -> list of landmark dict-or-None, aligned to frames_3d order
    for frame in frames_3d:
        for l in frame["landmarks"]:
            by_idx.setdefault(l["idx"], [None] * n)
    for fi, frame in enumerate(frames_3d):
        lm_by_idx = {l["idx"]: l for l in frame["landmarks"]}
        for idx, entries in by_idx.items():
            entries[fi] = lm_by_idx.get(idx)
    for entries in by_idx.values():
        mask = [e is not None and e["x"] is not None for e in entries]
        if not any(mask):
            continue
        for axis in ("x", "y", "z"):
            raw = np.array(
                [e[axis] if m else np.nan for e, m in zip(entries, mask)], dtype=float
            )
            nans = np.isnan(raw)
            if nans.all():
                continue
            if nans.any():
                filled_idx = np.where(~nans, np.arange(n), 0)
                np.maximum.accumulate(filled_idx, out=filled_idx)
                raw = raw[filled_idx]
            smoothed = smooth_series(
                raw, fps=fps, min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff
            )
            for fi in range(n):
                if mask[fi]:
                    entries[fi][axis] = round(float(smoothed[fi]), 5)
    return frames_3d

def _finish_weights(n: int, finish: tuple = None) -> np.ndarray:
    # smoothing weights for each frame in finish
    w = np.zeros(n)
    if finish is None:
        return w
    start, end = finish
    for fi in range(n):
        w[fi] = 1.0 if fi >= end else (0.0 if fi <= start else (fi - start) / max(end - start, 1))
    return w

def _blend_finish(pts: np.ndarray, sigma: float, finish_sigma: float, w: np.ndarray) -> np.ndarray:
    # gaussian smoothing for finish
    out = gaussian_filter1d(pts, sigma, axis=0, mode="nearest")
    if w.max() > 0:
        heavy = gaussian_filter1d(pts, finish_sigma, axis=0, mode="nearest")
        out = out + w[:, None] * (heavy - out)
    return out

def smooth_landmarks_3d_for_display(frames_3d: list, sigma: float = 1.0,
                                    finish: tuple = None,
                                    finish_sigma: float = 3.0) -> list:
    # gaussian smoothing of all visible landmark series for display
    out = [
        {**f, "landmarks": [dict(l) for l in f.get("landmarks", [])]}
        for f in frames_3d
    ]
    n = len(out)
    weights = _finish_weights(n, finish)
    by_idx = {}
    for fi, frame in enumerate(out):
        for l in frame["landmarks"]:
            by_idx.setdefault(l["idx"], [None] * n)[fi] = l
    for entries in by_idx.values():
        fi = 0
        while fi < n:
            if entries[fi] is None or entries[fi]["x"] is None:
                fi += 1
                continue
            end = fi
            while end < n and entries[end] is not None and entries[end]["x"] is not None:
                end += 1
            run = entries[fi:end]
            pts = np.array([[e["x"], e["y"], e["z"]] for e in run], dtype=float)
            pts = _blend_finish(pts, sigma, finish_sigma, weights[fi:end])
            for e, p in zip(run, pts):
                e["x"], e["y"], e["z"] = (round(float(v), 5) for v in p)
            fi = end
    return out

def smooth_club_for_display(club_data: list, sigma: float = 1.0,
                            finish: tuple = None, finish_sigma: float = 3.0) -> list:
    # gaussian smoothing of clubhead for display
    out = [dict(c) for c in club_data]
    n = len(out)
    weights = _finish_weights(n, finish)
    fi = 0
    while fi < n:
        if not out[fi].get("has_club"):
            fi += 1
            continue
        end = fi
        while end < n and out[end].get("has_club"):
            end += 1
        for key in ("grip", "shaft_end"):
            pts = np.array([[out[i][key][a] for a in "xyz"] for i in range(fi, end)], dtype=float)
            pts = _blend_finish(pts, sigma, finish_sigma, weights[fi:end])
            for i, p in zip(range(fi, end), pts):
                out[i][key] = {a: round(float(v), 5) for a, v in zip("xyz", p)}
        fi = end
    return out

def smooth_series_uniform( # alt method, just a simple moving average
    data: np.ndarray,
    window: int = 5,
) -> np.ndarray:
    return uniform_filter1d(data.astype(float), size=window, mode="nearest")