# extract coords using mediapipe

import os
import subprocess
from pathlib import Path
import cv2
import numpy as np
try:
    import mediapipe as mp
    from mediapipe.tasks.python.vision import (
        PoseLandmarker,
        PoseLandmarkerOptions,
        RunningMode,
        PoseLandmark,
    )
    from mediapipe.tasks.python.core.base_options import BaseOptions
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
_DEFAULT_MODEL = Path(__file__).parent.parent / "pose_landmarker.task"

# left/right landmark index pairs (left, right)
_LR_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
    (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]
LANDMARK_GROUPS = {
    "wrist": [PoseLandmark.LEFT_WRIST.value, PoseLandmark.RIGHT_WRIST.value] if MP_AVAILABLE else [],
    "hip": [PoseLandmark.LEFT_HIP.value, PoseLandmark.RIGHT_HIP.value] if MP_AVAILABLE else [],
    "shoulder": [PoseLandmark.LEFT_SHOULDER.value, PoseLandmark.RIGHT_SHOULDER.value] if MP_AVAILABLE else [],
}

# video helper funcs
def add_silent_audio(input_path: str, output_path: str) -> str:
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-shortest",
        "-c:v", "libx264", "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        output_path,
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return output_path
def get_video_info(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    info = {
        "width":        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps":          cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info

# the actual pose extraction from video using mediapipe
def extract_pose_features(
    video_path: str,
    track_landmarks: list = None,
    visibility_threshold: float = 0.4,
    model_path: str = None,
    progress_callback=None,
    apply_smoothing: bool = False,
    smoothing_min_cutoff: float = 1.0,
    smoothing_beta: float = 0.007,
) -> list:
    if not MP_AVAILABLE:
        raise RuntimeError("mediapipe is not installed. Run: pip install mediapipe")
    if track_landmarks is None:
        track_landmarks = ["wrist"]
    if model_path is None:
        model_path = str(_DEFAULT_MODEL)
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Pose model not found at: {model_path}\n"
            "Download it with:\n"
            "  curl -L -o pose_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
        )
    lm_indices = {group: LANDMARK_GROUPS[group] for group in track_landmarks
                  if group in LANDMARK_GROUPS}
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    frame_data = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    with PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            timestamp_ms = int(frame_idx * 1000 / fps)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            entry = {"frame_idx": frame_idx}
            poses = result.pose_landmarks
            if poses:
                landmarks = poses[0]
                for group, indices in lm_indices.items():
                    x_vals = []
                    y_vals = []
                    for idx in indices:
                        lm = landmarks[idx]
                        if lm.visibility is None or lm.visibility > visibility_threshold:
                            x_vals.append(lm.x * width)
                            y_vals.append(lm.y * height)
                    entry[f"{group}_x"] = float(np.mean(x_vals)) if x_vals else None
                    entry[f"{group}_y"] = float(np.mean(y_vals)) if y_vals else None
            else:
                for group in track_landmarks:
                    entry[f"{group}_x"] = None
                    entry[f"{group}_y"] = None
            frame_data.append(entry)
            frame_idx += 1
            if progress_callback:
                progress_callback(frame_idx, total_frames)
    cap.release()
    # one euro smoothing
    if apply_smoothing and frame_data:
        from .smoothing import smooth_landmarks
        frame_data = smooth_landmarks(
            frame_data, fps=fps,
            min_cutoff=smoothing_min_cutoff, beta=smoothing_beta,
        )
    return frame_data

def extract_full_3d_landmarks(
    video_path: str,
    visibility_threshold: float = 0.4,
    model_path: str = None,
    progress_callback=None,
    apply_smoothing: bool = True,
    smoothing_min_cutoff: float = 1.0,
    smoothing_beta: float = 0.007,
) -> list:
    if not MP_AVAILABLE:
        raise RuntimeError("mediapipe is not installed. Run: pip install mediapipe")
    if model_path is None:
        model_path = str(_DEFAULT_MODEL)
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Pose model not found at: {model_path}\n"
            "Download it with:\n"
            "  curl -L -o pose_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
        )
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    frames_3d = []
    image_pts = []
    with PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            timestamp_ms = int(frame_idx * 1000 / fps)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            entry = {"frame_idx": frame_idx, "timestamp_ms": timestamp_ms, "landmarks": []}
            world_landmarks = result.pose_world_landmarks
            if world_landmarks:
                wl = world_landmarks[0]
                for i, lm in enumerate(wl):
                    vis = lm.visibility if lm.visibility is not None else 0.0
                    if vis >= visibility_threshold:
                        entry["landmarks"].append({
                            "idx": i,
                            "x": round(float(lm.x), 5),
                            "y": round(float(lm.y), 5),
                            "z": round(float(lm.z), 5),
                            "visibility": round(float(vis), 3),
                        })
                    else:
                        entry["landmarks"].append({
                            "idx": i, "x": None, "y": None, "z": None,
                            "visibility": round(float(vis), 3),
                        })
            pose_2d = result.pose_landmarks
            image_pts.append(
                {i: np.array([pose_2d[0][i].x * width, pose_2d[0][i].y * height])
                 for i in _SCALE_FIT_IDX + _IMAGE_EXTRA_IDX}
                if pose_2d else None
            )
            frames_3d.append(entry)
            frame_idx += 1
            if progress_callback:
                progress_callback(frame_idx, total_frames)
    cap.release()
    _fix_left_right_swaps(frames_3d)
    _fix_hips_and_shoulders(frames_3d, image_pts)
    _fix_back_to_camera(frames_3d, image_pts)
    _fix_occluded_arms(frames_3d)
    _straighten_arms(frames_3d, image_pts)
    if apply_smoothing:
        from .smoothing import smooth_landmarks_3d
        smooth_landmarks_3d(
            frames_3d, fps=fps,
            min_cutoff=smoothing_min_cutoff, beta=smoothing_beta,
        )
    return frames_3d

def _dist2(p: tuple, q: tuple) -> float:
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2

def _fix_left_right_swaps(
    frames_3d: list, min_pairs: int = 4, margin: float = 0.5, ref_alpha: float = 0.25,
) -> list:
    # correct left/right limb swaps in mediapipe landmarks
    ref = {}  # idx -> (x, y, z), slow moving reference
    for frame in frames_3d:
        lm_by_idx = {l["idx"]: l for l in frame["landmarks"] if l["x"] is not None}
        keep_cost = 0.0
        swap_cost = 0.0
        swappable = []
        for a, b in _LR_PAIRS:
            la, lb = lm_by_idx.get(a), lm_by_idx.get(b)
            ra, rb = ref.get(a), ref.get(b)
            if la is None or lb is None or ra is None or rb is None:
                continue
            swappable.append((a, b))
            pa, pb = (la["x"], la["y"], la["z"]), (lb["x"], lb["y"], lb["z"])
            keep_cost += _dist2(pa, ra) + _dist2(pb, rb)
            swap_cost += _dist2(pa, rb) + _dist2(pb, ra)
        if len(swappable) >= min_pairs and swap_cost < keep_cost * margin:
            for a, b in swappable:
                la, lb = lm_by_idx[a], lm_by_idx[b]
                la["x"], lb["x"] = lb["x"], la["x"]
                la["y"], lb["y"] = lb["y"], la["y"]
                la["z"], lb["z"] = lb["z"], la["z"]
                la["visibility"], lb["visibility"] = lb["visibility"], la["visibility"]
        for l in frame["landmarks"]:
            if l["x"] is None:
                continue
            p = (l["x"], l["y"], l["z"])
            prev = ref.get(l["idx"])
            ref[l["idx"]] = p if prev is None else (
                ref_alpha * p[0] + (1 - ref_alpha) * prev[0],
                ref_alpha * p[1] + (1 - ref_alpha) * prev[1],
                ref_alpha * p[2] + (1 - ref_alpha) * prev[2],
            )
    return frames_3d

def _torso_yaw(pts: dict):
    # direction the chest faces
    if not all(i in pts for i in (11, 12, 23, 24)):
        return None
    across = pts[11] - pts[12]
    up = (pts[11] + pts[12]) / 2.0 - (pts[23] + pts[24]) / 2.0
    facing = np.cross(across, up)
    if np.linalg.norm(facing[[0, 2]]) < 1e-6:
        return None
    return float(np.degrees(np.arctan2(facing[0], -facing[2])))

# upper body left/right pairs
_UPPER_LR_PAIRS = [pair for pair in _LR_PAIRS if pair[0] <= 22]
_UPPER_SINGLE = (0,)

def _fix_back_to_camera(
    frames_3d: list, image_pts: list = None, drop_deg: float = 15.0, min_peak: float = 45.0,
    blend: int = 4,
):
    # fix mediapipe mirroring torso depth coordinates during finish
    n = len(frames_3d)
    pts_per_frame = [{l["idx"]: np.array([l["x"], l["y"], l["z"]], dtype=float)
                      for l in frame["landmarks"] if l["x"] is not None}
                     for frame in frames_3d]
    timing = _swing_timing(pts_per_frame)
    if timing is None:
        return None
    impact = timing[2]
    yaw = [_torso_yaw(p) for p in pts_per_frame]
    yaw = [None if y is None else np.array([y]) for y in yaw]
    if sum(y is not None for y in yaw) < 10:
        return None
    from scipy.ndimage import gaussian_filter1d, median_filter
    yaw = np.array([y[0] for y in _interp_vectors(yaw)])
    yaw = gaussian_filter1d(median_filter(yaw, size=11, mode="nearest"), 3.0)
    after = yaw[impact:]
    toward = 1.0 if after.max() - yaw[impact] >= yaw[impact] - after.min() else -1.0
    turn = toward * yaw
    peak = None
    top = impact
    for fi in range(impact, n):
        if turn[fi] > turn[top]:
            top = fi
        elif turn[fi] <= turn[top] - drop_deg:
            if turn[top] >= min_peak:
                peak = top
                break
            top = fi
    if peak is None:
        return None
    mirrored = [fi for fi in range(peak + 1, n) if turn[fi] < 90.0]
    if not mirrored:
        return None
    lo, hi = max(0, peak + 1 - blend), min(n - 1, peak + blend)
    upper = {i for pair in _UPPER_LR_PAIRS for i in pair} | set(_UPPER_SINGLE)
    start = {l["idx"]: (l["x"], l["y"], l["z"]) for l in frames_3d[lo]["landmarks"]
             if l["x"] is not None and l["idx"] in upper}
    for fi in mirrored:
        by_idx = {l["idx"]: l for l in frames_3d[fi]["landmarks"]}
        for a, b in _UPPER_LR_PAIRS:
            la, lb = by_idx.get(a), by_idx.get(b)
            if la is None or lb is None:
                continue
            for k in ("x", "y", "z", "visibility"):
                la[k], lb[k] = lb[k], la[k]
        ls, rs = by_idx.get(11), by_idx.get(12)
        if ls is None or rs is None or ls["z"] is None or rs["z"] is None:
            continue
        centre_z = (ls["z"] + rs["z"]) / 2.0
        for i in upper:
            l = by_idx.get(i)
            if l is not None and l["z"] is not None:
                l["z"] = round(2.0 * centre_z - l["z"], 5)
        img = image_pts[fi] if image_pts else None
        if img is not None:
            for a, b in _UPPER_LR_PAIRS:
                if a in img and b in img:
                    img[a], img[b] = img[b], img[a]
    # interpolating from end to start
    end = {l["idx"]: (l["x"], l["y"], l["z"]) for l in frames_3d[hi]["landmarks"]
           if l["x"] is not None and l["idx"] in upper}
    for fi in range(lo + 1, hi):
        t = (fi - lo) / (hi - lo)
        for l in frames_3d[fi]["landmarks"]:
            i = l["idx"]
            if i in start and i in end:
                l["x"], l["y"], l["z"] = (round((1 - t) * s0 + t * e0, 5) for s0, e0 in zip(start[i], end[i]))
    return peak + 1

# reconstructing low visibility arms
_ARM_CHAINS = {"left": (11, 13, 15), "right": (12, 14, 16)}
_HAND_TIPS = {"left": (17, 19, 21), "right": (18, 20, 22)}
def _unit(v):
    # normalize vector v
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else None

def _torso_basis(pts: dict):
    # 3D coord system from torso: right, up, forward
    ls, rs, lh, rh = pts.get(11), pts.get(12), pts.get(23), pts.get(24)
    if ls is None or rs is None or lh is None or rh is None:
        return None
    up = _unit((ls + rs) / 2.0 - (lh + rh) / 2.0)
    if up is None:
        return None
    right = rs - ls
    right = _unit(right - float(np.dot(right, up)) * up)
    if right is None:
        return None
    return right, up, np.cross(up, right)

def _to_local(v, basis):
    right, up, fwd = basis
    return np.array([float(np.dot(v, right)), float(np.dot(v, up)), float(np.dot(v, fwd))])

def _to_world(v, basis):
    right, up, fwd = basis
    return right * v[0] + up * v[1] + fwd * v[2]

def _elbow_pole(shoulder, elbow, wrist):
    # direction of elbow from shoulder to wrist axis
    axis = _unit(wrist - shoulder)
    if axis is None:
        return None
    d = elbow - shoulder
    return _unit(d - float(np.dot(d, axis)) * axis)

def _solve_elbow(shoulder, wrist, upper_len, fore_len, pole):
    # find elbow position given shoulder, wrist, bone lengths, and pole direction
    reach = wrist - shoulder
    d = float(np.linalg.norm(reach))
    if d < 1e-6 or upper_len <= 0 or fore_len <= 0:
        return None
    axis = reach / d
    if d >= upper_len + fore_len:
        # can't reach, return stretched line
        return shoulder + axis * d * (upper_len / (upper_len + fore_len))
    d = max(d, abs(upper_len - fore_len) + 1e-4)
    along = (d * d + upper_len ** 2 - fore_len ** 2) / (2.0 * d)
    out = float(np.sqrt(max(upper_len ** 2 - along ** 2, 0.0)))
    base = shoulder + axis * along
    if pole is None:
        return base
    p = _unit(pole - float(np.dot(pole, axis)) * axis)
    return base if p is None else base + p * out

def _interp_vectors(series: list) -> list:
    # interpolate missing vectors in the series
    known = [i for i, v in enumerate(series) if v is not None]
    if not known:
        return list(series)
    out = list(series)
    for i in range(known[0]):
        out[i] = series[known[0]].copy()
    for i in range(known[-1] + 1, len(series)):
        out[i] = series[known[-1]].copy()
    for a, b in zip(known, known[1:]):
        for i in range(a + 1, b):
            t = (i - a) / (b - a)
            out[i] = (1.0 - t) * series[a] + t * series[b]
    return out

def _set_landmark(frame: dict, idx: int, p) -> None:
    for l in frame["landmarks"]:
        if l["idx"] == idx:
            l["x"], l["y"], l["z"] = (round(float(p[0]), 5),
                                      round(float(p[1]), 5),
                                      round(float(p[2]), 5))
            return

def _fix_occluded_arms(
    frames_3d: list,
    vis_margin: float = 0.05,
    vis_reliable: float = 0.85,
    vis_floor: float = 0.7,
    span_factor: float = 1.6,
    bone_tol: tuple = (0.7, 1.4),
    min_calibration_frac: float = 0.1,
    reach_tol: float = 1.25,
) -> list:
    # reconstruct arm locations if not visible in video or sus
    n = len(frames_3d)
    if n == 0:
        return frames_3d
    pts_per_frame, vis_per_frame, basis_per_frame = [], [], []
    for frame in frames_3d:
        pts, vis = {}, {}
        for l in frame["landmarks"]:
            vis[l["idx"]] = l.get("visibility") or 0.0
            if l["x"] is not None:
                pts[l["idx"]] = np.array([l["x"], l["y"], l["z"]], dtype=float)
        pts_per_frame.append(pts)
        vis_per_frame.append(vis)
        basis_per_frame.append(_torso_basis(pts))

    def side_vis(fi, side):
        _, e, w = _ARM_CHAINS[side]
        return min(vis_per_frame[fi].get(e, 0.0), vis_per_frame[fi].get(w, 0.0))
    def has_chain(fi, side):
        return all(i in pts_per_frame[fi] for i in _ARM_CHAINS[side])
    
    need = max(15, int(np.ceil(min_calibration_frac * n)))
    both_vis = sorted((min(side_vis(fi, s) for s in _ARM_CHAINS)
                       for fi in range(n)
                       if all(has_chain(fi, s) for s in _ARM_CHAINS)),
                      reverse=True)
    if len(both_vis) >= need and both_vis[need - 1] < vis_reliable:
        vis_reliable = max(both_vis[need - 1], vis_floor)
    spans, shoulder_widths = [], []
    bones = {side: {"upper": [], "fore": []} for side in _ARM_CHAINS}
    for fi in range(n):
        pts = pts_per_frame[fi]
        if 11 in pts and 12 in pts:
            shoulder_widths.append(float(np.linalg.norm(pts[11] - pts[12])))
        if not all(has_chain(fi, s) and side_vis(fi, s) >= vis_reliable
                   for s in _ARM_CHAINS):
            continue
        spans.append(float(np.linalg.norm(pts[15] - pts[16])))
        for side, (s, e, w) in _ARM_CHAINS.items():
            bones[side]["upper"].append(float(np.linalg.norm(pts[s] - pts[e])))
            bones[side]["fore"].append(float(np.linalg.norm(pts[e] - pts[w])))
    if len(spans) < need:
        return frames_3d
    shoulder_w = float(np.median(shoulder_widths)) if shoulder_widths else 0.0
    max_span = max(span_factor * float(np.median(spans)), 0.5 * shoulder_w)
    lengths = {side: (float(np.median(bones[side]["upper"])),
                      float(np.median(bones[side]["fore"])))
               for side in _ARM_CHAINS}

    # reconstruction for each arm
    for side, other in (("left", "right"), ("right", "left")):
        s_i, e_i, w_i = _ARM_CHAINS[side]
        o_w = _ARM_CHAINS[other][2]
        upper_len, fore_len = lengths[side]
        def impossible(fi):
            pts = pts_per_frame[fi]
            if w_i not in pts or e_i not in pts:
                return True # mediapipe gave up on the joint entirely
            if np.linalg.norm(pts[w_i] - pts[o_w]) > max_span:
                return True # hands too far apart to be sharing one grip
            for a, b, ref in ((s_i, e_i, upper_len), (e_i, w_i, fore_len)):
                if a not in pts:
                    continue
                d = float(np.linalg.norm(pts[a] - pts[b]))
                if not (bone_tol[0] * ref <= d <= bone_tol[1] * ref):
                    return True # bone squashed or stretched out of existence
            return False
        bad = [False] * n
        for fi in range(n):
            if (basis_per_frame[fi] is None or not has_chain(fi, other)
                    or side_vis(fi, side) > side_vis(fi, other) - vis_margin):
                continue
            bad[fi] = impossible(fi)
        if not any(bad):
            continue

        offset_local = [None] * n # one wrist relative to the other, torso coords
        pole_local = [None] * n # elbow-out direction, torso coords
        tips_local = {t: [None] * n for t in _HAND_TIPS[side]} # hand shape, torso coords
        for fi in range(n):
            pts, basis = pts_per_frame[fi], basis_per_frame[fi]
            if (basis is None or not has_chain(fi, side)
                    or side_vis(fi, side) < vis_reliable
                    or o_w not in pts or impossible(fi)):
                continue
            offset_local[fi] = _to_local(pts[w_i] - pts[o_w], basis)
            pole = _elbow_pole(pts[s_i], pts[e_i], pts[w_i])
            if pole is not None:
                pole_local[fi] = _to_local(pole, basis)
            for t in _HAND_TIPS[side]:
                if t in pts:
                    tips_local[t][fi] = _to_local(pts[t] - pts[w_i], basis)
        offsets = _interp_vectors(offset_local)
        poles   = _interp_vectors(pole_local)
        tips    = {t: _interp_vectors(v) for t, v in tips_local.items()}
        for fi in range(n):
            if not bad[fi]:
                continue
            pts, basis = pts_per_frame[fi], basis_per_frame[fi]
            if s_i not in pts or o_w not in pts:
                continue
            off = offsets[fi]
            wrist = pts[o_w] + (_to_world(off, basis) if off is not None
                                else np.zeros(3))
            dropped = w_i not in pts or e_i not in pts
            if (not dropped
                    and np.linalg.norm(wrist - pts[s_i]) > (upper_len + fore_len) * reach_tol):
                continue
            pole = _to_world(poles[fi], basis) if poles[fi] is not None else None
            elbow = _solve_elbow(pts[s_i], wrist, upper_len, fore_len, pole)
            _set_landmark(frames_3d[fi], w_i, wrist)
            pts[w_i] = wrist
            if elbow is not None:
                _set_landmark(frames_3d[fi], e_i, elbow)
                pts[e_i] = elbow
            # carry the hand's own shape (pinky/index/thumb) to the new wrist
            for t in _HAND_TIPS[side]:
                if tips[t][fi] is None:
                    continue
                tip = wrist + _to_world(tips[t][fi], basis)
                _set_landmark(frames_3d[fi], t, tip)
                pts[t] = tip
    return frames_3d

# straightening arms
def _ramp(x: float, lo: float, hi: float) -> float:
    """0 at `lo`, 1 at `hi`, linear between (either order)."""
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))

def _elbow_angle(s, e, w) -> float:
    u, v = s - e, w - e
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu < 1e-8 or nv < 1e-8:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))))

def _image_elbow_offset(img: dict, chain: tuple, reach, trust_2d_angle: tuple) -> np.ndarray:
    # elbow offset from shoulder to wrist line
    s_i, e_i, w_i = chain
    if img is None or not all(i in img for i in chain):
        return np.zeros(2)
    s2, e2, w2 = img[s_i], img[e_i], img[w_i]
    axis2 = w2 - s2
    len2 = float(np.linalg.norm(axis2))
    reach_xy = np.asarray(reach[:2], dtype=float)
    len3 = float(np.linalg.norm(reach_xy))
    angle = _elbow_angle(np.append(s2, 0.0), np.append(e2, 0.0), np.append(w2, 0.0))
    if len2 < 1e-6 or len3 < 1e-6 or np.isnan(angle):
        return np.zeros(2)
    g = _ramp(angle, *trust_2d_angle)
    a2 = axis2 / len2
    off = (a2[0] * (e2 - s2)[1] - a2[1] * (e2 - s2)[0]) / len2
    a3 = reach_xy / len3
    return g * off * len3 * np.array([-a3[1], a3[0]])

def _swing_timing(pts_per_frame: list, min_swing_depth: float = 0.5):
    # returns (height per frame, top frame, impact frame)
    heights = []
    for pts in pts_per_frame:
        h = None
        if all(i in pts for i in (11, 12, 15, 16, 23, 24)):
            hip_y = (pts[23][1] + pts[24][1]) / 2.0
            sh_y  = (pts[11][1] + pts[12][1]) / 2.0
            hands_y = (pts[15][1] + pts[16][1]) / 2.0
            if hip_y - sh_y > 1e-4:
                h = np.array([(hip_y - hands_y) / (hip_y - sh_y)])
        heights.append(h)
    if len(heights) < 10 or sum(h is not None for h in heights) < 10:
        return None
    from scipy.ndimage import gaussian_filter1d, median_filter
    height = np.array([h[0] for h in _interp_vectors(heights)])
    height = gaussian_filter1d(median_filter(height, size=5, mode="nearest"), 2.0)
    peak_before = np.maximum.accumulate(height)
    peak_after  = np.maximum.accumulate(height[::-1])[::-1]
    depth = np.minimum(peak_before, peak_after) - height
    impact = int(np.argmax(depth))
    if depth[impact] < min_swing_depth or impact < 2:
        return None
    return height, int(np.argmax(height[:impact])), impact

def _straighten_arms(
    frames_3d: list,
    image_pts: list = None,
    trust_2d_angle: tuple = (155.0, 165.0),
    min_swing_depth: float = 0.5,
    lead_release: tuple = (0.7, 1.1),
    trail_fold: tuple = (0.2, 0.7),
    trail_unfold: tuple = (0.5, 0.1),
    trail_release: tuple = (0.4, 0.85),
    finish_lift: tuple = (0.8, 1.0),
    lead_fold: tuple = (1.1, 1.3),
) -> list:
    # straighten arms when mediapipe incorrectly bends their depth
    n = len(frames_3d)
    if n < 10:
        return frames_3d
    pts_per_frame = [{l["idx"]: np.array([l["x"], l["y"], l["z"]], dtype=float)
                      for l in frame["landmarks"] if l["x"] is not None}
                     for frame in frames_3d]
    timing = _swing_timing(pts_per_frame, min_swing_depth)
    if timing is None:
        return frames_3d
    height, top, impact = timing
    # find lead arm (the straighter one)
    def mean_angle(side):
        s, e, w = _ARM_CHAINS[side]
        a = [_elbow_angle(pts_per_frame[fi][s], pts_per_frame[fi][e], pts_per_frame[fi][w])
             for fi in range(top + 1)
             if height[fi] >= 0.5 and all(i in pts_per_frame[fi] for i in (s, e, w))]
        a = [x for x in a if not np.isnan(x)]
        return float(np.mean(a)) if len(a) >= 3 else None
    left, right = mean_angle("left"), mean_angle("right")
    if left is None or right is None:
        return frames_3d
    lead, trail = ("left", "right") if left >= right else ("right", "left")

    # correction strength for each arm per frame
    after = height[impact:]
    reached = np.where(after >= height[impact] + 0.95 * (after.max() - height[impact]))[0]
    finish = impact + int(reached[0]) if len(reached) else n - 1
    span = max(1, finish - impact)
    weight = {lead: np.ones(n), trail: np.ones(n)}
    folded = 0.0
    released = 0.0
    for fi in range(n):
        h = float(height[fi])
        if fi <= top:
            folded = max(folded, _ramp(h, *trail_fold))
            weight[trail][fi] = 1.0 - folded
        elif fi <= impact:
            weight[trail][fi] = _ramp(h, *trail_unfold)
        else:
            released = max(released, _ramp(h, *lead_release))
            weight[lead][fi] = 1.0 - released
            weight[trail][fi] = 1.0 - _ramp(fi, finish + trail_release[0] * span,
                                            finish + trail_release[1] * span)

    # move arms to straightened positions
    def arm_len(side):
        s_i, e_i, w_i = _ARM_CHAINS[side]
        lens = [float(np.linalg.norm(p[e_i] - p[s_i]) + np.linalg.norm(p[w_i] - p[e_i]))
                for p in pts_per_frame if all(i in p for i in (s_i, e_i, w_i))]
        return float(np.median(lens)) if lens else 0.0

    full_len = {lead: arm_len(lead), trail: arm_len(trail)}
    hand_idx = [_ARM_CHAINS[s][2] for s in _ARM_CHAINS] + [t for s in _HAND_TIPS for t in _HAND_TIPS[s]]
    for fi in range(n):
        side = lead if weight[lead][fi] >= weight[trail][fi] else trail
        pts, k = pts_per_frame[fi], float(weight[side][fi])
        s_i, w_i = _ARM_CHAINS[side][0], _ARM_CHAINS[side][2]
        if k <= 0.0 or full_len[side] <= 0.0 or s_i not in pts or w_i not in pts:
            continue
        reach = pts[w_i] - pts[s_i]
        d = float(np.linalg.norm(reach))
        if d < 1e-6 or d >= full_len[side]:
            continue
        shift = reach / d * (full_len[side] - d) * k
        for i in hand_idx:
            if i in pts:
                pts[i] = pts[i] + shift
                _set_landmark(frames_3d[fi], i, pts[i])
    for side, w_arr in weight.items():
        s_i, e_i, w_i = _ARM_CHAINS[side]
        for fi in range(n):
            pts, k = pts_per_frame[fi], float(w_arr[fi])
            if k <= 0.0 or not all(i in pts for i in (s_i, e_i, w_i)):
                continue
            upper = float(np.linalg.norm(pts[e_i] - pts[s_i]))
            fore  = float(np.linalg.norm(pts[w_i] - pts[e_i]))
            if upper + fore < 1e-6:
                continue
            straight = pts[s_i] + (pts[w_i] - pts[s_i]) * (upper / (upper + fore))
            straight[:2] += _image_elbow_offset(
                image_pts[fi] if image_pts else None, (s_i, e_i, w_i),
                pts[w_i] - pts[s_i], trust_2d_angle)
            _set_landmark(frames_3d[fi], e_i, pts[e_i] + k * (straight - pts[e_i]))
            pts[e_i] = pts[e_i] + k * (straight - pts[e_i])

    # finish
    past_twelve = np.zeros(n)
    over_head = np.zeros(n)   # likewise, as they climb past `lead_fold`
    for fi in range(impact + 1, n):
        past_twelve[fi] = max(past_twelve[fi - 1], _ramp(float(height[fi]), *finish_lift))
        over_head[fi] = max(over_head[fi - 1], _ramp(float(height[fi]), *lead_fold))
    if image_pts:
        _lift_trail_arm_from_image(frames_3d, pts_per_frame, image_pts, past_twelve,
                                   trail, weight[trail], full_len[trail], hand_idx)
    _bring_lead_arm_in_front(frames_3d, pts_per_frame, image_pts, over_head, lead, trail)
    return frames_3d

def _elbow_circle(s, w, upper: float, fore: float):
    # circle of possible elbow positions around shoulder to wrist line
    reach = w - s
    d = float(np.linalg.norm(reach))
    if d < 1e-6:
        return None
    axis = reach / d
    if d >= upper + fore:
        return s + axis * upper, axis, 0.0
    along = (upper * upper - fore * fore + d * d) / (2.0 * d)
    return s + axis * along, axis, float(np.sqrt(max(upper * upper - along * along, 0.0)))

def _circle_points(centre, axis, radius: float, count: int = 72) -> np.ndarray:
    # plausible elbow positions around the circle
    toward_camera = np.array([0.0, 0.0, -1.0])   # world z grows away from the camera
    front = toward_camera - axis * float(np.dot(toward_camera, axis))
    if np.linalg.norm(front) < 1e-6:
        front = np.cross(axis, [1.0, 0.0, 0.0])
    front = front / np.linalg.norm(front)
    side = np.cross(axis, front)
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return centre + (np.cos(angles)[:, None] * front + np.sin(angles)[:, None] * side) * radius

def _bring_lead_arm_in_front(
    frames_3d: list,
    pts_per_frame: list,
    image_pts: list,
    past_twelve: np.ndarray,
    lead: str,
    trail: str,
    margin: float = 0.08,
) -> None:
    # keep lead arm in front of head and trail arm 
    lengths = {}
    for side in (lead, trail):
        s_i, e_i, w_i = _ARM_CHAINS[side]
        have = [p for p in pts_per_frame if all(i in p for i in (s_i, e_i, w_i))]
        if not have:
            return
        lengths[side] = (float(np.median([np.linalg.norm(p[e_i] - p[s_i]) for p in have])),
                         float(np.median([np.linalg.norm(p[w_i] - p[e_i]) for p in have])))

    ls_i, le_i, lw_i = _ARM_CHAINS[lead]
    ts_i, te_i, tw_i = _ARM_CHAINS[trail]
    for fi, g in enumerate(past_twelve):
        pts = pts_per_frame[fi]
        if g <= 0.0 or not all(i in pts for i in (ls_i, le_i, lw_i, ts_i, te_i, tw_i)):
            continue
        # lead elbow: front half of its circle (world z falls toward the camera).
        circle = _elbow_circle(pts[ls_i], pts[lw_i], *lengths[lead])
        if circle is None:
            continue
        cands = _circle_points(*circle)
        cands = cands[cands[:, 2] <= circle[0][2]]
        if 0 in pts:
            ahead = cands[cands[:, 2] <= pts[0][2] - margin]
            if len(ahead):
                cands = ahead
        img = image_pts[fi] if image_pts else None
        scale = (_image_to_world_scale(img, pts)
                 if img is not None and ls_i in img and le_i in img else None)
        if scale is not None:
            target = pts[ls_i][:2] + (img[le_i] - img[ls_i]) * scale
            lead_elbow = cands[int(np.argmin(np.sum((cands[:, :2] - target) ** 2, axis=1)))]
        else:
            lead_elbow = cands[int(np.argmin(cands[:, 2]))]
        pts[le_i] = pts[le_i] + g * (lead_elbow - pts[le_i])
        _set_landmark(frames_3d[fi], le_i, pts[le_i])
        # trail elbow: the nearest point on its circle that is behind the lead.
        circle = _elbow_circle(pts[ts_i], pts[tw_i], *lengths[trail])
        if circle is None:
            continue
        cands = _circle_points(*circle)
        behind = cands[cands[:, 2] >= pts[le_i][2] + margin]
        if len(behind):
            trail_elbow = behind[int(np.argmin(np.linalg.norm(behind - pts[te_i], axis=1)))]
        else:
            trail_elbow = cands[int(np.argmax(cands[:, 2]))]
        pts[te_i] = pts[te_i] + g * (trail_elbow - pts[te_i])
        _set_landmark(frames_3d[fi], te_i, pts[te_i])

def _lift_trail_arm_from_image(
    frames_3d: list,
    pts_per_frame: list,
    image_pts: list,
    past_twelve: np.ndarray,
    trail: str,
    straight_weight: np.ndarray,
    reach_len: float,
    hand_idx: list,
) -> None:
    # rebuild trail arm's elbow and wrist after once hands past shoulders after impact
    s_i, e_i, w_i = _ARM_CHAINS[trail]
    chain = (s_i, e_i, w_i)
    have = [p for p in pts_per_frame if all(i in p for i in chain)]
    if not have or reach_len <= 0.0:
        return
    upper = float(np.median([np.linalg.norm(p[e_i] - p[s_i]) for p in have]))
    fore = float(np.median([np.linalg.norm(p[w_i] - p[e_i]) for p in have]))
    def lift(a2, b2, length, world_dz, scale):
        v = (b2 - a2) * scale
        m = float(np.linalg.norm(v))
        if m > length:
            v, m = v * (length / m), length
        dz = float(np.sqrt(length * length - m * m))
        return np.array([v[0], v[1], dz if world_dz >= 0 else -dz])
    for fi, g in enumerate(past_twelve):
        pts, img = pts_per_frame[fi], image_pts[fi]
        if g <= 0.0 or img is None or not all(i in pts and i in img for i in chain):
            continue
        scale = _image_to_world_scale(img, pts)
        if scale is None:
            continue
        s = pts[s_i]
        elbow = s + lift(img[s_i], img[e_i], upper, pts[e_i][2] - s[2], scale)
        wrist = elbow + lift(img[e_i], img[w_i], fore, pts[w_i][2] - pts[e_i][2], scale)
        k = float(straight_weight[fi])
        if k > 0.0:
            wrist_str = s + lift(img[s_i], img[w_i], reach_len, pts[w_i][2] - s[2], scale)
            elbow_str = s + (wrist_str - s) * (upper / (upper + fore))
            wrist = k * wrist_str + (1.0 - k) * wrist
            elbow = k * elbow_str + (1.0 - k) * elbow
        shift = g * (wrist - pts[w_i])
        for i in hand_idx:
            if i in pts:
                pts[i] = pts[i] + shift
                _set_landmark(frames_3d[fi], i, pts[i])
        pts[e_i] = pts[e_i] + g * (elbow - pts[e_i])
        _set_landmark(frames_3d[fi], e_i, pts[e_i])

# pelvis turn, slide reconstruction

_SCALE_FIT_IDX = (11, 12, 23, 24, 25, 26, 27, 28)
_IMAGE_EXTRA_IDX = (13, 14, 15, 16)

def _image_to_world_scale(img: dict, pts: dict):
    common = [i for i in _SCALE_FIT_IDX if i in img and i in pts]
    if len(common) < 4:
        return None
    a = np.array([img[i] for i in common])
    b = np.array([pts[i][:2] for i in common])
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    denom = float((a * a).sum())
    return float((a * b).sum()) / denom if denom > 1e-8 else None

def _rebuild_turn(
    pts_per_frame: list,
    image_pts: list,
    scales: list,
    valid: list,
    l_i: int,
    r_i: int,
    width_tol: float,
    sure_depth: float,
    square_pct: float,
    trust_2d: tuple,
):
    # rebuild turn for a left/right landmark pair
    n = len(pts_per_frame)
    have = [fi for fi in valid
            if l_i in pts_per_frame[fi] and r_i in pts_per_frame[fi]
            and l_i in image_pts[fi] and r_i in image_pts[fi]]
    if not have:
        return None
    width = float(np.median([np.linalg.norm(pts_per_frame[fi][r_i] - pts_per_frame[fi][l_i])
                             for fi in have]))
    if width < 1e-4:
        return None

    good_idx = []
    flat = [None] * n # image-plane line, meters
    for fi in have:
        pts, img = pts_per_frame[fi], image_pts[fi]
        world = pts[r_i] - pts[l_i]
        if np.linalg.norm(world) < width_tol * width:
            continue
        v = (img[r_i] - img[l_i]) * scales[fi]
        if float(np.dot(v, world[:2])) < 0:
            v = -v
        flat[fi] = v
        good_idx.append(fi)
    if not good_idx:
        return None
    square_w = float(np.percentile([np.linalg.norm(flat[fi]) for fi in good_idx], square_pct))
    if square_w < 1e-6:
        return None
    def world_depth(fi):
        pts = pts_per_frame[fi]
        return float(np.clip((pts[r_i][2] - pts[l_i][2]) / width, -1.0, 1.0))
    planar = [None] * n # image-plane part of the unit line
    depth = [0.0] * n # |depth| part of the unit line
    sign = [0] * n # +1 / -1 where the world landmarks clearly show it
    for fi in good_idx:
        p = flat[fi] / square_w
        mag = float(np.linalg.norm(p))
        if mag > 1.0:
            p, mag = p / mag, 1.0
        planar[fi] = p
        depth[fi] = float(np.sqrt(1.0 - mag * mag))
        wz = world_depth(fi)
        if abs(wz) >= sure_depth:
            sign[fi] = 1 if wz > 0 else -1

    known = [fi for fi in good_idx if sign[fi] != 0]
    if not known:
        for fi in good_idx:
            sign[fi] = 1 if world_depth(fi) >= 0 else -1
    else:
        for fi in good_idx:
            if fi < known[0]:
                sign[fi] = sign[known[0]]
            elif fi > known[-1]:
                sign[fi] = sign[known[-1]]
        for a, b in zip(known, known[1:]):
            gap = [fi for fi in good_idx if a < fi < b]
            if not gap:
                continue
            if sign[a] == sign[b]:
                for fi in gap:
                    sign[fi] = sign[a]
                continue
            flip = min(gap, key=lambda fi: depth[fi])
            for fi in gap:
                sign[fi] = sign[a] if fi < flip else sign[b]

    dirs = [None] * n
    lo, hi = trust_2d
    for fi in good_idx:
        w = float(np.clip((depth[fi] - lo) / (hi - lo), 0.0, 1.0))
        d = w * sign[fi] * depth[fi] + (1.0 - w) * world_depth(fi)
        p = _unit(planar[fi])
        if p is None:
            p = _unit(pts_per_frame[fi][r_i][:2] - pts_per_frame[fi][l_i][:2])
        if p is None:
            continue
        p = p * float(np.sqrt(max(1.0 - d * d, 0.0)))
        dirs[fi] = np.array([p[0], p[1], d])
    return _interp_vectors(dirs), width

def _repair_collapsed_line(pts_per_frame: list, l_i: int, r_i: int, width_tol: float) -> dict:
    n = len(pts_per_frame)
    have = [fi for fi in range(n) if l_i in pts_per_frame[fi] and r_i in pts_per_frame[fi]]
    if not have:
        return {}
    widths = {fi: float(np.linalg.norm(pts_per_frame[fi][r_i] - pts_per_frame[fi][l_i]))
              for fi in have}
    width = float(np.median(list(widths.values())))
    bad = [fi for fi in have if widths[fi] < width_tol * width]
    if not bad or width < 1e-4:
        return {}
    dirs = [None] * n
    for fi in have:
        if widths[fi] >= width_tol * width:
            dirs[fi] = (pts_per_frame[fi][r_i] - pts_per_frame[fi][l_i]) / widths[fi]
    if all(d is None for d in dirs):
        return {}
    dirs = _interp_vectors(dirs)
    out = {}
    for fi in bad:
        u = _unit(dirs[fi])
        if u is None:
            continue
        mid = (pts_per_frame[fi][l_i] + pts_per_frame[fi][r_i]) / 2.0
        out[fi] = (mid - u * width / 2.0, mid + u * width / 2.0)
    return out

def _get_landmark(frame: dict, idx: int):
    for l in frame["landmarks"]:
        if l["idx"] == idx:
            return None if l["x"] is None else np.array([l["x"], l["y"], l["z"]], dtype=float)
    return None

def _fix_backswing_shoulder_turn(
    frames_3d: list, image_pts: list, scales: list, valid: list,
    width_tol: float, sure_depth: float, square_pct: float, trust_2d: tuple,
) -> None:
    pts_per_frame = [{l["idx"]: np.array([l["x"], l["y"], l["z"]], dtype=float)
                      for l in frame["landmarks"] if l["x"] is not None}
                     for frame in frames_3d]
    timing = _swing_timing(pts_per_frame)
    if timing is None:
        return
    _, top, impact = timing
    rebuilt = _rebuild_turn(pts_per_frame, image_pts, scales, valid, 11, 12,
                            width_tol, sure_depth, square_pct, trust_2d)
    if rebuilt is None:
        return
    dirs, _ = rebuilt
    fade_end = top + max(1, (impact - top) // 2)
    for fi, frame in enumerate(frames_3d):
        k = 1.0 if fi <= top else 1.0 - _ramp(fi, top, fade_end)
        pts = pts_per_frame[fi]
        if k <= 0.0 or 11 not in pts or 12 not in pts or dirs[fi] is None:
            continue
        line = pts[12] - pts[11]
        width = float(np.linalg.norm(line))
        u_world, u_img = _unit(line), _unit(dirs[fi])
        if u_world is None or u_img is None:
            continue
        u = _unit((1.0 - k) * u_world + k * u_img)
        if u is None:
            continue
        mid = (pts[11] + pts[12]) / 2.0
        _set_landmark(frame, 11, mid - u * width / 2.0)
        _set_landmark(frame, 12, mid + u * width / 2.0)

def _fix_shoulder_tilt(
    frames_3d: list, image_pts: list, frames: list, trust_width: tuple = (0.4, 0.7),
) -> None:
    # tilt the shoulder line in the image plane to match the video
    have = [fi for fi in frames
            if image_pts[fi] is not None and 11 in image_pts[fi] and 12 in image_pts[fi]]
    if not have:
        return
    med_w = float(np.median([np.linalg.norm(image_pts[fi][12] - image_pts[fi][11]) for fi in have]))
    if med_w < 1e-6:
        return
    for fi in have:
        frame = frames_3d[fi]
        l, r = _get_landmark(frame, 11), _get_landmark(frame, 12)
        if l is None or r is None:
            continue
        v2 = image_pts[fi][12] - image_pts[fi][11]
        k = _ramp(float(np.linalg.norm(v2)) / med_w, *trust_width)
        line = r - l
        xy = line[:2]
        xy_len = float(np.linalg.norm(xy))
        if k <= 0.0 or xy_len < 1e-6:
            continue
        if float(np.dot(v2, xy)) < 0:
            v2 = -v2
        d = _unit((1.0 - k) * xy / xy_len + k * _unit(v2))
        if d is None:
            continue
        new_line = np.array([d[0] * xy_len, d[1] * xy_len, line[2]])
        mid = (l + r) / 2.0
        _set_landmark(frame, 11, mid - new_line / 2.0)
        _set_landmark(frame, 12, mid + new_line / 2.0)

def _fix_hips_and_shoulders(
    frames_3d: list,
    image_pts: list,
    width_tol: float = 0.75,
    shoulder_width_tol: float = 0.5,
    sure_depth: float = 0.25,
    square_pct: float = 95.0,
    trust_2d: tuple = (0.42, 0.71),
    min_valid_frac: float = 0.1,
) -> list:
    # remeasure how the hips turn and how the body slides from the 2D image,
    lh_i, rh_i = 23, 24
    n = len(frames_3d)
    if n == 0 or len(image_pts) != n:
        return frames_3d
    pts_per_frame = []
    for frame in frames_3d:
        pts_per_frame.append({l["idx"]: np.array([l["x"], l["y"], l["z"]], dtype=float)
                              for l in frame["landmarks"] if l["x"] is not None})
    def tracked(fi):
        img, pts = image_pts[fi], pts_per_frame[fi]
        return (img is not None and lh_i in img and rh_i in img
                and lh_i in pts and rh_i in pts)
    scales = [_image_to_world_scale(image_pts[fi], pts_per_frame[fi]) if tracked(fi) else None
              for fi in range(n)]
    valid = [fi for fi in range(n) if scales[fi] is not None and scales[fi] > 0]
    if len(valid) < max(15, min_valid_frac * n):
        return frames_3d

    # hip turn
    rebuilt = _rebuild_turn(pts_per_frame, image_pts, scales, valid, lh_i, rh_i,
                            width_tol, sure_depth, square_pct, trust_2d)
    if rebuilt is not None:
        dirs, width = rebuilt
        for fi, frame in enumerate(frames_3d):
            pts = pts_per_frame[fi]
            if lh_i not in pts or rh_i not in pts or dirs[fi] is None:
                continue
            u = _unit(dirs[fi])
            if u is None:
                continue
            mid = (pts[lh_i] + pts[rh_i]) / 2.0
            _set_landmark(frame, lh_i, mid - u * width / 2.0)
            _set_landmark(frame, rh_i, mid + u * width / 2.0)

    # shoulder collapse
    for fi, (l_pos, r_pos) in _repair_collapsed_line(pts_per_frame, 11, 12, shoulder_width_tol).items():
        _set_landmark(frames_3d[fi], 11, l_pos)
        _set_landmark(frames_3d[fi], 12, r_pos)
    # shoulder turn through the backswing
    _fix_backswing_shoulder_turn(frames_3d, image_pts, scales, valid,
                                 shoulder_width_tol, sure_depth, square_pct, trust_2d)
    # shoulder tilt
    _fix_shoulder_tilt(frames_3d, image_pts, valid)

    # slide
    scale = float(np.median([scales[fi] for fi in valid]))
    valid_set = set(valid)
    centre_px = [(image_pts[fi][lh_i] + image_pts[fi][rh_i]) / 2.0 if fi in valid_set else None
                 for fi in range(n)]
    origin = centre_px[valid[0]]
    shift = _interp_vectors([None if c is None else np.append((c - origin) * scale, 0.0)
                             for c in centre_px])
    for fi, frame in enumerate(frames_3d):
        if shift[fi] is None:
            continue
        for l in frame["landmarks"]:
            if l["x"] is None:
                continue
            l["x"] = round(l["x"] + float(shift[fi][0]), 5)
            l["y"] = round(l["y"] + float(shift[fi][1]), 5)
    return frames_3d

def extract_y_series(frame_data: list, key: str = "wrist_y") -> np.ndarray:
    values = [f.get(key) for f in frame_data]
    arr = np.array([v if v is not None else np.nan for v in values], dtype=float)
    nans = np.isnan(arr)
    if nans.any():
        idx = np.where(~nans, np.arange(len(arr)), 0)
        np.maximum.accumulate(idx, out=idx)
        arr = arr[idx]
    return arr