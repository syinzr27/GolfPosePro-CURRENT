# estimate golf club position from video

from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
# mediapipe landmark indices
LEFT_EAR, RIGHT_EAR = 7, 8
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_PINKY, RIGHT_PINKY = 17, 18
LEFT_INDEX, RIGHT_INDEX = 19, 20
LEFT_HIP, RIGHT_HIP = 23, 24
try:
    import cv2 # noqa: F401
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
try:
    from ultralytics import YOLOWorld
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
try:
    import mediapipe as mp
    from mediapipe.tasks.python.vision import (
        PoseLandmarker, PoseLandmarkerOptions, RunningMode,
    )
    from mediapipe.tasks.python.core.base_options import BaseOptions
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False

# model paths
_PROJ_ROOT = Path(__file__).parent.parent
_YOLO_MODEL_FP = _PROJ_ROOT / "models" / "yolov8s-world.pt"
_POSE_MODEL_FP = _PROJ_ROOT / "pose_landmarker.task"
_YOLO_PROMPTS = ["golf club", "club", "stick"] # prompt for golf club detection
_YOLO_CONF = 0.03
_YOLO_IOU = 0.45
_YOLO_IMGSZ = 640
_yolo_model = None
def _get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLOWorld(str(_YOLO_MODEL_FP))
        _yolo_model.set_classes(_YOLO_PROMPTS)
    return _yolo_model

# helper funcs
def _get_landmark_pos(landmarks, idx):
    for lm in landmarks:
        if lm["idx"] == idx and lm["x"] is not None:
            return np.array([lm["x"], lm["y"], lm["z"]])
    return None
def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else np.array([0.0, 1.0, 0.0])
def _round_point(p):
    return {"x": round(float(p[0]), 5),
            "y": round(float(p[1]), 5),
            "z": round(float(p[2]), 5)}
def _missing_entry():
    return {"grip": None, "shaft_end": None, "has_club": False}

# public
def estimate_club_positions(
    frames_3d: list,
    phase_ranges: dict = None,
    video_path: str = None,
    use_ai_detection: bool = True,
    physical_shaft_m: float = None,
) -> list:
    if (use_ai_detection
            and video_path
            and Path(video_path).exists()
            and _CV2_AVAILABLE and _YOLO_AVAILABLE and _MP_AVAILABLE
            and _YOLO_MODEL_FP.exists()
            and _POSE_MODEL_FP.exists()):
        try:
            return _estimate_with_ai(frames_3d, video_path, physical_shaft_m)
        except Exception as exc:
            # use alternate method if yolo fails
            import sys
            print(f"[club_estimation] AI detection failed: {exc!r}. "
                  "Falling back to heuristic.", file=sys.stderr)

    return _estimate_with_heuristic(frames_3d, physical_shaft_m)

# ai detection method
def _extract_image_pose(video_path: str):
    # extract 2d pixel coords for ears, wrists, knuckles, shoulders, hips, frame size (mediapipe)
    import cv2
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(_POSE_MODEL_FP)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose_px = []  # list of dicts per frame
    with PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            ts = int(frame_idx * 1000 / fps)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect_for_video(mp_image, ts)
            entry = {"lw": None, "rw": None, "ls": None, "rs": None,
                     "lhip": None, "rhip": None, "lear": None, "rear": None,
                     "lpinky": None, "rpinky": None, "lindex": None, "rindex": None}
            if res.pose_landmarks:
                lms = res.pose_landmarks[0]
                def _xy(i):
                    lm = lms[i]
                    return np.array([lm.x * width, lm.y * height])
                entry["lw"] = _xy(LEFT_WRIST)
                entry["rw"] = _xy(RIGHT_WRIST)
                entry["ls"] = _xy(LEFT_SHOULDER)
                entry["rs"] = _xy(RIGHT_SHOULDER)
                entry["lhip"] = _xy(LEFT_HIP)
                entry["rhip"] = _xy(RIGHT_HIP)
                entry["lear"] = _xy(LEFT_EAR)
                entry["rear"] = _xy(RIGHT_EAR)
                entry["lpinky"] = _xy(LEFT_PINKY)
                entry["rpinky"] = _xy(RIGHT_PINKY)
                entry["lindex"] = _xy(LEFT_INDEX)
                entry["rindex"] = _xy(RIGHT_INDEX)
            pose_px.append(entry)
            frame_idx += 1
    cap.release()
    return pose_px, (width, height)

def _detect_club_boxes(video_path: str):
    # detect club bounding box in each frame (yolo-world)
    import cv2
    model = _get_yolo_model()
    cap = cv2.VideoCapture(video_path)
    boxes = []
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        res = model.predict(
            frame,
            conf=_YOLO_CONF,
            iou=_YOLO_IOU,
            imgsz=_YOLO_IMGSZ,
            verbose=False,
        )[0]
        if res.boxes is None or len(res.boxes) == 0:
            boxes.append(None)
            continue
        confs = res.boxes.conf.cpu().numpy()
        xyxy  = res.boxes.xyxy.cpu().numpy()
        boxes.append((xyxy, confs))
    cap.release()
    return boxes

def _pick_best_box(detections, wrist_mid_px):
    # pick best bounding box for club (closest to the wrist)
    if detections is None or wrist_mid_px is None:
        return None
    xyxy, confs = detections
    best, best_score = None, -1.0
    for box, c in zip(xyxy, confs):
        cx = 0.5 * (box[0] + box[2])
        cy = 0.5 * (box[1] + box[3])
        center = np.array([cx, cy])
        dist = np.linalg.norm(center - wrist_mid_px)
        # Soft proximity prior: clubs are usually within ~3× wrist span of grip
        proximity = 1.0 / (1.0 + dist / 200.0)
        score = float(c) * proximity
        if score > best_score:
            best_score = score
            best = box
    return best

def _bbox_head_endpoint(bbox, wrist_mid_px):
    # decide which end of the bbox is the club head (farthest from wrist)
    x1, y1, x2, y2 = bbox
    corners = np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]], dtype=float)
    dists = np.linalg.norm(corners - wrist_mid_px, axis=1)
    tau = max(0.025 * float(np.hypot(x2 - x1, y2 - y1)), 1e-6)
    w = np.exp((dists - dists.max()) / tau)
    return (w[:, None] * corners).sum(axis=0) / w.sum()

def _interp_missing(series):
    # linearly interpolate missing points in series of 2D or 3D coordinates
    n = len(series)
    valid = [i for i, v in enumerate(series) if v is not None]
    if not valid:
        return series
    out = list(series)
    for i in range(valid[0]):
        out[i] = series[valid[0]].copy()
    for i in range(valid[-1] + 1, n):
        out[i] = series[valid[-1]].copy()
    for a, b in zip(valid, valid[1:]):
        if b - a <= 1:
            continue
        va, vb = series[a], series[b]
        for i in range(a + 1, b):
            t = (i - a) / (b - a)
            out[i] = (1 - t) * va + t * vb
    return out

def _estimate_with_ai(frames_3d: list, video_path: str, physical_shaft_m: float = None) -> list:
    # main club estimation func
    pose_px, _frame_size = _extract_image_pose(video_path)
    yolo_boxes = _detect_club_boxes(video_path)
    n = min(len(frames_3d), len(pose_px), len(yolo_boxes))
    wrist_mid_series = _smooth_points_robust([_hands_px(pose_px[i], fists=False) for i in range(n)])
    # derive grip_3d (wrist midpoint, world) and raw 2D head endpoint (pixels) per frame
    grip_3d_series = []
    head_px_series = []
    px_per_meter_series = []
    forearm_dir_series = []
    for i in range(n):
        world_lms = frames_3d[i].get("landmarks", [])
        lw_w = _get_landmark_pos(world_lms, LEFT_WRIST)
        rw_w = _get_landmark_pos(world_lms, RIGHT_WRIST)
        le_w = _get_landmark_pos(world_lms, LEFT_ELBOW)
        re_w = _get_landmark_pos(world_lms, RIGHT_ELBOW)
        if lw_w is None or rw_w is None:
            grip_3d_series.append(None)
            head_px_series.append(None)
            px_per_meter_series.append(None)
            forearm_dir_series.append(None)
            continue
        grip_3d = (lw_w + rw_w) / 2.0
        # forearm direction (used to infer z)
        if le_w is not None and re_w is not None:
            forearm = _normalize(_normalize(lw_w - le_w) + _normalize(rw_w - re_w))
        else:
            forearm = np.array([0.0, 1.0, 0.0])
        # calculate pixel ratio from wrist span
        lw_px = pose_px[i]["lw"]
        rw_px = pose_px[i]["rw"]
        scale = None
        wrist_mid_px = wrist_mid_series[i]
        if lw_px is not None and rw_px is not None:
            wrist_span_px = np.linalg.norm(lw_px - rw_px)
            wrist_span_meter = np.linalg.norm(lw_w - rw_w)
            # try shoulder span if wrists too close together
            if wrist_span_px < 8.0 or wrist_span_meter < 0.02:
                ls_px = pose_px[i]["ls"]; rs_px = pose_px[i]["rs"]
                ls_w = _get_landmark_pos(world_lms, LEFT_SHOULDER)
                rs_w = _get_landmark_pos(world_lms, RIGHT_SHOULDER)
                if (ls_px is not None and rs_px is not None
                        and ls_w is not None and rs_w is not None):
                    shoulder_span_px = np.linalg.norm(ls_px - rs_px)
                    shoulder_span_meter = np.linalg.norm(ls_w - rs_w)
                    if shoulder_span_px > 1.0 and shoulder_span_meter > 1e-3:
                        scale = shoulder_span_px / shoulder_span_meter
            else:
                scale = wrist_span_px / wrist_span_meter
        # find clubhead pixel position
        head_px = None
        if wrist_mid_px is not None and yolo_boxes[i] is not None:
            bbox = _pick_best_box(yolo_boxes[i], wrist_mid_px)
            if bbox is not None:
                head_px = _bbox_head_endpoint(bbox, wrist_mid_px)
        grip_3d_series.append(grip_3d)
        head_px_series.append(head_px)
        px_per_meter_series.append(scale)
        forearm_dir_series.append(forearm)
    # interpolate missing pixel/scale values
    head_px_series = _interp_missing(head_px_series)
    px_per_meter_series = _interp_missing(
        [None if s is None else np.array([s]) for s in px_per_meter_series]
    )
    # smooth clubhead pixel location
    smoothed_head = _smooth_points_robust(head_px_series)
    # find club length
    body_scale   = _estimate_body_scale(frames_3d)
    fallback_len = body_scale * 1.3 if body_scale > 0 else 1.1
    if physical_shaft_m is not None:
        median_len = float(np.clip(physical_shaft_m, 0.5, 1.5))
    else:
        club_lengths = []
        for i in range(n):
            if (grip_3d_series[i] is None or smoothed_head[i] is None
                    or px_per_meter_series[i] is None
                    or wrist_mid_series[i] is None):
                continue
            wrist_mid_px   = wrist_mid_series[i]
            scale_px_per_m = float(px_per_meter_series[i][0])
            head_meters_xy = (smoothed_head[i] - wrist_mid_px) / scale_px_per_m
            club_lengths.append(np.linalg.norm(head_meters_xy))
        if club_lengths:
            median_len = float(np.clip(float(np.median(club_lengths)), 0.8, 1.6))
        else:
            median_len = fallback_len
    hands_series = _smooth_points_robust([_hands_px(pose_px[i]) for i in range(n)])
    screen_frac = _screen_fractions(frames_3d, pose_px, hands_series, yolo_boxes, n)
    grip_from_head = _grip_from_head(pose_px, hands_series, n)

    # calculate final output with z coord
    out = []
    for i in range(n):
        grip_3d = grip_3d_series[i]
        if grip_3d is None:
            out.append(_missing_entry())
            continue
        if (smoothed_head[i] is not None
                and px_per_meter_series[i] is not None
                and wrist_mid_series[i] is not None):
            wrist_mid_px = wrist_mid_series[i]
            scale_px_per_m = float(px_per_meter_series[i][0])
            dxy_meters = (smoothed_head[i] - wrist_mid_px) / scale_px_per_m
            planar_mag = np.linalg.norm(dxy_meters)
            if planar_mag > 1e-6:
                dxy_meters = dxy_meters * (median_len / planar_mag)
            # z coord
            forearm = forearm_dir_series[i]
            dz = float(forearm[2]) * median_len * 0.35 if forearm is not None else 0.0
            offset = np.array([float(dxy_meters[0]), float(dxy_meters[1]), dz])
            mag = np.linalg.norm(offset)
            if mag > 1e-6:
                offset = offset * (median_len / mag)
            shaft_end = grip_3d + offset
            entry = {
                "grip": _round_point(grip_3d),
                "shaft_end": _round_point(shaft_end),
                "has_club": True,
            }
            if screen_frac[i] is not None:
                entry["screen_frac"] = round(float(screen_frac[i]), 3)
            if grip_from_head[i] is not None:
                entry["grip_from_head"] = {"x": round(float(grip_from_head[i][0]), 4),
                                           "y": round(float(grip_from_head[i][1]), 4)}
            out.append(entry)
            continue

        # old: if failed, use forearm projection
        forearm = forearm_dir_series[i]
        if forearm is None:
            forearm = np.array([0.0, 1.0, 0.0])
        down = np.array([0.0, 1.0, 0.0])
        shaft_dir = _normalize(forearm * 0.6 + down * 0.4)
        shaft_end = grip_3d + shaft_dir * median_len
        out.append({
            "grip": _round_point(grip_3d),
            "shaft_end": _round_point(shaft_end),
            "has_club": True,
        })
    while len(out) < len(frames_3d):
        out.append(_missing_entry())
    return out

def _hands_px(p, apart=0.5, fists=True):
    # calculate pixel location of hands
    if p["lw"] is None or p["rw"] is None:
        return None
    fist = {s: (np.mean([q for q in (p[s + "w"], p.get(s + "pinky"), p.get(s + "index"))
                         if q is not None], axis=0) if fists else p[s + "w"])
            for s in ("l", "r")}
    mid = (fist["l"] + fist["r"]) / 2.0
    if any(p.get(k) is None for k in ("ls", "rs", "lhip", "rhip")):
        return mid
    torso_px = np.linalg.norm((p["ls"] + p["rs"]) / 2.0 - (p["lhip"] + p["rhip"]) / 2.0)
    if np.linalg.norm(p["lw"] - p["rw"]) <= apart * torso_px:
        return mid
    return fist["l"] if p["lw"][1] < p["rw"][1] else fist["r"]

def _screen_fractions(frames_3d, pose_px, hands_series, yolo_boxes, n,
                      ref_percentile=90.0, median_size=9, sigma=2.0,
                      limits=(0.3, 1.0)):
    # returns (visible length of club)/(full club length)
    raw = [None] * n
    for i in range(n):
        p = pose_px[i]
        if (hands_series[i] is None or yolo_boxes[i] is None
                or any(p.get(k) is None for k in ("ls", "rs", "lhip", "rhip"))):
            continue
        lms = frames_3d[i].get("landmarks", [])
        world = [_get_landmark_pos(lms, k)
                 for k in (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)]
        if any(w is None for w in world):
            continue
        bbox = _pick_best_box(yolo_boxes[i], hands_series[i])
        if bbox is None:
            continue
        head = _bbox_head_endpoint(bbox, hands_series[i])
        torso_px = np.linalg.norm((p["ls"] + p["rs"]) / 2.0 - (p["lhip"] + p["rhip"]) / 2.0)
        torso_m = np.linalg.norm(((world[0] + world[1]) / 2.0 - (world[2] + world[3]) / 2.0)[:2])
        if torso_px < 1.0 or torso_m < 1e-3:
            continue
        raw[i] = float(np.linalg.norm(head - hands_series[i]) / torso_px * torso_m)
    idx = [i for i, v in enumerate(raw) if v is not None]
    if len(idx) < 2:
        return [None] * n
    ref = float(np.percentile([raw[i] for i in idx], ref_percentile))
    if ref <= 1e-6:
        return [None] * n
    vals = median_filter(np.array([raw[i] for i in idx]), size=median_size, mode="nearest")
    full = np.interp(np.arange(idx[0], idx[-1] + 1), idx, vals)
    full = gaussian_filter1d(full, sigma, mode="nearest")
    out = [None] * n
    for k, v in enumerate(full):
        out[idx[0] + k] = float(np.clip(v / ref, *limits))
    return out

def _grip_from_head(pose_px, hands_series, n, median_size=5, sigma=2.0):
    # returns the position of the hands relative to the clubhead in torso lengths
    raw = [None] * n
    for i in range(n):
        p = pose_px[i]
        if (hands_series[i] is None
                or any(p.get(k) is None for k in ("lear", "rear", "ls", "rs", "lhip", "rhip"))):
            continue
        torso_px = np.linalg.norm((p["ls"] + p["rs"]) / 2.0 - (p["lhip"] + p["rhip"]) / 2.0)
        if torso_px < 1.0:
            continue
        raw[i] = (hands_series[i] - (p["lear"] + p["rear"]) / 2.0) / torso_px
    return _smooth_points_robust(raw, median_size=median_size, sigma=sigma)

def _smooth_points_robust(series, median_size=5, sigma=1.5):
    idx = [i for i, v in enumerate(series) if v is not None]
    if len(idx) < 2:
        return list(series)
    pts = np.array([series[i] for i in idx], dtype=float)
    pts = median_filter(pts, size=(median_size, 1), mode="nearest")
    pts = gaussian_filter1d(pts, sigma, axis=0, mode="nearest")
    out = [None] * len(series)
    for k, i in enumerate(idx):
        out[i] = pts[k]
    return out

# old: forearm projection
def _estimate_with_heuristic(frames_3d: list, physical_shaft_m: float = None) -> list:
    body_scale = _estimate_body_scale(frames_3d)
    club_length = (physical_shaft_m if physical_shaft_m is not None
                   else (body_scale * 1.8 if body_scale > 0 else 1.1))
    club_data = []
    for frame in frames_3d:
        landmarks = frame.get("landmarks", [])
        if not landmarks:
            club_data.append(_missing_entry())
            continue
        lw = _get_landmark_pos(landmarks, LEFT_WRIST)
        rw = _get_landmark_pos(landmarks, RIGHT_WRIST)
        le = _get_landmark_pos(landmarks, LEFT_ELBOW)
        re = _get_landmark_pos(landmarks, RIGHT_ELBOW)
        if lw is None or rw is None:
            club_data.append(_missing_entry())
            continue
        grip = (lw + rw) / 2.0
        shaft_dir = np.array([0.0, 1.0, 0.0])
        if le is not None and re is not None:
            left_forearm  = _normalize(lw - le)
            right_forearm = _normalize(rw - re)
            forearm_avg   = _normalize(left_forearm + right_forearm)
            down = np.array([0.0, 1.0, 0.0])
            shaft_dir = _normalize(forearm_avg * 0.6 + down * 0.4)
        shaft_end = grip + shaft_dir * club_length
        club_data.append({
            "grip": _round_point(grip),
            "shaft_end": _round_point(shaft_end),
            "has_club": True,
        })
    return club_data

def calculate_club_head_speed(
    club_positions: list,
    actual_fps: float = 480.0,
    physical_shaft_m: float = None,
) -> dict:
    # calculate clubhead speed at impact
    import math
    dt   = 1.0 / actual_fps
    _MPH = 2.23694  # m/s to mph
    grip_pos = []
    head_pos = []
    for cp in club_positions:
        if (cp.get("has_club")
                and cp.get("grip") is not None
                and cp.get("shaft_end") is not None):
            g = cp["grip"]
            s = cp["shaft_end"]
            grip_pos.append(np.array([g["x"], g["y"], g["z"]]))
            head_pos.append(np.array([s["x"], s["y"], s["z"]]))
        else:
            grip_pos.append(None)
            head_pos.append(None)
    n = len(grip_pos)
    apparent_lengths = [
        float(np.linalg.norm(head_pos[i] - grip_pos[i]))
        for i in range(n)
        if grip_pos[i] is not None and head_pos[i] is not None
    ]
    apparent_median = float(np.median(apparent_lengths)) if apparent_lengths else 1.0

    shaft_len = physical_shaft_m if physical_shaft_m is not None else apparent_median

    # gaussian smoothing
    _SIG = 4.0
    _HWIN = 10
    _w = [math.exp(-0.5 * (k / _SIG) ** 2) for k in range(-_HWIN, _HWIN + 1)]
    def _gauss_smooth(series):
        out = [None] * n
        for i in range(n):
            accum, wtot = np.zeros(3), 0.0
            for di, w in enumerate(_w):
                j = i - _HWIN + di
                if 0 <= j < n and series[j] is not None:
                    accum += w * series[j]
                    wtot  += w
            if wtot > 0.1:
                out[i] = accum / wtot
        return out
    smooth_grip = _gauss_smooth(grip_pos)
    # shaft unit-direction vectors, smoothing
    raw_dir = []
    for i in range(n):
        if grip_pos[i] is not None and head_pos[i] is not None:
            d = head_pos[i] - grip_pos[i]
            norm = float(np.linalg.norm(d))
            raw_dir.append(d / norm if norm > 0.01 else None)
        else:
            raw_dir.append(None)
    smooth_dir = _gauss_smooth(raw_dir)
    for i in range(n):
        if smooth_dir[i] is not None:
            m = float(np.linalg.norm(smooth_dir[i]))
            smooth_dir[i] = smooth_dir[i] / m if m > 1e-6 else smooth_dir[i]
    smooth_head = [
        smooth_grip[i] + smooth_dir[i] * shaft_len
        if smooth_grip[i] is not None and smooth_dir[i] is not None
        else None
        for i in range(n)
    ]
    # velocities
    speeds_ms = [None] * n
    for i in range(n):
        p = i > 0 and smooth_head[i - 1] is not None
        nx = i < n-1 and smooth_head[i + 1] is not None
        c  = smooth_head[i] is not None
        if c and p and nx:
            speeds_ms[i] = float(np.linalg.norm(smooth_head[i+1] - smooth_head[i-1]) / (2*dt))
        elif c and nx:
            speeds_ms[i] = float(np.linalg.norm(smooth_head[i+1] - smooth_head[i]) / dt)
        elif c and p:
            speeds_ms[i] = float(np.linalg.norm(smooth_head[i] - smooth_head[i-1]) / dt)
    speeds_mph = [round(s * _MPH, 1) if s is not None else None for s in speeds_ms]
    valid = [(i, s) for i, s in enumerate(speeds_ms) if s is not None]
    if not valid:
        return {
            "impact_frame": None,
            "speed_mph": None,
            "speed_ms": None,
            "apparent_shaft_m": round(apparent_median, 3),
            "frame_speeds_mph": speeds_mph,
        }
    # let impact be frame w/ peak speed
    impact_frame, impact_speed_ms = max(valid, key=lambda x: x[1])
    return {
        "impact_frame": impact_frame,
        "speed_mph": round(impact_speed_ms * _MPH, 1),
        "speed_ms": round(impact_speed_ms, 2),
        "apparent_shaft_m": round(apparent_median, 3),
        "frame_speeds_mph": speeds_mph,
    }

def calculate_attack_angle(
    club_positions: list,
    actual_fps: float = 480.0,
    physical_shaft_m: float = None,
) -> dict:
    import math
    dt = 1.0 / actual_fps
    grip_pos = []
    head_pos = []
    for cp in club_positions:
        if (cp.get("has_club") and cp.get("grip") is not None and cp.get("shaft_end") is not None):
            g = cp["grip"]
            s = cp["shaft_end"]
            grip_pos.append(np.array([g["x"], g["y"], g["z"]]))
            head_pos.append(np.array([s["x"], s["y"], s["z"]]))
        else:
            grip_pos.append(None)
            head_pos.append(None)
    n = len(grip_pos)
    apparent_lengths = [
        float(np.linalg.norm(head_pos[i] - grip_pos[i]))
        for i in range(n)
        if grip_pos[i] is not None and head_pos[i] is not None
    ]
    apparent_median = float(np.median(apparent_lengths)) if apparent_lengths else 1.0
    shaft_len = physical_shaft_m if physical_shaft_m is not None else apparent_median
    # gaussian smoothing
    _SIG = 4.0
    _HWIN = 10
    _w = [math.exp(-0.5 * (k / _SIG) ** 2) for k in range(-_HWIN, _HWIN + 1)]
    def _gauss_smooth(series):
        out = [None] * n
        for i in range(n):
            accum, wtot = np.zeros(3), 0.0
            for di, w in enumerate(_w):
                j = i - _HWIN + di
                if 0 <= j < n and series[j] is not None:
                    accum += w * series[j]
                    wtot += w
            if wtot > 0.1:
                out[i] = accum / wtot
        return out
    smooth_grip = _gauss_smooth(grip_pos)
    raw_dir = []
    for i in range(n):
        if grip_pos[i] is not None and head_pos[i] is not None:
            d = head_pos[i] - grip_pos[i]
            norm = float(np.linalg.norm(d))
            raw_dir.append(d / norm if norm > 0.01 else None)
        else:
            raw_dir.append(None)
    smooth_dir = _gauss_smooth(raw_dir)
    for i in range(n):
        if smooth_dir[i] is not None:
            m = float(np.linalg.norm(smooth_dir[i]))
            smooth_dir[i] = smooth_dir[i] / m if m > 1e-6 else smooth_dir[i]
    smooth_head = [
        smooth_grip[i] + smooth_dir[i] * shaft_len
        if smooth_grip[i] is not None and smooth_dir[i] is not None
        else None
        for i in range(n)
    ]
    # speeds
    speeds_ms = [None] * n
    for i in range(n):
        p = i > 0 and smooth_head[i - 1] is not None
        nx = i < n-1 and smooth_head[i + 1] is not None
        c = smooth_head[i] is not None
        if c and p and nx:
            speeds_ms[i] = float(np.linalg.norm(smooth_head[i+1] - smooth_head[i-1]) / (2*dt))
        elif c and nx:
            speeds_ms[i] = float(np.linalg.norm(smooth_head[i+1] - smooth_head[i]) / dt)
        elif c and p:
            speeds_ms[i] = float(np.linalg.norm(smooth_head[i] - smooth_head[i-1]) / dt)
    # vertical angles
    frame_angles = [None] * n
    for i in range(n):
        p = i > 0 and smooth_head[i - 1] is not None
        nx = i < n-1 and smooth_head[i + 1] is not None
        c = smooth_head[i] is not None
        if not c:
            continue
        if p and nx:
            vel = (smooth_head[i + 1] - smooth_head[i - 1]) / (2 * dt)
        elif nx:
            vel = (smooth_head[i + 1] - smooth_head[i]) / dt
        elif p:
            vel = (smooth_head[i] - smooth_head[i - 1]) / dt
        else:
            continue
        vx, vy, vz = vel
        horiz = math.sqrt(vx ** 2 + vz ** 2)
        if horiz > 1e-6:
            frame_angles[i] = round(math.degrees(math.atan2(-vy, horiz)), 2)
    valid = [(i, s) for i, s in enumerate(speeds_ms) if s is not None]
    if not valid:
        return {
            "attack_angle_deg": None,
            "impact_frame": None,
            "apparent_shaft_m": round(apparent_median, 3),
            "frame_angles_deg": frame_angles,
        }
    impact_frame = max(valid, key=lambda x: x[1])[0]
    return {
        "attack_angle_deg": frame_angles[impact_frame],
        "impact_frame": impact_frame,
        "apparent_shaft_m": round(apparent_median, 3),
        "frame_angles_deg": frame_angles,
    }

def calculate_club_path(
    club_positions: list,
    actual_fps: float = 480.0,
    physical_shaft_m: float = None,
    frames_3d: list = None,
) -> dict:
    # angle of club path
    import math
    dt = 1.0 / actual_fps
    grip_pos = []
    head_pos = []
    for cp in club_positions:
        if (cp.get("has_club")
                and cp.get("grip") is not None
                and cp.get("shaft_end") is not None):
            g = cp["grip"]
            s = cp["shaft_end"]
            grip_pos.append(np.array([g["x"], g["y"], g["z"]]))
            head_pos.append(np.array([s["x"], s["y"], s["z"]]))
        else:
            grip_pos.append(None)
            head_pos.append(None)
    n = len(grip_pos)
    apparent_lengths = [
        float(np.linalg.norm(head_pos[i] - grip_pos[i]))
        for i in range(n)
        if grip_pos[i] is not None and head_pos[i] is not None
    ]
    apparent_median = float(np.median(apparent_lengths)) if apparent_lengths else 1.0
    shaft_len = physical_shaft_m if physical_shaft_m is not None else apparent_median
    _SIG = 4.0
    _HWIN = 10
    _w = [math.exp(-0.5 * (k / _SIG) ** 2) for k in range(-_HWIN, _HWIN + 1)]
    def _gauss_smooth(series):
        out = [None] * n
        for i in range(n):
            accum, wtot = np.zeros(3), 0.0
            for di, w in enumerate(_w):
                j = i - _HWIN + di
                if 0 <= j < n and series[j] is not None:
                    accum += w * series[j]
                    wtot  += w
            if wtot > 0.1:
                out[i] = accum / wtot
        return out
    smooth_grip = _gauss_smooth(grip_pos)
    raw_dir = []
    for i in range(n):
        if grip_pos[i] is not None and head_pos[i] is not None:
            d = head_pos[i] - grip_pos[i]
            norm = float(np.linalg.norm(d))
            raw_dir.append(d / norm if norm > 0.01 else None)
        else:
            raw_dir.append(None)
    smooth_dir = _gauss_smooth(raw_dir)
    for i in range(n):
        if smooth_dir[i] is not None:
            m = float(np.linalg.norm(smooth_dir[i]))
            smooth_dir[i] = smooth_dir[i] / m if m > 1e-6 else smooth_dir[i]
    smooth_head = [
        smooth_grip[i] + smooth_dir[i] * shaft_len
        if smooth_grip[i] is not None and smooth_dir[i] is not None
        else None
        for i in range(n)
    ]
    # speeds
    speeds_ms = [None] * n
    for i in range(n):
        p = i > 0 and smooth_head[i - 1] is not None
        nx = i < n-1 and smooth_head[i + 1] is not None
        c = smooth_head[i] is not None
        if c and p and nx:
            speeds_ms[i] = float(np.linalg.norm(smooth_head[i+1] - smooth_head[i-1]) / (2*dt))
        elif c and nx:
            speeds_ms[i] = float(np.linalg.norm(smooth_head[i+1] - smooth_head[i]) / dt)
        elif c and p:
            speeds_ms[i] = float(np.linalg.norm(smooth_head[i] - smooth_head[i-1]) / dt)
    valid = [(i, s) for i, s in enumerate(speeds_ms) if s is not None]
    if not valid:
        return {
            "club_path_deg": None,
            "impact_frame": None,
            "apparent_shaft_m": round(apparent_median, 3),
            "frame_paths_deg": [None] * n,
        }
    impact_frame = max(valid, key=lambda x: x[1])[0]
    # club path angle calc
    # atan2(vz, vx): angle of horizontal velocity from +x axis toward +z
    frame_paths = [None] * n
    for i in range(n):
        p = i > 0 and smooth_head[i - 1] is not None
        nx = i < n-1 and smooth_head[i + 1] is not None
        c = smooth_head[i] is not None
        if not c:
            continue
        if p and nx:
            vel = (smooth_head[i + 1] - smooth_head[i - 1]) / (2 * dt)
        elif nx:
            vel = (smooth_head[i + 1] - smooth_head[i]) / dt
        elif p:
            vel = (smooth_head[i] - smooth_head[i - 1]) / dt
        else:
            continue
        vx, _vy, vz = vel
        horiz = math.sqrt(vx ** 2 + vz ** 2)
        if horiz < 1e-6:
            continue
        frame_paths[i] = round(math.degrees(math.atan2(vz, vx)), 2)
    return {
        "club_path_deg": frame_paths[impact_frame],
        "impact_frame": impact_frame,
        "apparent_shaft_m": round(apparent_median, 3),
        "frame_paths_deg": frame_paths,
    }

def calculate_swing_plane(
    club_positions: list,
    actual_fps: float = 480.0,
    physical_shaft_m: float = None,
) -> dict:
    # angle of the plane the clubhead travels through
    import math
    dt = 1.0 / actual_fps
    grip_pos = []
    head_pos = []
    for cp in club_positions:
        if (cp.get("has_club")
                and cp.get("grip") is not None
                and cp.get("shaft_end") is not None):
            g = cp["grip"]
            s = cp["shaft_end"]
            grip_pos.append(np.array([g["x"], g["y"], g["z"]]))
            head_pos.append(np.array([s["x"], s["y"], s["z"]]))
        else:
            grip_pos.append(None)
            head_pos.append(None)
    n = len(grip_pos)
    apparent_lengths = [
        float(np.linalg.norm(head_pos[i] - grip_pos[i]))
        for i in range(n)
        if grip_pos[i] is not None and head_pos[i] is not None
    ]
    apparent_median = float(np.median(apparent_lengths)) if apparent_lengths else 1.0
    shaft_len = physical_shaft_m if physical_shaft_m is not None else apparent_median
    _SIG = 4.0
    _HWIN = 10
    _w = [math.exp(-0.5 * (k / _SIG) ** 2) for k in range(-_HWIN, _HWIN + 1)]
    def _gauss_smooth(series):
        out = [None] * n
        for i in range(n):
            accum, wtot = np.zeros(3), 0.0
            for di, w in enumerate(_w):
                j = i - _HWIN + di
                if 0 <= j < n and series[j] is not None:
                    accum += w * series[j]
                    wtot  += w
            if wtot > 0.1:
                out[i] = accum / wtot
        return out
    smooth_grip = _gauss_smooth(grip_pos)
    raw_dir = []
    for i in range(n):
        if grip_pos[i] is not None and head_pos[i] is not None:
            d = head_pos[i] - grip_pos[i]
            norm = float(np.linalg.norm(d))
            raw_dir.append(d / norm if norm > 0.01 else None)
        else:
            raw_dir.append(None)
    smooth_dir = _gauss_smooth(raw_dir)
    for i in range(n):
        if smooth_dir[i] is not None:
            m = float(np.linalg.norm(smooth_dir[i]))
            smooth_dir[i] = smooth_dir[i] / m if m > 1e-6 else smooth_dir[i]
    smooth_head = [
        smooth_grip[i] + smooth_dir[i] * shaft_len
        if smooth_grip[i] is not None and smooth_dir[i] is not None
        else None
        for i in range(n)
    ]
    # speeds
    speeds_ms = [None] * n
    for i in range(n):
        p = i > 0 and smooth_head[i - 1] is not None
        nx = i < n-1 and smooth_head[i + 1] is not None
        c = smooth_head[i] is not None
        if c and p and nx:
            speeds_ms[i] = float(np.linalg.norm(smooth_head[i+1] - smooth_head[i-1]) / (2*dt))
        elif c and nx:
            speeds_ms[i] = float(np.linalg.norm(smooth_head[i+1] - smooth_head[i]) / dt)
        elif c and p:
            speeds_ms[i] = float(np.linalg.norm(smooth_head[i] - smooth_head[i-1]) / dt)
    valid = [(i, s) for i, s in enumerate(speeds_ms) if s is not None]
    if not valid:
        return {
            "swing_plane_deg": None,
            "impact_frame": None,
            "n_points": 0,
            "apparent_shaft_m": round(apparent_median, 3),
        }
    impact_frame = max(valid, key=lambda x: x[1])[0]

    # 2D ellipse projection of swing plane
    pts_xy = np.array([smooth_head[i][:2] for i in range(n) if smooth_head[i] is not None])
    if len(pts_xy) < 5:
        return {
            "swing_plane_deg": None,
            "impact_frame": impact_frame,
            "n_points": int(len(pts_xy)),
            "apparent_shaft_m": round(apparent_median, 3),
        }
    centroid = pts_xy.mean(axis=0)
    sigma = np.linalg.svd(pts_xy - centroid, compute_uv=False)
    sigma_major, sigma_minor = float(sigma[0]), float(sigma[1])
    if sigma_major < 1e-9:
        return {
            "swing_plane_deg": None,
            "impact_frame": impact_frame,
            "n_points": int(len(pts_xy)),
            "apparent_shaft_m": round(apparent_median, 3),
        }
    ratio = min(1.0, sigma_minor / sigma_major)
    swing_plane_deg = round(math.degrees(math.asin(ratio)), 1)
    return {
        "swing_plane_deg": swing_plane_deg,
        "impact_frame": impact_frame,
        "n_points": int(len(pts_xy)),
        "apparent_shaft_m": round(apparent_median, 3),
    }

def _estimate_body_scale(frames_3d: list) -> float:
    for frame in frames_3d[:30]:
        landmarks = frame.get("landmarks", [])
        ls = _get_landmark_pos(landmarks, LEFT_SHOULDER)
        lh = _get_landmark_pos(landmarks, LEFT_HIP)
        if ls is not None and lh is not None:
            return float(np.linalg.norm(ls - lh))
    return 0.0