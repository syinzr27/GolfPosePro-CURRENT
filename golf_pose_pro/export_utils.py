# export results to csv, json

import csv
import json
import io
from pathlib import Path
import numpy as np

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def export_to_csv(
    frame_data: list,
    phase_ranges: dict,
    output_path: str = None,
    velocity_data: dict = None,
    angle_data: dict = None,
    angular_velocity_data: dict = None,
) -> bytes:
    if not frame_data:
        return b""
    sample = frame_data[0]
    landmark_keys = [k for k in sample if k != "frame_idx"]
    extra_keys = []
    if velocity_data:
        extra_keys += sorted(velocity_data.keys())
    if angle_data:
        extra_keys += sorted(angle_data.keys())
    if angular_velocity_data:
        extra_keys += [f"{k}_angular_vel" for k in sorted(angular_velocity_data.keys())]
    fieldnames = ["frame_idx", "phase"] + landmark_keys + extra_keys
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for i, row in enumerate(frame_data):
        frame_idx = row["frame_idx"]
        phase = _get_phase(frame_idx, phase_ranges)
        out_row = {"frame_idx": frame_idx, "phase": phase}
        for k in landmark_keys:
            v = row.get(k)
            out_row[k] = round(v, 3) if v is not None else ""
        # velocity cols
        if velocity_data:
            for k, arr in velocity_data.items():
                out_row[k] = round(float(arr[i]), 2) if i < len(arr) else ""
        # angle cols
        if angle_data:
            for k, arr in angle_data.items():
                val = arr[i] if i < len(arr) else np.nan
                out_row[k] = round(float(val), 2) if not np.isnan(val) else ""
        # angular velocity cols
        if angular_velocity_data:
            for k, arr in angular_velocity_data.items():
                col = f"{k}_angular_vel"
                val = arr[i] if i < len(arr) else np.nan
                out_row[col] = round(float(val), 2) if not np.isnan(val) else ""
        writer.writerow(out_row)
    csv_bytes = buf.getvalue().encode("utf-8")
    if output_path:
        Path(output_path).write_bytes(csv_bytes)
    return csv_bytes

def export_to_json(
    frame_data: list,
    phase_ranges: dict,
    swing_start: int,
    swing_end: int,
    similarity_score: float = None,
    output_path: str = None,
    velocity_data: dict = None,
    angle_data: dict = None,
    angular_velocity_data: dict = None,
    peak_velocities: dict = None,
) -> bytes:
    phases_summary = {}
    for name, (s, e) in phase_ranges.items():
        phases_summary[name] = {
            "start": int(s),
            "end": int(e),
            "duration_frames": int(e - s),
        }
    frames_out = []
    for i, row in enumerate(frame_data):
        frame_idx = row["frame_idx"]
        out = {
            "frame_idx": frame_idx,
            "phase": _get_phase(frame_idx, phase_ranges),
        }
        for k, v in row.items():
            if k != "frame_idx":
                out[k] = round(v, 3) if v is not None else None
        if velocity_data:
            for k, arr in velocity_data.items():
                out[k] = round(float(arr[i]), 2) if i < len(arr) else None
        if angle_data:
            for k, arr in angle_data.items():
                val = arr[i] if i < len(arr) else np.nan
                out[k] = round(float(val), 2) if not np.isnan(val) else None
        if angular_velocity_data:
            for k, arr in angular_velocity_data.items():
                val = arr[i] if i < len(arr) else np.nan
                out[f"{k}_angular_vel"] = round(float(val), 2) if not np.isnan(val) else None
        frames_out.append(out)
    payload = {
        "metadata": {
            "generator": "GolfPosePro",
            "total_frames": len(frame_data),
        },
        "swing_window": {
            "start": int(swing_start),
            "end": int(swing_end),
            "duration_frames": int(swing_end - swing_start),
        },
        "similarity_score": similarity_score,
        "phases": phases_summary,
        "frames": frames_out,
    }
    json_bytes = json.dumps(payload, indent=2, cls=_NumpyEncoder).encode("utf-8")
    # kinematics
    if peak_velocities:
        payload["kinematics"] = {
            "peak_velocities": {
                phase: {"peak_speed_px_per_s": round(v["peak_speed"], 2),
                        "peak_frame": v["peak_frame"]}
                for phase, v in peak_velocities.items()
            }
        }
        json_bytes = json.dumps(payload, indent=2, cls=_NumpyEncoder).encode("utf-8")
    if output_path:
        Path(output_path).write_bytes(json_bytes)
    return json_bytes

def export_3d_json(
    frames_3d: list,
    club_data: list,
    phase_ranges: dict,
    output_path: str = None,
) -> bytes:
    phases_summary = {}
    for name, (s, e) in phase_ranges.items():
        phases_summary[name] = {"start": int(s), "end": int(e)}
    frames_out = []
    for i, frame in enumerate(frames_3d):
        entry = {
            "frame_idx": frame["frame_idx"],
            "timestamp_ms": frame.get("timestamp_ms", 0),
            "landmarks": frame.get("landmarks", []),
        }
        if i < len(club_data):
            entry["club"] = club_data[i]
        frames_out.append(entry)
    payload = {
        "metadata": {
            "generator": "GolfPosePro",
            "format": "3d_landmarks",
            "total_frames": len(frames_3d),
        },
        "phases": phases_summary,
        "frames": frames_out,
    }
    json_bytes = json.dumps(payload, cls=_NumpyEncoder).encode("utf-8")
    if output_path:
        Path(output_path).write_bytes(json_bytes)
    return json_bytes

def _get_phase(frame_idx: int, phase_ranges: dict) -> str:
    for name, (s, e) in phase_ranges.items():
        if s <= frame_idx <= e:
            return name
    return ""