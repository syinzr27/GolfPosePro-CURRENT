# three.js for 3D golf pose viewport

import base64
import json
from pathlib import Path
from .smoothing import smooth_club_for_display, smooth_landmarks_3d_for_display

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_MODELS_DIR = Path(__file__).parent.parent / "models"

def _model_data_url(filename: str) -> str:
    path = _MODELS_DIR / filename
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:model/gltf-binary;base64,{data}"

def _finish_range(phase_ranges: dict):
    # identify finish of swing for special smoothing
    rng = (phase_ranges or {}).get("Follow Through")
    return (int(rng[0]), int(rng[1])) if rng else None

def build_3d_viewer_html(
    frames_3d: list,
    club_data: list,
    phase_ranges: dict,
    phase_colors: dict,
    mode: str = "replay",
    comparison_pose: list = None,
    comparison_club: list = None,
    dtw_alignment: dict = None,
    comparison_phase_ranges: dict = None,
    height: int = 600,
) -> str:
    template_path = _TEMPLATES_DIR / "viewer_3d.html"
    html = template_path.read_text(encoding="utf-8")

    phase_data = {}
    for name, (s, e) in phase_ranges.items():
        phase_data[name] = {"start": int(s), "end": int(e)}
    comp_phase_data = {}
    for name, (s, e) in (comparison_phase_ranges or {}).items():
        comp_phase_data[name] = {"start": int(s), "end": int(e)}

    dtw_js = "null"
    if dtw_alignment:
        dtw_js = json.dumps({
            k: [int(v[0]), int(v[1]), float(v[2]) if v[2] is not None else 0] if len(v) >= 3 else [int(v[0]), int(v[1]), 0]
            for k, v in dtw_alignment.items()
        })

    html = html.replace("__MODEL_URL__", _model_data_url("Xbot.glb"))
    finish = _finish_range(phase_ranges)
    html = html.replace("__POSE_DATA__", json.dumps(smooth_landmarks_3d_for_display(frames_3d, finish=finish)))
    html = html.replace("__CLUB_DATA__", json.dumps(smooth_club_for_display(club_data, finish=finish)))
    html = html.replace("__PHASE_DATA__", json.dumps(phase_data))
    html = html.replace("__COMPARISON_PHASE_DATA__", json.dumps(comp_phase_data))
    html = html.replace("__PHASE_COLORS__", json.dumps(phase_colors))
    html = html.replace("__MODE__", mode)
    html = html.replace("__COMPARISON_POSE__",
                        json.dumps(smooth_landmarks_3d_for_display(
                            comparison_pose or [], finish=_finish_range(comparison_phase_ranges))))
    html = html.replace("__COMPARISON_CLUB__", json.dumps(smooth_club_for_display(
        comparison_club or [], finish=_finish_range(comparison_phase_ranges))))
    html = html.replace("__DTW_ALIGNMENT__", dtw_js)
    html = html.replace("__HEIGHT__", str(height))
    return html

def build_live_tracking_html(height: int = 650) -> str:
    template_path = _TEMPLATES_DIR / "live_tracking.html"
    html = template_path.read_text(encoding="utf-8")
    html = html.replace("__HEIGHT__", str(height))
    return html