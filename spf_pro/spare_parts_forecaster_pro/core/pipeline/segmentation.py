# core/pipeline/segmentation.py
import numpy as np
import pandas as pd
from spare_parts_forecaster_pro.core.types import ABCXYZSegment

# Service-level z-scores by segment (higher value = tighter policy)
_SEGMENT_Z: dict[str, float] = {
    "AX": 2.05, "AY": 1.88, "AZ": 1.75,
    "BX": 1.65, "BY": 1.48, "BZ": 1.28,
    "CX": 1.28, "CY": 1.04, "CZ": 0.84,
}


def classify_abc(total_demand: float, all_demands: list[float]) -> str:
    """Classify a part into A/B/C by cumulative demand value."""
    sorted_d = sorted(all_demands, reverse=True)
    cumsum = np.cumsum(sorted_d)
    total = cumsum[-1]
    if total == 0:
        return "C"
    rank_cum = cumsum[sorted_d.index(total_demand)] / total if total_demand in sorted_d else 1.0
    if rank_cum <= 0.70:
        return "A"
    elif rank_cum <= 0.90:
        return "B"
    return "C"


def classify_xyz(cv2: float) -> str:
    """Classify a part into X/Y/Z by squared CV."""
    if cv2 <= 0.25:
        return "X"
    elif cv2 <= 1.00:
        return "Y"
    return "Z"


def get_segment(total_demand: float, cv2: float, all_demands: list[float] | None = None) -> ABCXYZSegment:
    all_d = all_demands if all_demands else [total_demand]
    abc = classify_abc(total_demand, all_d)
    xyz = classify_xyz(cv2)
    label = abc + xyz
    return ABCXYZSegment(
        abc=abc, xyz=xyz, label=label,
        service_level_z=_SEGMENT_Z.get(label, 1.65),
    )
