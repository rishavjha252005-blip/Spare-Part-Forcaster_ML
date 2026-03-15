# core/pipeline/classification.py
import numpy as np
from spare_parts_forecaster_pro.config.settings import ADI_THRESHOLD, CV2_THRESHOLD
from spare_parts_forecaster_pro.core.types import DemandProfile


def classify_demand(demand: np.ndarray) -> DemandProfile:
    demand = np.asarray(demand, dtype=float)
    non_zero = demand[demand > 0]

    if non_zero.size == 0:
        return DemandProfile(
            category="No Demand", adi=0.0, cv2=0.0,
            zero_rate=1.0, mean_nonzero=0.0, total_periods=len(demand)
        )

    adi = len(demand) / len(non_zero)
    cv2 = (np.std(non_zero) / np.mean(non_zero)) ** 2
    zero_rate = float((demand == 0).mean())
    mean_nonzero = float(np.mean(non_zero))

    if adi <= ADI_THRESHOLD and cv2 <= CV2_THRESHOLD:
        category = "Smooth"
    elif adi <= ADI_THRESHOLD and cv2 > CV2_THRESHOLD:
        category = "Erratic"
    elif adi > ADI_THRESHOLD and cv2 <= CV2_THRESHOLD:
        category = "Intermittent"
    else:
        category = "Lumpy"

    return DemandProfile(
        category=category, adi=round(adi, 4), cv2=round(cv2, 4),
        zero_rate=round(zero_rate, 4), mean_nonzero=round(mean_nonzero, 4),
        total_periods=len(demand)
    )
