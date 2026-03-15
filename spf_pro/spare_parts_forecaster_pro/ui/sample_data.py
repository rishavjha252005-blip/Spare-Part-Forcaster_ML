# ui/sample_data.py
"""Generate a sample dataset so users can try the app without uploading a file."""
import numpy as np
import pandas as pd
import io


def generate_sample_csv() -> bytes:
    np.random.seed(42)
    periods = 60
    dates = pd.date_range(start="2020-01-01", periods=periods, freq="W")
    records = []

    parts = {
        "PUMP-001":  dict(pattern="intermittent", base=5,  noise=2),
        "SEAL-202":  dict(pattern="lumpy",        base=15, noise=8),
        "VALVE-033": dict(pattern="smooth",       base=10, noise=1),
        "BEAR-104":  dict(pattern="erratic",      base=8,  noise=6),
        "FILT-055":  dict(pattern="intermittent", base=3,  noise=2),
    }

    for part_id, cfg in parts.items():
        for i, date in enumerate(dates):
            if cfg["pattern"] == "smooth":
                d = max(0, int(np.random.normal(cfg["base"], cfg["noise"])))
            elif cfg["pattern"] == "erratic":
                d = max(0, int(np.random.normal(cfg["base"], cfg["noise"])))
                if np.random.rand() < 0.3:
                    d = int(d * np.random.uniform(2, 4))
            elif cfg["pattern"] == "intermittent":
                d = int(np.random.poisson(cfg["base"])) if np.random.rand() < 0.4 else 0
            else:  # lumpy
                d = int(np.random.poisson(cfg["base"]) * np.random.choice([0,0,0,1,3,5])) 
            records.append({"date": date.strftime("%Y-%m-%d"), "part_id": part_id, "demand": d})

    df = pd.DataFrame(records)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()
