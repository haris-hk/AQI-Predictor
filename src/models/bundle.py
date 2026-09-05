"""The deployed artefact.

One bundle holds the six fitted regressors (2 targets x 3 horizons), the exact
feature list they were trained on, the residual quantiles that produce the
prediction intervals, and the provenance metadata. Serving loads this and
nothing else, so there is no way for the app to assemble a different feature
set than the one the models were fitted on.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import ARTIFACT_DIR, HORIZONS, TARGETS, band_for

log = logging.getLogger(__name__)

BUNDLE_FILE = "model_bundle.joblib"
METADATA_FILE = "model_metadata.json"


@dataclass
class ModelBundle:
    models: dict[tuple[str, int], Any] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)
    residual_quantiles: dict[str, dict[str, float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ io
    def save(self, directory: Path | None = None) -> Path:
        directory = Path(directory or ARTIFACT_DIR)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "models": {f"{t}|{h}": m for (t, h), m in self.models.items()},
                "feature_names": self.feature_names,
                "residual_quantiles": self.residual_quantiles,
                "metadata": self.metadata,
            },
            directory / BUNDLE_FILE,
            compress=3,
        )
        (directory / METADATA_FILE).write_text(
            json.dumps(self.metadata, indent=2, default=str))
        log.info("saved bundle to %s", directory / BUNDLE_FILE)
        return directory / BUNDLE_FILE

    @classmethod
    def load(cls, directory: Path | None = None) -> "ModelBundle | None":
        path = Path(directory or ARTIFACT_DIR) / BUNDLE_FILE
        if not path.exists():
            log.warning("no model bundle at %s", path)
            return None
        raw = joblib.load(path)
        models = {}
        for key, model in raw["models"].items():
            target, horizon = key.split("|")
            models[(target, int(horizon))] = model
        return cls(models, raw["feature_names"],
                   raw.get("residual_quantiles", {}), raw.get("metadata", {}))

    # ------------------------------------------------------------- inference
    def align(self, X: pd.DataFrame) -> pd.DataFrame:
        """Force the serving frame onto the training feature list.

        Missing columns are added as NaN and extras dropped, so a feature added
        upstream after the model was trained cannot silently reorder the matrix
        and corrupt every prediction.
        """
        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            log.warning("%d training features missing at serving time: %s",
                        len(missing), missing[:8])
        aligned = X.reindex(columns=self.feature_names)
        return aligned

    def predict_row(self, features: pd.DataFrame) -> pd.DataFrame:
        """Forecast every (target, horizon) for the latest row of `features`."""
        aligned = self.align(features).iloc[[-1]]
        as_of = features.index[-1]
        rows = []
        for h in HORIZONS:
            target_date = pd.Timestamp(as_of) + pd.Timedelta(days=h)
            row: dict[str, Any] = {
                "as_of_date": pd.Timestamp(as_of),
                "target_date": target_date,
                "horizon": h,
            }
            for target in TARGETS:
                model = self.models.get((target, h))
                if model is None:
                    continue
                value = float(np.clip(model.predict(aligned)[0], 0, 500))
                row[target] = value
                q = self.residual_quantiles.get(f"{target}|{h}", {})
                if q:
                    # Residuals are y - yhat, so the interval is the prediction
                    # plus the residual quantiles.
                    row[f"{target}_lower"] = float(np.clip(value + q.get("q10", 0), 0, 500))
                    row[f"{target}_upper"] = float(np.clip(value + q.get("q90", 0), 0, 500))
            if "aqi_mean" in row:
                band = band_for(row.get("aqi_max", row["aqi_mean"]))
                row["band"] = band.label
                row["band_color"] = band.color
                row["guidance"] = band.guidance
            rows.append(row)
        out = pd.DataFrame(rows)
        if not out.empty:
            out = out.set_index("target_date")
            out["predicted_at"] = datetime.now(timezone.utc)
            out["model_version"] = self.metadata.get("trained_at", "unknown")
        return out


def load_metadata(directory: Path | None = None) -> dict:
    path = Path(directory or ARTIFACT_DIR) / METADATA_FILE
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
