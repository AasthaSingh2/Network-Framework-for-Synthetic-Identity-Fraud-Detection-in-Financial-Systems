"""Read-only integration helpers for the saved application and graph outputs."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pandas as pd

APPLICATION_REQUIRED_COLUMNS = {
    "applicant_id",
    "identity_coherence_score",
    "identity_risk_from_ics",
    "xgb_proba",
    "final_identity_risk_score",
    "risk_category",
}
GRAPH_REQUIRED_COLUMNS = {
    "node_id",
    "out_degree",
    "in_degree",
    "total_degree",
    "ics",
    "ics_risk",
    "fraud_probability",
    "predicted_fraud",
}


def _read_csv(path: str | Path, required: set[str]) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Required saved output is missing: {file_path}")

    frame = pd.read_csv(file_path)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{file_path} is missing required columns: {', '.join(missing)}")
    return frame


def _lookup_record(
    path: str | Path,
    key_column: str,
    key: str,
    required: set[str],
) -> dict[str, Any]:
    normalized_key = key.strip().upper()
    if not normalized_key:
        raise ValueError(f"{key_column} cannot be blank")

    frame = _read_csv(path, required)
    matches = frame.loc[frame[key_column].astype(str).str.upper() == normalized_key]
    if matches.empty:
        raise KeyError(f"No saved record found for {normalized_key}")
    if len(matches) != 1:
        raise ValueError(f"Expected one record for {normalized_key}; found {len(matches)}")

    return matches.iloc[0].to_dict()


def load_applicant_snapshot(path: str | Path, applicant_id: str) -> dict[str, Any]:
    """Return a saved historical applicant output, not a live model prediction."""
    row = _lookup_record(path, "applicant_id", applicant_id, APPLICATION_REQUIRED_COLUMNS)
    score_fields = (
        "identity_coherence_score",
        "identity_risk_from_ics",
        "xgb_proba",
        "final_identity_risk_score",
    )
    for field in score_fields:
        value = float(row[field])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Saved score {field} must be between 0 and 1")
        row[field] = value
    row["score_origin"] = "Saved historical application-stage output; not live inference"
    return row


def load_graph_snapshot(path: str | Path, node_id: str) -> dict[str, Any]:
    """Return one saved graph-node result; no applicant mapping is inferred."""
    row = _lookup_record(path, "node_id", node_id, GRAPH_REQUIRED_COLUMNS)
    for field in ("ics", "ics_risk", "fraud_probability"):
        value = float(row[field])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Saved score {field} must be between 0 and 1")
        row[field] = value
    row["predicted_fraud"] = int(row["predicted_fraud"])
    row["score_origin"] = "Saved graph-risk CSV; no applicant-to-node mapping is present"
    return row


def inspect_application_model(model_path: str | Path, csv_path: str | Path) -> dict[str, Any]:
    """Check model metadata and runtime availability without loading or fitting it."""
    model_file = Path(model_path)
    csv_file = Path(csv_path)
    result: dict[str, Any] = {
        "artifact_exists": model_file.is_file(),
        "runtime_available": importlib.util.find_spec("xgboost") is not None,
        "feature_names": [],
        "missing_csv_features": [],
        "feature_order_can_be_selected": False,
        "loadable": False,
        "reason": "",
    }

    if not result["artifact_exists"]:
        result["reason"] = f"Model artifact is missing: {model_file}"
        return result

    try:
        model_data = json.loads(model_file.read_text(encoding="utf-8"))
        feature_names = model_data["learner"]["feature_names"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        result["reason"] = f"Model metadata could not be read: {error}"
        return result

    result["feature_names"] = list(feature_names)
    try:
        csv_columns = list(pd.read_csv(csv_file, nrows=0).columns)
    except (OSError, pd.errors.ParserError) as error:
        result["reason"] = f"Application CSV could not be read: {error}"
        return result

    result["missing_csv_features"] = [
        name for name in feature_names if name not in csv_columns
    ]
    result["feature_order_can_be_selected"] = (
        len(feature_names) > 0
        and len(set(feature_names)) == len(feature_names)
        and set(feature_names).issubset(csv_columns)
    )

    if result["missing_csv_features"]:
        result["reason"] = "One or more model features are absent from the saved application CSV."
    elif not result["feature_order_can_be_selected"]:
        result["reason"] = "The model feature order cannot be reconstructed from the saved CSV."
    elif not result["runtime_available"]:
        result["reason"] = "The xgboost package is not installed in the active environment."
    else:
        result["reason"] = (
            "Metadata and feature contract look compatible; runtime model loading "
            "has not been attempted."
        )
    return result


def build_review(
    applicant: dict[str, Any] | None,
    graph_node: dict[str, Any] | None,
) -> dict[str, Any]:
    """Keep module evidence separate; no combined decision is defined."""
    return {
        "application": applicant,
        "graph": graph_node,
        "combined_decision": None,
        "decision_status": (
            "Not calculated: no validated applicant-to-node mapping or "
            "cross-module aggregation rule is available."
        ),
    }
