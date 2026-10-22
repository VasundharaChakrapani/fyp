
from pathlib import Path
import json

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# ---------- Application setup ----------

app = FastAPI(
    title="NeuroMotion API",
    description="Research prototype for explainable infant movement analysis",
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__file__).resolve().parent / "data"
RESEARCH_DIR = BASE_DIR / "research"
PUBLIC_DIR = BASE_DIR / "frontend" / "public"


# ---------- Shared file readers ----------

def read_json_file(path: Path, label: str):
    """Read a JSON file or return a useful API error."""
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"{label} was not found at {path}",
        )

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError) as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read {label}: {error}",
        )


def read_csv_file(path: Path, label: str) -> dict:
    """Read a CSV and convert missing values to JSON-safe nulls."""
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"{label} was not found at {path}",
        )

    try:
        df = pd.read_csv(path)

        # Convert pandas missing values into Python None values.
        df = df.astype(object).where(pd.notna(df), None)

        return {
            "count": len(df),
            "columns": list(df.columns),
            "results": df.to_dict(orient="records"),
        }
    except (ValueError, OSError, pd.errors.ParserError) as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read {label}: {error}",
        )


def read_research_csv(filename: str) -> list[dict]:
    """Read a CSV from the research data directory."""
    data = read_csv_file(DATA_DIR / filename, filename)
    return data["results"]


def require_columns(
    columns: list[str],
    required: set[str],
    label: str,
) -> None:
    """Validate the schema of a research CSV."""
    missing = required.difference(columns)

    if missing:
        raise HTTPException(
            status_code=500,
            detail=(
                f"{label} is missing required columns: "
                f"{sorted(missing)}. Found: {columns}"
            ),
        )


# ---------- Core endpoints ----------

@app.get("/")
def home():
    return {
        "message": "NeuroMotion API is running",
        "version": app.version,
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "neuromotion",
    }


@app.get("/api/metrics")
def get_metrics():
    return read_json_file(
        DATA_DIR / "baseline_metrics.json",
        "Baseline metrics",
    )


@app.get("/api/features")
def get_feature_importance():
    data = read_csv_file(
        DATA_DIR / "feature_importance.csv",
        "Feature importance CSV",
    )

    return {
        "count": data["count"],
        "features": data["results"],
    }


@app.get("/api/evaluation")
def get_evaluation():
    metrics = read_json_file(
        DATA_DIR / "baseline_metrics.json",
        "Baseline metrics",
    )

    return {
        "experiment": metrics.get("experiment"),
        "validation_records": metrics.get("validation_records"),
        "labels": metrics.get("confusion_matrix_labels", []),
        "confusion_matrix": metrics.get("confusion_matrix", []),
        "classification_report": metrics.get(
            "classification_report", {}
        ),
        "clinical_validation": metrics.get(
            "clinical_validation", False
        ),
    }


@app.get("/api/experiments")
def get_experiments():
    data = read_csv_file(
        DATA_DIR / "perturbation_results.csv",
        "Perturbation results CSV",
    )

    required_columns = {
        "Perturbation strength",
        "Mean model output",
        "Change from baseline",
    }
    require_columns(
        data["columns"],
        required_columns,
        "perturbation_results.csv",
    )

    return {
        "source": "saved_experiment",
        "model": "DeepInMotion-Y pretrained ST-GCN",
        "joint": "right_wrist",
        "prediction_semantics_verified": False,
        "count": data["count"],
        "results": data["results"],
    }


# ---------- Grad-CAM explainability endpoints ----------

@app.get("/api/explainability")
def get_explainability():
    """Return per-window Grad-CAM scores for the tracked joints."""
    return {
        "source": "gradcam_search19_fold1_window_joint_scores.csv",
        "scope": "search19, fold 1; four movement windows",
        "records": read_research_csv(
            "gradcam_search19_fold1_window_joint_scores.csv"
        ),
    }


@app.get("/api/explainability/joint-scores")
def get_gradcam_joint_scores():
    """Return the full CSV structure for the joint-score viewer."""
    return read_csv_file(
        DATA_DIR / "gradcam_search19_fold1_window_joint_scores.csv",
        "Grad-CAM joint scores",
    )


@app.get("/api/explainability/joint-consistency")
def get_joint_consistency_details():
    """Return the full CSV structure for consistency statistics."""
    return read_csv_file(
        DATA_DIR / "gradcam_joint_consistency_search19_fold1.csv",
        "Grad-CAM joint consistency",
    )


@app.get("/api/joint-consistency")
def get_joint_consistency():
    """Return consistency statistics in the records format."""
    return {
        "source": "gradcam_joint_consistency_search19_fold1.csv",
        "scope": "search19, fold 1",
        "records": read_research_csv(
            "gradcam_joint_consistency_search19_fold1.csv"
        ),
    }


# ---------- Kinematic perturbation endpoints ----------

@app.get("/api/velocity-perturbations")
def get_velocity_perturbations():
    data = read_csv_file(
        DATA_DIR / "velocity_perturbation_results.csv",
        "Velocity perturbation results",
    )

    required_columns = {
        "Joint",
        "Joint index",
        "Factor",
        "Mean absolute score change",
        "Max absolute score change",
    }
    require_columns(
        data["columns"],
        required_columns,
        "velocity_perturbation_results.csv",
    )

    joints = sorted({
        row["Joint"]
        for row in data["results"]
        if row.get("Joint") is not None
    })

    factors = sorted({
        row["Factor"]
        for row in data["results"]
        if row.get("Factor") is not None
    })

    return {
        "source": "saved_experiment",
        "count": data["count"],
        "columns": data["columns"],
        "joints": joints,
        "factors": factors,
        "results": data["results"],
        "prediction_semantics_verified": False,
    }


@app.get("/api/window-sensitivity")
def get_window_sensitivity():
    return read_csv_file(
        DATA_DIR / "window_level_summary.csv",
        "Window-level summary",
    )


# ---------- Grad-CAM vs perturbation comparison ----------

@app.get("/api/explainability/perturbation-comparison")
def get_perturbation_comparison():
    """
    Align Grad-CAM joint attribution with velocity perturbation sensitivity.

    Grad-CAM scores describe attribution under the current explanation method.
    Perturbation changes describe model-output sensitivity to the tested
    velocity modifications. They are related but are not interchangeable.
    """

    consistency = read_research_csv(
        "gradcam_joint_consistency_search19_fold1.csv"
    )
    velocity = read_research_csv(
        "velocity_perturbation_results.csv"
    )

    consistency_columns = {
        "joint",
        "mean_score",
        "mean_rank",
    }
    velocity_columns = {
        "Joint",
        "Factor",
        "Mean absolute score change",
        "Max absolute score change",
    }

    if consistency:
        require_columns(
            list(consistency[0].keys()),
            consistency_columns,
            "gradcam_joint_consistency_search19_fold1.csv",
        )

    if velocity:
        require_columns(
            list(velocity[0].keys()),
            velocity_columns,
            "velocity_perturbation_results.csv",
        )

    gradcam_df = pd.DataFrame(consistency)
    velocity_df = pd.DataFrame(velocity)

    if gradcam_df.empty or velocity_df.empty:
        return {
            "source": {
                "gradcam": "gradcam_joint_consistency_search19_fold1.csv",
                "perturbation": "velocity_perturbation_results.csv",
            },
            "records": [],
            "matched_joint_count": 0,
            "correlations": {
                "pearson": None,
                "spearman": None,
            },
            "interpretation": (
                "No joint-level records were available for comparison."
            ),
            "prediction_semantics_verified": False,
        }

    # Normalize joint labels before matching.
    gradcam_df["joint_key"] = (
        gradcam_df["joint"].astype(str).str.strip().str.lower()
    )
    velocity_df["joint_key"] = (
        velocity_df["Joint"].astype(str).str.strip().str.lower()
    )

    # Multiple rows may represent different perturbation factors for
    # the same joint. Aggregate them so each joint has one comparison row.
    velocity_df["Mean absolute score change"] = pd.to_numeric(
        velocity_df["Mean absolute score change"],
        errors="coerce",
    )
    velocity_df["Max absolute score change"] = pd.to_numeric(
        velocity_df["Max absolute score change"],
        errors="coerce",
    )

    velocity_summary = (
        velocity_df.groupby("joint_key", as_index=False)
        .agg(
            joint=("Joint", "first"),
            perturbation_mean_abs_change=(
                "Mean absolute score change", "mean"
            ),
            perturbation_max_abs_change=(
                "Max absolute score change", "max"
            ),
            perturbation_factor_count=("Factor", "nunique"),
        )
    )

    gradcam_df["mean_score"] = pd.to_numeric(
        gradcam_df["mean_score"],
        errors="coerce",
    )
    gradcam_df["mean_rank"] = pd.to_numeric(
        gradcam_df["mean_rank"],
        errors="coerce",
    )

    merged = gradcam_df.merge(
        velocity_summary,
        on="joint_key",
        how="inner",
        suffixes=("_gradcam", "_perturbation"),
    )

    if merged.empty:
        return {
            "source": {
                "gradcam": "gradcam_joint_consistency_search19_fold1.csv",
                "perturbation": "velocity_perturbation_results.csv",
            },
            "records": [],
            "matched_joint_count": 0,
            "gradcam_joint_count": len(gradcam_df),
            "perturbation_joint_count": len(velocity_summary),
            "unmatched_gradcam_joints": sorted(
                set(gradcam_df["joint_key"])
                - set(velocity_summary["joint_key"])
            ),
            "unmatched_perturbation_joints": sorted(
                set(velocity_summary["joint_key"])
                - set(gradcam_df["joint_key"])
            ),
            "correlations": {
                "pearson": None,
                "spearman": None,
            },
            "interpretation": (
                "No matching joint names were found between the two datasets."
            ),
            "prediction_semantics_verified": False,
        }

    # Correlations are descriptive and are calculated only on the
    # matched joints with numeric values. Small samples require caution.
    valid = merged[
        merged["mean_score"].notna()
        & merged["perturbation_mean_abs_change"].notna()
    ].copy()

    pearson = None
    spearman = None

    if len(valid) >= 2:
        pearson_value = valid["mean_score"].corr(
            valid["perturbation_mean_abs_change"],
            method="pearson",
        )
        spearman_value = valid["mean_score"].corr(
            valid["perturbation_mean_abs_change"],
            method="spearman",
        )

        if pd.notna(pearson_value):
            pearson = float(pearson_value)

        if pd.notna(spearman_value):
            spearman = float(spearman_value)

    # Return only JSON-safe Python values.
    records = []

    for _, row in merged.iterrows():
        records.append({
            "joint": str(row.get("joint", row["joint_key"])),
            "gradcam_mean_score": (
                float(row["mean_score"])
                if pd.notna(row["mean_score"])
                else None
            ),
            "gradcam_mean_rank": (
                float(row["mean_rank"])
                if pd.notna(row["mean_rank"])
                else None
            ),
            "gradcam_std_score": (
                float(row["std_score"])
                if "std_score" in row and pd.notna(row["std_score"])
                else None
            ),
            "gradcam_top_5_windows": (
                int(row["top_5_windows"])
                if "top_5_windows" in row
                and pd.notna(row["top_5_windows"])
                else None
            ),
            "perturbation_mean_abs_change": (
                float(row["perturbation_mean_abs_change"])
                if pd.notna(row["perturbation_mean_abs_change"])
                else None
            ),
            "perturbation_max_abs_change": (
                float(row["perturbation_max_abs_change"])
                if pd.notna(row["perturbation_max_abs_change"])
                else None
            ),
            "perturbation_factor_count": int(
                row["perturbation_factor_count"]
            ),
        })

    records.sort(
        key=lambda item: (
            item["gradcam_mean_score"] is None,
            -(item["gradcam_mean_score"] or 0),
        )
    )

    return {
        "source": {
            "gradcam": "gradcam_joint_consistency_search19_fold1.csv",
            "perturbation": "velocity_perturbation_results.csv",
        },
        "scope": (
            "Grad-CAM: search19, fold 1 consistency summary; "
            "perturbation: saved velocity perturbation experiments"
        ),
        "aggregation": (
            "Velocity perturbation rows are averaged by joint across "
            "available factors; maximum absolute change is the maximum "
            "across those factors."
        ),
        "records": records,
        "matched_joint_count": len(records),
        "gradcam_joint_count": len(gradcam_df),
        "perturbation_joint_count": len(velocity_summary),
        "unmatched_gradcam_joints": sorted(
            set(gradcam_df["joint_key"])
            - set(velocity_summary["joint_key"])
        ),
        "unmatched_perturbation_joints": sorted(
            set(velocity_summary["joint_key"])
            - set(gradcam_df["joint_key"])
        ),
        "correlations": {
            "pearson": pearson,
            "spearman": spearman,
            "sample_size": len(valid),
        },
        "interpretation": (
            "Grad-CAM attribution and velocity perturbation sensitivity "
            "measure different properties. Correlations are descriptive, "
            "not proof of causal importance or clinical significance."
        ),
        "prediction_semantics_verified": False,
        "clinical_validation": False,
    }


# ---------- Model report endpoint ----------

@app.get("/api/model-report")
def get_model_report():
    return read_json_file(
        RESEARCH_DIR / "search_19_seven_fold_report.json",
        "search_19 seven-fold report",
    )


# ---------- Movement assets ----------

@app.get("/api/movement/keypoints")
def get_movement_keypoints():
    return read_csv_file(
        PUBLIC_DIR / "movement_keypoints.csv",
        "Movement keypoints CSV",
    )


@app.get("/api/movement/coordinates")
def get_movement_coordinates():
    """Return frame-by-frame joint coordinates from the uploaded CSV."""
    filename = "video_exampl_coordinates.csv"
    path = DATA_DIR / filename

    data = read_csv_file(path, "Video example coordinates CSV")

    required_columns = {"frame"}
    for joint in [
        "head_top", "nose", "right_ear", "left_ear",
        "upper_neck", "right_shoulder", "right_elbow",
        "right_wrist", "thorax", "left_shoulder",
        "left_elbow", "left_wrist", "pelvis",
        "right_hip", "right_knee", "right_ankle",
        "left_hip", "left_knee", "left_ankle",
        "right_little_finger", "right_index_finger",
        "left_little_finger", "left_index_finger",
        "right_heel", "right_little_toe", "right_big_toe",
        "left_heel", "left_little_toe", "left_big_toe",
    ]:
        required_columns.update({
            f"{joint}_x",
            f"{joint}_y",
            f"{joint}_conf",
        })

    require_columns(
        data["columns"],
        required_columns,
        filename,
    )

    return {
        "source": filename,
        "count": data["count"],
        "columns": data["columns"],
        "results": data["results"],
    }



@app.get("/api/research/assets")
def get_research_assets():
    assets = [
        "joint_movement.png",
        "movement_keypoints.csv",
    ]

    return {
        "assets": [
            {
                "name": name,
                "available": (PUBLIC_DIR / name).is_file(),
                "url": f"/{name}",
            }
            for name in assets
        ]
    }
