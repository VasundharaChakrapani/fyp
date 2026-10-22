
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  ZAxis,
} from "recharts";
import {
  Activity,
  BrainCircuit,
  ChartNoAxesCombined,
  CheckCircle2,
  CircleAlert,
  FlaskConical,
  HeartPulse,
  RefreshCw,
  ShieldCheck,
  Waves,
} from "lucide-react";
import ResearchExplorer from "./ResearchExplorer";
import "./App.css";
import GradCAMSkeleton from "./GRADCAMSkeleton";

const API = (
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/$/, "");

async function fetchJson(path) {
  let response;

  try {
    response = await fetch(`${API}${path}`);
  } catch {
    throw new Error(
      `Cannot reach the backend at ${API}. Check that FastAPI is running.`
    );
  }

  if (!response.ok) {
    let detail = "";

    try {
      const body = await response.json();
      detail = body.detail || body.message || "";
    } catch {
      // The server did not return JSON.
    }

    throw new Error(
      detail || `Request failed for ${path} (HTTP ${response.status}).`
    );
  }

  try {
    return await response.json();
  } catch {
    throw new Error(`The backend returned invalid JSON for ${path}.`);
  }
}

function displayNumber(value, digits = 3) {
  if (value === null || value === undefined || value === "") return "—";

  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "—";
}

function displayPercent(value, digits = 1) {
  if (value === null || value === undefined || value === "") return "—";

  const number = Number(value);
  if (!Number.isFinite(number)) return "—";

  return `${(number * 100).toFixed(digits)}%`;
}

function humanize(value) {
  return String(value ?? "")
    .replaceAll("_", " ")
    .replaceAll("-", " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function App() {
  const [metrics, setMetrics] = useState(null);
  const [features, setFeatures] = useState([]);
  const [evaluation, setEvaluation] = useState(null);
  const [experiments, setExperiments] = useState([]);
  const [experimentMeta, setExperimentMeta] = useState(null);

  const [activeTab, setActiveTab] = useState("overview");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [experimentLoading, setExperimentLoading] = useState(false);
  const [experimentError, setExperimentError] = useState("");
  const [selectedStrength, setSelectedStrength] = useState(null);
  const [movementImageMissing, setMovementImageMissing] = useState(false);

  // Grad-CAM joint explanation state.
  const [jointScores, setJointScores] = useState([]);
  const [jointConsistency, setJointConsistency] = useState([]);
  const [selectedWindow, setSelectedWindow] = useState(1);
  const [selectedJoint, setSelectedJoint] = useState("");
  const [jointLoading, setJointLoading] = useState(false);
  const [jointError, setJointError] = useState("");

  // Grad-CAM versus perturbation comparison state.
  const [comparison, setComparison] = useState(null);
  const [comparisonLoading, setComparisonLoading] = useState(false);
  const [comparisonError, setComparisonError] = useState("");

  const loadResearchData = useCallback(async () => {
    setLoading(true);
    setError("");

    try {
      const [metricsData, featuresData, evaluationData] = await Promise.all([
        fetchJson("/api/metrics"),
        fetchJson("/api/features"),
        fetchJson("/api/evaluation"),
      ]);

      setMetrics(metricsData || {});
      setFeatures(
        Array.isArray(featuresData?.features) ? featuresData.features : []
      );
      setEvaluation(evaluationData || {});
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Unable to load research data."
      );
    } finally {
      setLoading(false);
    }
  }, []);

  const loadExperiments = useCallback(async () => {
    setExperimentLoading(true);
    setExperimentError("");

    try {
      const data = await fetchJson("/api/experiments");
      const rows = Array.isArray(data?.results) ? data.results : [];

      setExperiments(rows);
      setExperimentMeta(data || {});

      setSelectedStrength((current) => {
        if (
          current !== null &&
          rows.some(
            (row) => Number(row["Perturbation strength"]) === current
          )
        ) {
          return current;
        }

        const first = rows.find((row) =>
          Number.isFinite(Number(row["Perturbation strength"]))
        );

        return first ? Number(first["Perturbation strength"]) : null;
      });
    } catch (err) {
      setExperimentError(
        err instanceof Error
          ? err.message
          : "Unable to load saved experiments."
      );
    } finally {
      setExperimentLoading(false);
    }
  }, []);

  const loadJointExplanations = useCallback(async () => {
    setJointLoading(true);
    setJointError("");

    try {
      const [scoresData, consistencyData] = await Promise.all([
        fetchJson("/api/explainability/joint-scores"),
        fetchJson("/api/explainability/joint-consistency"),
      ]);

      // joint-scores returns CSV records in wide format:
      // { window, head_top, nose, right_ear, ... }
      const wideRows = Array.isArray(scoresData?.results)
        ? scoresData.results
        : [];

      // Convert each joint column into a normalized row.
      const convertedScores = wideRows.flatMap((row) =>
        Object.entries(row)
          .filter(
            ([key, value]) =>
              key !== "window" &&
              value !== null &&
              value !== "" &&
              Number.isFinite(Number(value))
          )
          .map(([joint, score]) => ({
            window: Number(row.window),
            joint,
            score: Number(score),
          }))
      );

      const consistency = Array.isArray(consistencyData?.results)
        ? consistencyData.results
        : [];

      setJointScores(convertedScores);
      setJointConsistency(consistency);

      setSelectedJoint((current) => {
        if (
          current &&
          convertedScores.some(
            (row) =>
              row.window === Number(selectedWindow) &&
              row.joint === current
          )
        ) {
          return current;
        }

        return (
          convertedScores.find(
            (row) => row.window === Number(selectedWindow)
          )?.joint ||
          convertedScores[0]?.joint ||
          ""
        );
      });
    } catch (err) {
      setJointError(
        err instanceof Error
          ? err.message
          : "Unable to load joint explanations."
      );
    } finally {
      setJointLoading(false);
    }
  }, [selectedWindow]);

  const loadComparison = useCallback(async () => {
    setComparisonLoading(true);
    setComparisonError("");

    try {
      // Exact endpoint defined in backend/main.py.
      const data = await fetchJson(
        "/api/explainability/perturbation-comparison"
      );

      setComparison(data || {});
    } catch (err) {
      setComparisonError(
        err instanceof Error
          ? err.message
          : "Unable to load the explanation comparison."
      );
    } finally {
      setComparisonLoading(false);
    }
  }, []);

  useEffect(() => {
    loadResearchData();
  }, [loadResearchData]);

  useEffect(() => {
    if (
      activeTab === "experiments" &&
      experiments.length === 0 &&
      !experimentLoading
    ) {
      loadExperiments();
    }
  }, [
    activeTab,
    experiments.length,
    experimentLoading,
    loadExperiments,
  ]);

  useEffect(() => {
    if (activeTab !== "experiments") return;

    if (jointScores.length === 0 && !jointLoading && !jointError) {
      loadJointExplanations();
    }

    if (comparison === null && !comparisonLoading && !comparisonError) {
      loadComparison();
    }
  }, [
    activeTab,
    jointScores.length,
    jointLoading,
    jointError,
    loadJointExplanations,
    comparison,
    comparisonLoading,
    comparisonError,
    loadComparison,
  ]);

  // Re-select the first available joint in the chosen window if needed.
  useEffect(() => {
    const rows = jointScores.filter(
      (row) => Number(row.window) === Number(selectedWindow)
    );

    if (
      rows.length > 0 &&
      !rows.some((row) => row.joint === selectedJoint)
    ) {
      setSelectedJoint(rows[0].joint);
    }
  }, [jointScores, selectedWindow, selectedJoint]);

  const featureColumns = useMemo(
    () => (features[0] ? Object.keys(features[0]) : []),
    [features]
  );

  const importanceColumn = useMemo(() => {
    const exact = featureColumns.find((key) =>
      /^(importance_mean|mean_importance|importance|permutation_importance)$/i.test(
        key
      )
    );

    return (
      exact ||
      featureColumns.find((key) =>
        /importance|mean.*score|score.*mean/i.test(key)
      )
    );
  }, [featureColumns]);

  const featureNameColumn = useMemo(() => {
    const exact = featureColumns.find((key) =>
      /^(feature_name|feature|feature_name_mean|name|variable)$/i.test(key)
    );

    return (
      exact ||
      featureColumns.find((key) => /feature|variable|name/i.test(key))
    );
  }, [featureColumns]);

  const rankedFeatures = useMemo(() => {
    if (!importanceColumn || !featureNameColumn) return [];

    return features
      .filter(
        (row) =>
          row[featureNameColumn] != null &&
          Number.isFinite(Number(row[importanceColumn]))
      )
      .sort(
        (a, b) =>
          Number(b[importanceColumn]) - Number(a[importanceColumn])
      )
      .slice(0, 8);
  }, [features, importanceColumn, featureNameColumn]);

  const confusionMatrix = evaluation?.confusion_matrix;
  const classReport = evaluation?.classification_report;

  const classRows = useMemo(() => {
    if (!classReport || typeof classReport !== "object") return [];

    return Object.entries(classReport).filter(
      ([, value]) =>
        value &&
        typeof value === "object" &&
        ("precision" in value ||
          "recall" in value ||
          "f1-score" in value)
    );
  }, [classReport]);

  const experimentRows = useMemo(
    () =>
      [...experiments]
        .filter((row) =>
          Number.isFinite(Number(row["Perturbation strength"]))
        )
        .sort(
          (a, b) =>
            Number(a["Perturbation strength"]) -
            Number(b["Perturbation strength"])
        ),
    [experiments]
  );

  const selectedResult = experimentRows.find(
    (row) => Number(row["Perturbation strength"]) === selectedStrength
  );

  const baselineResult = experimentRows.find(
    (row) => Number(row["Perturbation strength"]) === 0
  );

  const windowRows = useMemo(
    () =>
      jointScores.filter(
        (row) => Number(row.window) === Number(selectedWindow)
      ),
    [jointScores, selectedWindow]
  );

  const selectedJointData = useMemo(
    () => windowRows.find((row) => row.joint === selectedJoint) || null,
    [windowRows, selectedJoint]
  );

  const sortedConsistency = useMemo(
    () =>
      [...jointConsistency].sort(
        (a, b) =>
          Number(a.mean_rank ?? Infinity) -
          Number(b.mean_rank ?? Infinity)
      ),
    [jointConsistency]
  );

  // The comparison endpoint returns { records, correlations, ... }.
  const comparisonRows = useMemo(
    () =>
      Array.isArray(comparison?.records) ? comparison.records : [],
    [comparison]
  );

  const comparisonChartRows = useMemo(
    () =>
      comparisonRows
        .map((row) => ({
          ...row,
          _joint: row.joint,
          _attribution: Number(row.gradcam_mean_score),
          _sensitivity: Number(row.perturbation_mean_abs_change),
        }))
        .filter(
          (row) =>
            row._joint &&
            Number.isFinite(row._attribution) &&
            Number.isFinite(row._sensitivity)
        ),
    [comparisonRows]
  );

  function renderPageHeader(
    eyebrow,
    title,
    refresh = loadResearchData,
    refreshing = loading
  ) {
    return (
      <header className="topbar">
        <div>
          <p className="eyebrow">{eyebrow}</p>
          <h1>{title}</h1>
        </div>

        <button
          className="refresh-button"
          onClick={refresh}
          disabled={refreshing}
        >
          <RefreshCw size={16} className={refreshing ? "spin" : ""} />
          Refresh data
        </button>
      </header>
    );
  }

  function renderLoadingAndError(message) {
    return (
      <>
        {error && (
          <div className="message-panel error-panel" role="alert">
            <CircleAlert size={20} />
            <div>
              <strong>Unable to load results</strong>
              <p>{error}</p>
              <button
                className="refresh-button"
                onClick={loadResearchData}
                disabled={loading}
              >
                Try again
              </button>
            </div>
          </div>
        )}

        {loading && (
          <div className="message-panel">
            <RefreshCw size={20} className="spin" />
            {message}
          </div>
        )}
      </>
    );
  }

  function renderMetricCards() {
    const featureCount = metrics?.feature_count ?? features.length;

    return (
      <section className="metrics-grid">
        <div className="metric-card">
          <div className="metric-top">
            <span>Validation accuracy</span>
            <span className="metric-icon teal">
              <CheckCircle2 size={19} />
            </span>
          </div>
          <strong className="metric-value">
            {displayPercent(metrics?.accuracy)}
          </strong>
          <p>Overall correct predictions</p>
        </div>

        <div className="metric-card">
          <div className="metric-top">
            <span>Balanced accuracy</span>
            <span className="metric-icon purple">
              <Activity size={19} />
            </span>
          </div>
          <strong className="metric-value">
            {displayPercent(metrics?.balanced_accuracy)}
          </strong>
          <p>Accounts for class imbalance</p>
        </div>

        <div className="metric-card">
          <div className="metric-top">
            <span>Validation ROC–AUC</span>
            <span className="metric-icon blue">
              <ChartNoAxesCombined size={19} />
            </span>
          </div>
          <strong className="metric-value">
            {displayNumber(metrics?.roc_auc, 3)}
          </strong>
          <p>Baseline discrimination metric</p>
        </div>

        <div className="metric-card">
          <div className="metric-top">
            <span>Features analysed</span>
            <span className="metric-icon amber">
              <BrainCircuit size={19} />
            </span>
          </div>
          <strong className="metric-value">{featureCount ?? "—"}</strong>
          <p>Baseline movement features</p>
        </div>
      </section>
    );
  }

  function renderConfusionMatrix() {
    if (
      !Array.isArray(confusionMatrix) ||
      confusionMatrix.length !== 2 ||
      !confusionMatrix.every(
        (row) => Array.isArray(row) && row.length === 2
      )
    ) {
      return (
        <p className="empty-state">
          Confusion matrix data is unavailable or not a 2 × 2 matrix.
        </p>
      );
    }

    const labels =
      Array.isArray(evaluation?.labels) && evaluation.labels.length >= 2
        ? evaluation.labels.map(humanize)
        : ["Class 0", "Class 1"];

    return (
      <>
        <p className="chart-note">
          Rows represent actual classes; columns represent predicted classes.
        </p>

        <div
          className="confusion-matrix"
          role="table"
          aria-label="Confusion matrix"
        >
          <div className="matrix-empty" />
          <strong>Predicted {labels[0]}</strong>
          <strong>Predicted {labels[1]}</strong>

          <strong>Actual {labels[0]}</strong>
          <div className="matrix-cell correct" role="cell">
            {confusionMatrix[0][0]}
          </div>
          <div className="matrix-cell incorrect" role="cell">
            {confusionMatrix[0][1]}
          </div>

          <strong>Actual {labels[1]}</strong>
          <div className="matrix-cell incorrect" role="cell">
            {confusionMatrix[1][0]}
          </div>
          <div className="matrix-cell correct" role="cell">
            {confusionMatrix[1][1]}
          </div>
        </div>

        <div className="matrix-legend">
          <span>
            <i className="legend-dot correct-dot" />
            Correct predictions
          </span>
          <span>
            <i className="legend-dot incorrect-dot" />
            Incorrect predictions
          </span>
        </div>

        <p className="chart-note">
          Off-diagonal cells represent errors. Interpret the counts in light
          of the size and balance of this evaluation split.
        </p>
      </>
    );
  }

  function renderClassPerformance() {
    if (!classRows.length) {
      return (
        <p className="empty-state">
          Class-wise metrics are unavailable.
        </p>
      );
    }

    return (
      <>
        {classRows.map(([className, report]) => (
          <div className="class-metric" key={className}>
            <strong>{humanize(className)}</strong>
            <div className="class-metric-values">
              <span>
                Precision <b>{displayPercent(report.precision)}</b>
              </span>
              <span>
                Recall <b>{displayPercent(report.recall)}</b>
              </span>
              <span>
                F1 score <b>{displayPercent(report["f1-score"])}</b>
              </span>
            </div>
            {report.support != null && (
              <p className="chart-note">Records: {report.support}</p>
            )}
          </div>
        ))}

        <p className="chart-note">
          Class-wise results can be unstable when the evaluation set is
          small or imbalanced.
        </p>
      </>
    );
  }

  function renderEvaluationGrid() {
    return (
      <section className="evaluation-grid">
        <div className="panel">
          <div className="panel-heading">
            <div>
              <h3>Confusion matrix</h3>
              <p>
                Baseline evaluation ·{" "}
                {evaluation?.validation_records ?? "—"} records
              </p>
            </div>
            <span className="panel-icon">
              <ShieldCheck size={19} />
            </span>
          </div>
          {renderConfusionMatrix()}
        </div>

        <div className="panel">
          <div className="panel-heading">
            <div>
              <h3>Class-wise performance</h3>
              <p>Precision, recall and F1 score</p>
            </div>
            <span className="panel-icon">
              <ChartNoAxesCombined size={19} />
            </span>
          </div>
          {renderClassPerformance()}
        </div>
      </section>
    );
  }

  function renderFeatureImportance() {
    const maxValue = Math.max(
      ...rankedFeatures.map((row) => Number(row[importanceColumn])),
      0.000001
    );

    return (
      <div className="panel feature-panel">
        <div className="panel-heading">
          <div>
            <h3>Movement feature importance</h3>
            <p>Ranked baseline input features</p>
          </div>
          <span className="panel-icon">
            <ChartNoAxesCombined size={19} />
          </span>
        </div>

        {rankedFeatures.length ? (
          <div className="feature-list">
            {rankedFeatures.map((feature, index) => {
              const value = Number(feature[importanceColumn]);

              return (
                <div
                  className="feature-row"
                  key={`${feature[featureNameColumn]}-${index}`}
                >
                  <div className="feature-label">
                    <span>{humanize(feature[featureNameColumn])}</span>
                    <strong>{displayNumber(value, 4)}</strong>
                  </div>
                  <div className="bar-track">
                    <div
                      className="bar-fill"
                      style={{
                        width: `${Math.max(0, (value / maxValue) * 100)}%`,
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <p className="empty-state">
            No compatible feature-importance columns were found. Check the
            column names in feature_importance.csv.
          </p>
        )}

        <p className="chart-note">
          These values come from the saved feature-importance CSV. They are
          not CAM/Grad-CAM joint explanations and do not establish causation
          or a clinical biomarker.
        </p>
      </div>
    );
  }

  function renderResearchNotes() {
    const notes = [
      {
        title: "Movement patterns",
        text: "The displayed metrics describe the saved baseline evaluation and its target classes.",
      },
      {
        title: "Explainable evidence",
        text: "Feature importance and gradient-based explanations are different methods and should be reported separately.",
      },
      {
        title: "Class imbalance",
        text: "Accuracy alone can conceal poor performance on a minority class. Review precision, recall, F1 and the confusion matrix together.",
      },
      {
        title: "CP-risk validation",
        text: "This prototype is not a clinically validated CP prediction system and must not be used to make medical decisions.",
      },
    ];

    return (
      <div className="panel insights-panel">
        <div className="panel-heading">
          <div>
            <h3>Research notes</h3>
            <p>Interpreting current results</p>
          </div>
          <span className="panel-icon">
            <FlaskConical size={19} />
          </span>
        </div>

        {notes.map((note, index) => (
          <div className="insight-item" key={note.title}>
            <span className="insight-number">
              {String(index + 1).padStart(2, "0")}
            </span>
            <div>
              <strong>{note.title}</strong>
              <p>{note.text}</p>
            </div>
          </div>
        ))}
      </div>
    );
  }

  function renderOverview() {
    return (
      <>
        {renderPageHeader(
          "EXPLAINABLE AI · INFANT MOVEMENT",
          "Research overview"
        )}

        <section className="welcome-panel">
          <div className="welcome-copy">
            <div className="project-tag">
              <HeartPulse size={15} />
              EARLY CP RESEARCH
            </div>
            <h2>
              Understanding movement.
              <br />
              Exploring explainability.
            </h2>
            <p>
              An explainable research prototype for exploring infant movement
              patterns, model behaviour and kinematic perturbations in early
              cerebral palsy research.
            </p>
          </div>

          <div className="welcome-art" aria-hidden="true">
            <div className="orbit orbit-one" />
            <div className="orbit orbit-two" />
            <div className="art-core">
              <BrainCircuit size={48} />
            </div>
            <span className="art-node node-one">
              <Activity size={18} />
            </span>
            <span className="art-node node-two">
              <Waves size={18} />
            </span>
            <span className="art-node node-three">
              <HeartPulse size={18} />
            </span>
          </div>
        </section>

        <section className="section-heading">
          <div>
            <h2>Model evaluation</h2>
            <p>Saved baseline results · Evaluation split</p>
          </div>
          <span className="section-badge">
            <ShieldCheck size={15} />
            Research results
          </span>
        </section>

        {renderLoadingAndError("Loading research data…")}

        {!loading && !error && (
          <>
            {renderMetricCards()}
            {renderEvaluationGrid()}
            <section className="content-grid">
              {renderFeatureImportance()}
              {renderResearchNotes()}
            </section>
          </>
        )}
      </>
    );
  }

  function renderInsights() {
    return (
      <>
        {renderPageHeader(
          "EXPLAINABLE AI · MODEL EVALUATION",
          "Model insights"
        )}

        {renderLoadingAndError("Loading model insights…")}

        {!loading && !error && (
          <>
            {renderMetricCards()}
            {renderEvaluationGrid()}

            <section className="panel" style={{ marginTop: 20 }}>
              <div className="panel-heading">
                <div>
                  <h3>Interpretation</h3>
                  <p>What these results do—and do not—show</p>
                </div>
                <BrainCircuit size={20} />
              </div>

              <p>
                Precision indicates how often predictions assigned to a
                class are correct. Recall indicates how many records
                belonging to that class were identified. The confusion
                matrix shows the underlying counts.
              </p>
              <p className="chart-note">
                These are saved baseline research results, not independently
                clinically validated performance and not a CP diagnosis.
              </p>
            </section>
          </>
        )}
      </>
    );
  }

  function renderMovement() {
    return (
      <>
        {renderPageHeader("EXPLORATORY DATA · MOVEMENT", "Movement analysis")}
        <GradCAMSkeleton />

        <section className="panel" style={{ marginTop: 24 }}>
          <div className="panel-heading">
            <div>
              <h2>Joint movement</h2>
              <p>Exploratory keypoint visualization</p>
            </div>
            <Activity size={24} />
          </div>

          <p className="chart-note">
            The plot and CSV are static project assets if present in the
            frontend public folder. This page does not run pose tracking or
            model inference.
          </p>

          {!movementImageMissing ? (
            <img
              src="/joint_movement.png"
              alt="Joint movement trajectory across video frames"
              onError={() => setMovementImageMissing(true)}
              style={{
                display: "block",
                width: "100%",
                maxWidth: 850,
                margin: "24px auto",
                borderRadius: 12,
              }}
            />
          ) : (
            <p className="empty-state">
              joint_movement.png was not found. Add the generated plot to
              frontend/public/ to display it here.
            </p>
          )}

          <a
            href="/movement_keypoints.csv"
            download="movement_keypoints.csv"
            className="refresh-button"
            style={{ display: "inline-flex", textDecoration: "none" }}
          >
            Download extracted keypoints CSV
          </a>
        </section>
      </>
    );
  }

  // Existing saved perturbation experiment.
  function renderSavedExperiment() {
    const chartData = experimentRows.map((row) => ({
      ...row,
      _strength: Number(row["Perturbation strength"]),
      _output: Number(row["Mean model output"]),
    }));

    return (
      <section className="panel" style={{ marginTop: 24 }}>
        <div className="panel-heading">
          <div>
            <h2>Kinematic perturbation experiment</h2>
            <p>Saved experiment results</p>
          </div>
          <FlaskConical size={24} />
        </div>

        <div className="experiment-meta">
          <span>
            <strong>Joint:</strong>{" "}
            {experimentMeta?.joint || "As recorded in experiment metadata"}
          </span>
          <span>
            <strong>Model:</strong>{" "}
            {experimentMeta?.model || "Saved model output"}
          </span>
          <span>
            <strong>Source:</strong> Recorded experiment
          </span>
        </div>

        {experimentLoading && (
          <div className="message-panel">
            <RefreshCw size={18} className="spin" />
            Loading experiment results…
          </div>
        )}

        {experimentError && (
          <div className="message-panel error-panel" role="alert">
            <CircleAlert size={20} />
            <div>
              <strong>Could not load experiment</strong>
              <p>{experimentError}</p>
              <button className="refresh-button" onClick={loadExperiments}>
                Try again
              </button>
            </div>
          </div>
        )}

        {!experimentLoading &&
          !experimentError &&
          chartData.length === 0 && (
            <p className="empty-state">
              No saved experiment rows were returned by the API.
            </p>
          )}

        {!experimentLoading &&
          !experimentError &&
          chartData.length > 0 && (
            <>
              <div className="experiment-summary">
                <div className="metric-card">
                  <div className="metric-top">
                    <span>Baseline output</span>
                    <Activity size={18} />
                  </div>
                  <strong className="metric-value">
                    {displayNumber(baselineResult?.["Mean model output"], 6)}
                  </strong>
                  <p>At 0% perturbation, if included</p>
                </div>

                <div className="metric-card">
                  <div className="metric-top">
                    <span>Selected output</span>
                    <ChartNoAxesCombined size={18} />
                  </div>
                  <strong className="metric-value">
                    {displayNumber(selectedResult?.["Mean model output"], 6)}
                  </strong>
                  <p>
                    {selectedStrength == null
                      ? "Select a recorded strength"
                      : `${(selectedStrength * 100).toFixed(1)}% perturbation`}
                  </p>
                </div>

                <div className="metric-card">
                  <div className="metric-top">
                    <span>Change from baseline</span>
                    <Waves size={18} />
                  </div>
                  <strong className="metric-value">
                    {displayNumber(selectedResult?.["Change from baseline"], 6)}
                  </strong>
                  <p>Value from saved CSV</p>
                </div>
              </div>

              <div className="experiment-control">
                <label htmlFor="perturbation-strength">
                  Select a recorded perturbation strength
                </label>
                <select
                  id="perturbation-strength"
                  value={selectedStrength ?? ""}
                  onChange={(event) =>
                    setSelectedStrength(Number(event.target.value))
                  }
                >
                  {chartData.map((row) => (
                    <option key={row._strength} value={row._strength}>
                      {(row._strength * 100).toFixed(1)}%
                    </option>
                  ))}
                </select>
              </div>

              <div className="panel experiment-chart-panel">
                <div className="panel-heading">
                  <div>
                    <h3>Perturbation response</h3>
                    <p className="chart-note">
                      Recorded model output across perturbation strengths
                    </p>
                  </div>
                </div>

                <div className="experiment-chart">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={chartData}
                      margin={{ top: 12, right: 20, left: 12, bottom: 24 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis
                        dataKey="_strength"
                        type="number"
                        domain={["dataMin", "dataMax"]}
                        tickFormatter={(value) =>
                          `${(Number(value) * 100).toFixed(1)}%`
                        }
                        label={{
                          value: "Perturbation strength",
                          position: "insideBottom",
                          offset: -16,
                        }}
                      />
                      <YAxis
                        domain={["auto", "auto"]}
                        tickFormatter={(value) => displayNumber(value, 3)}
                      />
                      <Tooltip
                        formatter={(value) => [
                          displayNumber(value, 6),
                          "Mean model output",
                        ]}
                        labelFormatter={(value) =>
                          `Perturbation strength: ${(Number(value) * 100).toFixed(1)}%`
                        }
                      />
                      <Line
                        type="monotone"
                        dataKey="_output"
                        name="Mean model output"
                        stroke="#0d9488"
                        strokeWidth={3}
                        dot={{ r: 5 }}
                        activeDot={{ r: 7 }}
                        connectNulls={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                <p className="chart-note">
                  Saved experiment values only. The selector does not run
                  new inference. Outputs are not verified clinical risk
                  probabilities.
                </p>
              </div>

              <div className="experiment-table-wrap">
                <table className="experiment-table">
                  <thead>
                    <tr>
                      <th>Perturbation</th>
                      <th>Mean model output</th>
                      <th>Change from baseline</th>
                    </tr>
                  </thead>
                  <tbody>
                    {chartData.map((row) => (
                      <tr
                        key={row._strength}
                        className={
                          row._strength === selectedStrength
                            ? "selected-experiment-row"
                            : ""
                        }
                        onClick={() => setSelectedStrength(row._strength)}
                        tabIndex={0}
                        onKeyDown={(event) => {
                          if (
                            event.key === "Enter" ||
                            event.key === " "
                          ) {
                            setSelectedStrength(row._strength);
                          }
                        }}
                      >
                        <td>{(row._strength * 100).toFixed(1)}%</td>
                        <td>{displayNumber(row["Mean model output"], 6)}</td>
                        <td>{displayNumber(row["Change from baseline"], 6)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
      </section>
    );
  }

  // New: interactive Grad-CAM joint heatmap and consistency explorer.
  function renderJointHeatmap() {
    const maxScore = Math.max(
      ...windowRows
        .map((row) => Number(row.score))
        .filter(Number.isFinite),
      0.000001
    );

    const rankedWindowRows = [...windowRows].sort(
      (a, b) => Number(b.score) - Number(a.score)
    );

    return (
      <section className="panel explainability-panel">
        <div className="panel-heading">
          <div>
            <h2>Interactive Grad-CAM joint heatmap</h2>
            <p>
              Explore joint attribution scores across the four movement
              windows
            </p>
          </div>
          <BrainCircuit size={24} />
        </div>

        {jointLoading && (
          <div className="message-panel">
            <RefreshCw size={18} className="spin" />
            Loading joint explanations…
          </div>
        )}

        {jointError && (
          <div className="message-panel error-panel" role="alert">
            <CircleAlert size={20} />
            <div>
              <strong>Unable to load Grad-CAM data</strong>
              <p>{jointError}</p>
              <button
                className="refresh-button"
                onClick={loadJointExplanations}
                disabled={jointLoading}
              >
                Try again
              </button>
            </div>
          </div>
        )}

        {!jointLoading && !jointError && (
          <>
            <div className="experiment-control">
              <label htmlFor="gradcam-window">Select temporal window</label>
              <select
                id="gradcam-window"
                value={selectedWindow}
                onChange={(event) =>
                  setSelectedWindow(Number(event.target.value))
                }
              >
                {[1, 2, 3, 4].map((window) => (
                  <option key={window} value={window}>
                    Window {window}
                  </option>
                ))}
              </select>
            </div>

            <p className="chart-note">
              Each cell represents one joint. Darker cells indicate larger
              scores relative to the maximum score in the selected window.
              This is an attribution visualization, not a clinical risk
              heatmap.
            </p>

            {rankedWindowRows.length === 0 ? (
              <p className="empty-state">
                No joint scores were returned for this window. Check the
                backend CSV's window values and joint columns.
              </p>
            ) : (
              <>
                <div className="joint-heatmap">
                  {rankedWindowRows.map((row, index) => {
                    const score = Number(row.score);
                    const intensity = Math.max(
                      0.06,
                      Math.min(1, score / maxScore)
                    );

                    return (
                      <button
                        key={`${row.window}-${row.joint}`}
                        type="button"
                        className={`joint-heatmap-cell ${
                          selectedJoint === row.joint ? "selected" : ""
                        }`}
                        style={{
                          "--joint-intensity": intensity,
                        }}
                        onClick={() => setSelectedJoint(row.joint)}
                        title={`${humanize(row.joint)}: ${displayNumber(score, 6)}`}
                        aria-pressed={selectedJoint === row.joint}
                      >
                        <span className="joint-rank">#{index + 1}</span>
                        <strong>{humanize(row.joint)}</strong>
                        <span>{displayNumber(score, 6)}</span>
                      </button>
                    );
                  })}
                </div>

                {selectedJointData && (
                  <div className="joint-detail-card">
                    <span className="eyebrow">
                      SELECTED JOINT · WINDOW {selectedWindow}
                    </span>
                    <h3>{humanize(selectedJointData.joint)}</h3>
                    <p>
                      Attribution score:{" "}
                      <strong>
                        {displayNumber(selectedJointData.score, 6)}
                      </strong>
                    </p>
                    <p className="chart-note">
                      This score reflects the saved Grad-CAM explanation for
                      the selected window.
                    </p>
                  </div>
                )}
              </>
            )}

            <div className="panel-heading consistency-heading">
              <div>
                <h3>Across-window consistency</h3>
                <p>
                  Saved mean attribution, score variation and mean rank
                </p>
              </div>
            </div>

            {sortedConsistency.length > 0 ? (
              <div className="experiment-table-wrap">
                <table className="experiment-table">
                  <thead>
                    <tr>
                      <th>Joint</th>
                      <th>Mean score</th>
                      <th>Score std.</th>
                      <th>Mean rank</th>
                      <th>Top-5 windows</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sortedConsistency.map((row) => (
                      <tr
                        key={row.joint}
                        className={
                          row.joint === selectedJoint
                            ? "selected-experiment-row"
                            : ""
                        }
                        onClick={() => setSelectedJoint(row.joint)}
                      >
                        <td>{humanize(row.joint)}</td>
                        <td>{displayNumber(row.mean_score, 6)}</td>
                        <td>{displayNumber(row.std_score, 6)}</td>
                        <td>{displayNumber(row.mean_rank, 2)}</td>
                        <td>{displayNumber(row.top_5_windows, 0)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="empty-state">
                No across-window consistency records were returned.
              </p>
            )}

            <p className="chart-note">
              Consistency statistics describe the saved four-window
              explanation. They do not establish clinical significance.
            </p>
          </>
        )}
      </section>
    );
  }

  // New: compare Grad-CAM attribution with measured perturbation sensitivity.
  function renderComparisonDashboard() {
    const correlations = comparison?.correlations || {};
    const correlationCards = [
      {
        label: "Pearson correlation",
        value: correlations.pearson,
        description: "Linear association",
      },
      {
        label: "Spearman correlation",
        value: correlations.spearman,
        description: "Rank-based association",
      },
      {
        label: "Matched joints",
        value: comparison?.matched_joint_count,
        description: "Joints shared by both datasets",
      },
      {
        label: "Correlation sample",
        value: correlations.sample_size,
        description: "Numeric pairs used",
      },
    ];

    const hasCorrelation = (value) =>
      value !== null && value !== undefined && Number.isFinite(Number(value));

    return (
      <section className="panel explainability-panel">
        <div className="panel-heading">
          <div>
            <h2>Grad-CAM vs. perturbation sensitivity</h2>
            <p>
              Compare gradient attribution with observed model-output changes
            </p>
          </div>
          <ChartNoAxesCombined size={24} />
        </div>

        {comparisonLoading && (
          <div className="message-panel">
            <RefreshCw size={18} className="spin" />
            Loading comparison results…
          </div>
        )}

        {comparisonError && (
          <div className="message-panel error-panel" role="alert">
            <CircleAlert size={20} />
            <div>
              <strong>Unable to load comparison</strong>
              <p>{comparisonError}</p>
              <button
                className="refresh-button"
                onClick={loadComparison}
                disabled={comparisonLoading}
              >
                Try again
              </button>
            </div>
          </div>
        )}

        {!comparisonLoading && !comparisonError && comparison && (
          <>
            <div className="comparison-metrics">
              {correlationCards.map((item) => (
                <div className="metric-card" key={item.label}>
                  <div className="metric-top">
                    <span>{item.label}</span>
                    <ChartNoAxesCombined size={18} />
                  </div>
                  <strong className="metric-value">
                    {item.value === null || item.value === undefined
                      ? "—"
                      : item.label === "Matched joints" ||
                          item.label === "Correlation sample"
                        ? displayNumber(item.value, 0)
                        : displayNumber(item.value, 4)}
                  </strong>
                  <p>{item.description}</p>
                </div>
              ))}
            </div>

            {comparisonChartRows.length > 0 ? (
              <>
                <div className="comparison-chart">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart
                      margin={{
                        top: 16,
                        right: 24,
                        bottom: 32,
                        left: 16,
                      }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis
                        type="number"
                        dataKey="_attribution"
                        name="Grad-CAM mean score"
                        tickFormatter={(value) => displayNumber(value, 4)}
                        label={{
                          value: "Grad-CAM mean attribution",
                          position: "insideBottom",
                          offset: -20,
                        }}
                      />
                      <YAxis
                        type="number"
                        dataKey="_sensitivity"
                        name="Mean absolute output change"
                        tickFormatter={(value) => displayNumber(value, 4)}
                        label={{
                          value: "Perturbation sensitivity",
                          angle: -90,
                          position: "insideLeft",
                        }}
                      />
                      <ZAxis range={[65, 65]} />
                      <Tooltip
                        cursor={{ strokeDasharray: "3 3" }}
                        formatter={(value, name) => [
                          displayNumber(value, 6),
                          name,
                        ]}
                        labelFormatter={(_, payload) =>
                          payload?.[0]?.payload?._joint
                            ? humanize(payload[0].payload._joint)
                            : "Joint"
                        }
                      />
                      <Scatter
                        name="Matched joints"
                        data={comparisonChartRows}
                        fill="#0d9488"
                      />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>

                <p className="chart-note">
                  Each point represents one matched joint. The X-axis shows
                  mean Grad-CAM attribution; the Y-axis shows mean absolute
                  model-output change across the available velocity
                  perturbation factors.
                </p>

                <div className="experiment-table-wrap">
                  <table className="experiment-table">
                    <thead>
                      <tr>
                        <th>Joint</th>
                        <th>Grad-CAM mean</th>
                        <th>Grad-CAM rank</th>
                        <th>Mean output change</th>
                        <th>Maximum change</th>
                        <th>Factors</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[...comparisonChartRows]
                        .sort(
                          (a, b) => b._sensitivity - a._sensitivity
                        )
                        .map((row) => (
                          <tr key={row._joint}>
                            <td>{humanize(row._joint)}</td>
                            <td>
                              {displayNumber(row.gradcam_mean_score, 6)}
                            </td>
                            <td>
                              {displayNumber(row.gradcam_mean_rank, 2)}
                            </td>
                            <td>
                              {displayNumber(
                                row.perturbation_mean_abs_change,
                                6
                              )}
                            </td>
                            <td>
                              {displayNumber(
                                row.perturbation_max_abs_change,
                                6
                              )}
                            </td>
                            <td>
                              {displayNumber(
                                row.perturbation_factor_count,
                                0
                              )}
                            </td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>
              </>
            ) : (
              <p className="empty-state">
                No matched joints with numeric attribution and sensitivity
                values were returned. Check the CSV contents and the API
                response.
              </p>
            )}

            {Array.isArray(comparison.unmatched_gradcam_joints) &&
              comparison.unmatched_gradcam_joints.length > 0 && (
                <p className="chart-note">
                  Unmatched Grad-CAM joint labels:{" "}
                  {comparison.unmatched_gradcam_joints
                    .map(humanize)
                    .join(", ")}
                </p>
              )}

            {Array.isArray(comparison.unmatched_perturbation_joints) &&
              comparison.unmatched_perturbation_joints.length > 0 && (
                <p className="chart-note">
                  Unmatched perturbation joint labels:{" "}
                  {comparison.unmatched_perturbation_joints
                    .map(humanize)
                    .join(", ")}
                </p>
              )}

            <p className="chart-note">
              {comparison.interpretation ||
                "Correlations are descriptive and do not establish causation."}
              {" "}Prediction semantics are not independently verified, and
              this comparison is not clinically validated.
            </p>

            <button
              className="refresh-button"
              onClick={loadComparison}
              disabled={comparisonLoading}
            >
              <RefreshCw
                size={16}
                className={comparisonLoading ? "spin" : ""}
              />
              Refresh comparison
            </button>
          </>
        )}

        {!comparisonLoading && !comparisonError && !comparison && (
          <div className="empty-state">
            <p>No comparison response is available yet.</p>
            <button className="refresh-button" onClick={loadComparison}>
              Load comparison
            </button>
          </div>
        )}
      </section>
    );
  }

  function renderExperiments() {
    return (
      <>
        {renderPageHeader(
          "EXPLAINABLE AI · RESEARCH WORKSPACE",
          "Experiments",
          loadExperiments,
          experimentLoading
        )}

        {renderSavedExperiment()}
        {renderJointHeatmap()}
        {renderComparisonDashboard()}
      </>
    );
  }

  function renderSidebarNavItem(tab, label, Icon) {
    return (
      <button
        className={`nav-item ${activeTab === tab ? "active" : ""}`}
        onClick={() => setActiveTab(tab)}
        aria-current={activeTab === tab ? "page" : undefined}
      >
        <Icon size={19} />
        {label}
      </button>
    );
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">
            <Waves size={25} />
          </div>
          <div>
            <h2>NeuroMotion</h2>
            <p>RESEARCH PLATFORM</p>
          </div>
        </div>

        <div className="nav-label">WORKSPACE</div>

        {renderSidebarNavItem(
          "overview",
          "Research overview",
          ChartNoAxesCombined
        )}
        {renderSidebarNavItem("insights", "Model insights", BrainCircuit)}
        {renderSidebarNavItem("movement", "Movement analysis", Activity)}
        {renderSidebarNavItem("experiments", "Experiments", FlaskConical)}
        {renderSidebarNavItem(
          "research",
          "Research explorer",
          ChartNoAxesCombined
        )}

        <div className="sidebar-bottom">
          <div className="sidebar-status">
            <span className="status-dot" />
            <div>
              <strong>Research workspace</strong>
              <p>Local development</p>
            </div>
          </div>

          <p className="sidebar-disclaimer">
            Research prototype. Not intended for clinical diagnosis.
          </p>
        </div>
      </aside>

      <main className="main-content">
        {activeTab === "movement" ? (
          renderMovement()
        ) : activeTab === "insights" ? (
          renderInsights()
        ) : activeTab === "experiments" ? (
          renderExperiments()
        ) : activeTab === "research" ? (
          <ResearchExplorer />
        ) : (
          renderOverview()
        )}

        <footer className="page-footer">
          <span>NEUROMOTION · EXPLAINABLE MOVEMENT RESEARCH</span>
          <span>Research prototype · Not for clinical use</span>
        </footer>
      </main>
    </div>
  );
}

export default App;
