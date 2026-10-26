
import { useEffect, useMemo, useState } from "react";
import "./ResearchExplorer.css";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";

const API = (
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/$/, "");

const ENDPOINTS = {
  explainability: "/api/explainability",
  consistency: "/api/joint-consistency",
  velocity: "/api/velocity-perturbations",
  windows: "/api/window-sensitivity",
  integratedGradients: "/api/explainability/integrated-gradients",
  igConvergence: "/api/explainability/ig-convergence",
  igJointConvergence: "/api/explainability/ig-joint-convergence",
  methodComparison: "/api/explainability/method-comparison",
};

async function fetchRecords(endpoint) {
  const response = await fetch(`${API}${endpoint}`);

  if (!response.ok) {
    throw new Error(`Request failed (${response.status}): ${endpoint}`);
  }

  const json = await response.json();
  const records = json.records ?? json.results;

  if (!Array.isArray(records)) {
    throw new Error(
      `Unexpected response format from ${endpoint}: expected "records" or "results" array`
    );
  }

  return records;
}

function Panel({ title, subtitle, children }) {
  return (
    <section className="re-panel">
      <div className="re-panel-heading">
        <h3>{title}</h3>
        {subtitle && <p>{subtitle}</p>}
      </div>
      {children}
    </section>
  );
}

function Metric({ label, value, detail }) {
  return (
    <div className="re-metric">
      <span>{label}</span>
      <strong>{value}</strong>
      {detail && <small>{detail}</small>}
    </div>
  );
}

function DownloadCsv({ filename, rows }) {
  function download() {
    if (!rows.length) return;

    const columns = Object.keys(rows[0]);

    const escapeCell = (value) => {
      const text = value == null ? "" : String(value);
      return `"${text.replace(/"/g, '""')}"`;
    };

    const csv = [
      columns.map(escapeCell).join(","),
      ...rows.map((row) =>
        columns.map((column) => escapeCell(row[column])).join(",")
      ),
    ].join("\n");

    const blob = new Blob(["\uFEFF" + csv], {
      type: "text/csv;charset=utf-8;",
    });

    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
  }

  return (
    <button
      className="re-button"
      onClick={download}
      disabled={!rows.length}
    >
      Download CSV
    </button>
  );
}

function formatNumber(value, digits = 5) {
  if (value == null || value === "") return "—";

  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "—";
}

export default function ResearchExplorer() {
  const [data, setData] = useState({
    explainability: [],
    consistency: [],
    velocity: [],
    windows: [],
    integratedGradients: [],
    igConvergence: [],
    igJointConvergence: [],
    methodComparison: [],
  });

  const [comparisonSummary, setComparisonSummary] = useState(null);
  const [comparisonError, setComparisonError] = useState("");

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [windowNumber, setWindowNumber] = useState("1");
  const [selectedJoint, setSelectedJoint] = useState("All joints");
  const [velocityWindow, setVelocityWindow] = useState("1");
  const [section, setSection] = useState("explainability");

  useEffect(() => {
    let active = true;

    async function load() {
      setLoading(true);
      setError("");
      setComparisonError("");

      // Fetch the ordinary datasets separately from the comparison
      // endpoint because the latter also returns summary metrics.
      const ordinaryEndpoints = Object.entries(ENDPOINTS).filter(
        ([key]) => key !== "methodComparison"
      );

      const results = await Promise.allSettled(
        ordinaryEndpoints.map(async ([key, endpoint]) => [
          key,
          await fetchRecords(endpoint),
        ])
      );

      if (!active) return;

      const nextData = {};
      const failures = [];

      for (const result of results) {
        if (result.status === "fulfilled") {
          const [key, records] = result.value;
          nextData[key] = records;
        } else {
          failures.push(
            result.reason?.message || "Unknown request error"
          );
        }
      }

      setData((previous) => ({ ...previous, ...nextData }));

      // Fetch the complete comparison response so its metrics
      // are available as well as its joint-level records.
      try {
        const response = await fetch(
          `${API}${ENDPOINTS.methodComparison}`
        );

        if (!response.ok) {
          throw new Error(
            `Comparison request failed (${response.status})`
          );
        }

        const json = await response.json();

        if (!Array.isArray(json.records)) {
          throw new Error(
            "Comparison response does not contain a records array."
          );
        }

        if (active) {
          setComparisonSummary(json);
          setData((previous) => ({
            ...previous,
            methodComparison: json.records,
          }));
        }
      } catch (err) {
        if (active) {
          setComparisonError(
            err.message || "Could not load method comparison."
          );
          setComparisonSummary(null);
        }
      }

      if (!active) return;

      if (failures.length) {
        setError(
          `Some datasets could not be loaded. ${failures.join("; ")}`
        );
      }

      setLoading(false);
    }

    load();

    return () => {
      active = false;
    };
  }, []);

  const jointColumns = useMemo(() => {
    if (!data.explainability.length) return [];

    return Object.keys(data.explainability[0]).filter(
      (column) => column.toLowerCase() !== "window"
    );
  }, [data.explainability]);

  const explanationRows = useMemo(() => {
    const row = data.explainability.find(
      (item) => String(item.window) === windowNumber
    );

    if (!row) return [];

    return jointColumns
      .map((joint) => ({ joint, score: Number(row[joint]) }))
      .filter((item) => Number.isFinite(item.score))
      .sort((a, b) => b.score - a.score);
  }, [data.explainability, jointColumns, windowNumber]);

  const topJoint = explanationRows[0];

  const meanScore = explanationRows.length
    ? explanationRows.reduce((sum, item) => sum + item.score, 0) /
      explanationRows.length
    : 0;

  const consistencyRows = useMemo(
    () =>
      [...data.consistency].sort(
        (a, b) => Number(b.mean_score) - Number(a.mean_score)
      ),
    [data.consistency]
  );

  const velocityJoints = useMemo(
    () => [...new Set(data.velocity.map((row) => row.Joint).filter(Boolean))],
    [data.velocity]
  );

  const filteredVelocity = useMemo(
    () =>
      data.velocity.filter(
        (row) =>
          selectedJoint === "All joints" || row.Joint === selectedJoint
      ),
    [data.velocity, selectedJoint]
  );

  const velocityChart = useMemo(
    () =>
      filteredVelocity.map((row) => ({
        ...row,
        factorLabel: String(row.Factor),
        scoreChange: Number(row["Mean absolute score change"]),
        windowScore: Number(row[`Window ${velocityWindow} score`]),
        windowDelta: Number(row[`Window ${velocityWindow} delta`]),
      })),
    [filteredVelocity, velocityWindow]
  );

  const windowRows = useMemo(
    () =>
      data.windows.map((row) => ({
        ...row,
        windowLabel: row.Window,
        meanChange: Number(row["Mean absolute change"]),
        maxChange: Number(row["Maximum absolute change"]),
      })),
    [data.windows]
  );

  const integratedGradientsRows = useMemo(
    () =>
      [...data.integratedGradients]
        .map((row) => ({
          ...row,
          ig_absolute_score: Number(row.ig_absolute_score),
          ig_normalized_score: Number(row.ig_normalized_score),
        }))
        .filter(
          (row) =>
            row.joint &&
            Number.isFinite(row.ig_absolute_score) &&
            Number.isFinite(row.ig_normalized_score)
        )
        .sort((a, b) => b.ig_absolute_score - a.ig_absolute_score),
    [data.integratedGradients]
  );

  const topIGJoint = integratedGradientsRows[0];

  const igConvergenceRows = useMemo(
    () =>
      [...data.igConvergence]
        .map((row) => ({
          ...row,
          ig_steps: Number(row.ig_steps),
          completeness_delta: Number(row.completeness_delta),
          joint_attribution_change:
            row.joint_attribution_change == null ||
            row.joint_attribution_change === ""
              ? null
              : Number(row.joint_attribution_change),
          top_joint_score: Number(row.top_joint_score),
        }))
        .filter(
          (row) =>
            Number.isFinite(row.ig_steps) &&
            Number.isFinite(row.completeness_delta)
        )
        .sort((a, b) => a.ig_steps - b.ig_steps),
    [data.igConvergence]
  );

  const latestIGConvergence =
    igConvergenceRows[igConvergenceRows.length - 1];

  const finalAttributionChange = latestIGConvergence
    ? latestIGConvergence.joint_attribution_change
    : null;

  const igConvergenceChart = useMemo(
    () =>
      igConvergenceRows.map((row) => ({
        ...row,
        stepsLabel: `${row.ig_steps} steps`,
      })),
    [igConvergenceRows]
  );

  const comparisonRows = useMemo(
    () =>
      [...data.methodComparison]
        .map((row) => ({
          ...row,
          gradcam_score: Number(row.gradcam_score),
          ig_score: Number(row.ig_score),
          gradcam_rank: Number(row.gradcam_rank),
          ig_rank: Number(row.ig_rank),
        }))
        .filter(
          (row) =>
            row.joint &&
            Number.isFinite(row.gradcam_score) &&
            Number.isFinite(row.ig_score)
        )
        .sort((a, b) => a.gradcam_rank - b.gradcam_rank),
    [data.methodComparison]
  );

  const comparisonChartRows = useMemo(
    () =>
      comparisonRows.map((row) => ({
        ...row,
        jointLabel: row.joint.replaceAll("_", " "),
      })),
    [comparisonRows]
  );

  if (loading) {
    return <div className="re-status">Loading research datasets…</div>;
  }

  return (
    <div className="research-explorer">
      <header className="re-header">
        <div>
          <p className="re-eyebrow">RESEARCH WORKSPACE</p>
          <h2>Explainability &amp; Perturbation Explorer</h2>
          <p>
            Explore recorded joint-level explanations, Integrated
            Gradients, kinematic perturbations, and movement-window
            sensitivity.
          </p>
        </div>
        <span className="re-status-pill">Saved experiment results</span>
      </header>

      {error && <div className="re-warning">{error}</div>}

      <div className="re-tabs">
        {[
          ["explainability", "Joint explanations"],
          ["integratedGradients", "Integrated Gradients"],
          ["methodComparison", "Grad-CAM vs IG"],
          ["consistency", "Joint consistency"],
          ["velocity", "Velocity lab"],
          ["windows", "Window sensitivity"],
        ].map(([key, label]) => (
          <button
            key={key}
            className={section === key ? "active" : ""}
            onClick={() => setSection(key)}
          >
            {label}
          </button>
        ))}
      </div>

      {section === "explainability" && (
        <>
          <div className="re-metrics">
            <Metric
              label="Joints evaluated"
              value={jointColumns.length || "—"}
              detail="Available in the Grad-CAM score file"
            />
            <Metric
              label="Movement windows"
              value={data.explainability.length || "—"}
              detail="Saved window-level explanations"
            />
            <Metric
              label="Highest-scoring joint"
              value={topJoint?.joint || "—"}
              detail={
                topJoint ? `Score: ${formatNumber(topJoint.score)}` : ""
              }
            />
            <Metric
              label="Mean joint score"
              value={formatNumber(meanScore)}
              detail={`Window ${windowNumber}`}
            />
          </div>

          <Panel
            title="Joint-level Grad-CAM scores"
            subtitle="Select a window to compare the saved joint scores."
          >
            <div className="re-controls">
              <label>
                Movement window
                <select
                  value={windowNumber}
                  onChange={(event) => setWindowNumber(event.target.value)}
                >
                  {data.explainability.map((row) => (
                    <option key={row.window} value={String(row.window)}>
                      Window {row.window}
                    </option>
                  ))}
                </select>
              </label>

              <DownloadCsv
                filename={`gradcam-window-${windowNumber}.csv`}
                rows={explanationRows}
              />
            </div>

            {explanationRows.length ? (
              <div className="re-chart">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={explanationRows.slice(0, 15)}
                    layout="vertical"
                    margin={{ left: 15, right: 24, top: 5, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" />
                    <YAxis
                      dataKey="joint"
                      type="category"
                      width={145}
                      tick={{ fontSize: 11 }}
                    />
                    <Tooltip
                      formatter={(value) => formatNumber(value, 6)}
                    />
                    <Bar
                      dataKey="score"
                      name="Grad-CAM score"
                      fill="#6d75df"
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <p>No explanation scores are available for this window.</p>
            )}

            <p className="re-note">
              These are saved model explanation scores, not clinical
              measures. The chart displays the 15 highest-scoring joints
              for the selected window.
            </p>
          </Panel>
        </>
      )}

      {section === "integratedGradients" && (
        <>
          <div className="re-metrics">
            <Metric
              label="Joints attributed"
              value={integratedGradientsRows.length || "—"}
              detail="Joint scores in the exported IG file"
            />
            <Metric
              label="Highest-attribution joint"
              value={topIGJoint?.joint || "—"}
              detail={
                topIGJoint
                  ? `Absolute score: ${formatNumber(
                      topIGJoint.ig_absolute_score,
                      6
                    )}`
                  : "No IG results loaded"
              }
            />
            <Metric
              label="Normalized top attribution"
              value={
                topIGJoint
                  ? `${formatNumber(
                      topIGJoint.ig_normalized_score * 100,
                      2
                    )}%`
                  : "—"
              }
              detail="Share of summed absolute attributions"
            />
            <Metric
              label="Scope"
              value="1 window"
              detail="First movement window only"
            />
          </div>

          <Panel
            title="Integrated Gradients — Joint Attribution"
            subtitle="Absolute attribution scores aggregated across feature channels and time for the first movement window."
          >
            <div className="re-controls">
              <span>{integratedGradientsRows.length} joints</span>
              <DownloadCsv
                filename="integrated-gradients-joint-attributions.csv"
                rows={integratedGradientsRows}
              />
            </div>

            {integratedGradientsRows.length ? (
              <>
                <div className="re-chart">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={integratedGradientsRows.slice(0, 15)}
                      layout="vertical"
                      margin={{
                        left: 15,
                        right: 24,
                        top: 5,
                        bottom: 5,
                      }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" />
                      <YAxis
                        dataKey="joint"
                        type="category"
                        width={145}
                        tick={{ fontSize: 11 }}
                      />
                      <Tooltip
                        formatter={(value) => formatNumber(value, 6)}
                      />
                      <Bar
                        dataKey="ig_absolute_score"
                        name="Absolute IG attribution"
                        fill="#d97757"
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                <div className="re-table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Joint</th>
                        <th>Absolute attribution</th>
                        <th>Normalized attribution</th>
                      </tr>
                    </thead>
                    <tbody>
                      {integratedGradientsRows.map((row) => (
                        <tr key={row.joint}>
                          <td>{row.joint}</td>
                          <td>
                            {formatNumber(row.ig_absolute_score, 6)}
                          </td>
                          <td>
                            {formatNumber(row.ig_normalized_score, 6)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            ) : (
              <p>
                No Integrated Gradients results are available. Check that
                the CSV exists in backend/data and that the API endpoint
                is running.
              </p>
            )}
          </Panel>

          <Panel
            title="Integrated Gradients — Convergence"
            subtitle="Compare attribution stability and completeness as the integration step count increases."
          >
            <div className="re-metrics">
              <Metric
                label="Step counts completed"
                value={igConvergenceRows.length || "—"}
                detail="Expected: 50, 100, and 200"
              />
              <Metric
                label="Top joint at 200 steps"
                value={latestIGConvergence?.top_joint || "—"}
                detail={
                  latestIGConvergence
                    ? `Score: ${formatNumber(
                        latestIGConvergence.top_joint_score,
                        6
                      )}`
                    : "No convergence data loaded"
                }
              />
              <Metric
                label="Completeness delta at final run"
                value={
                  latestIGConvergence
                    ? formatNumber(
                        latestIGConvergence.completeness_delta,
                        6
                      )
                    : "—"
                }
                detail="Closer to zero generally indicates better completeness"
              />
              <Metric
                label="Attribution change at final run"
                value={
                  finalAttributionChange == null
                    ? "—"
                    : formatNumber(finalAttributionChange, 6)
                }
                detail="Change from the preceding step count"
              />
            </div>

            {igConvergenceRows.length ? (
              <>
                <div className="re-controls">
                  <span>
                    {igConvergenceRows.length} convergence records
                  </span>
                  <DownloadCsv
                    filename="ig-convergence.csv"
                    rows={igConvergenceRows}
                  />
                </div>

                <div className="re-chart">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={igConvergenceChart}
                      margin={{
                        left: 12,
                        right: 24,
                        top: 10,
                        bottom: 10,
                      }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="stepsLabel" />
                      <YAxis
                        yAxisId="delta"
                        tickFormatter={(value) => formatNumber(value, 3)}
                      />
                      <Tooltip
                        formatter={(value, name) => [
                          formatNumber(value, 6),
                          name,
                        ]}
                      />
                      <Legend />
                      <Line
                        yAxisId="delta"
                        type="monotone"
                        dataKey="completeness_delta"
                        name="Completeness delta"
                        stroke="#6d75df"
                        strokeWidth={2}
                        connectNulls
                      />
                      <Line
                        yAxisId="delta"
                        type="monotone"
                        dataKey="joint_attribution_change"
                        name="Joint attribution change"
                        stroke="#159a91"
                        strokeWidth={2}
                        connectNulls
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                <div className="re-table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>IG steps</th>
                        <th>Completeness delta</th>
                        <th>Joint attribution change</th>
                        <th>Top joint</th>
                        <th>Top joint score</th>
                      </tr>
                    </thead>
                    <tbody>
                      {igConvergenceRows.map((row) => (
                        <tr key={row.ig_steps}>
                          <td>{row.ig_steps}</td>
                          <td>
                            {formatNumber(row.completeness_delta, 6)}
                          </td>
                          <td>
                            {row.joint_attribution_change == null
                              ? "—"
                              : formatNumber(
                                  row.joint_attribution_change,
                                  6
                                )}
                          </td>
                          <td>{row.top_joint || "—"}</td>
                          <td>{formatNumber(row.top_joint_score, 6)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <p className="re-note">
                  A smaller attribution-change value suggests the
                  normalized joint attribution distribution changed less
                  between successive runs. These measurements help assess
                  numerical stability; three step counts alone do not
                  prove full convergence or clinical validity.
                </p>
              </>
            ) : (
              <p>
                No IG convergence results loaded. Confirm that
                <code> ig_convergence.csv </code>
                is in <code>backend/data/</code> and that the API route
                <code> /api/explainability/ig-convergence </code>
                is running.
              </p>
            )}
          </Panel>

          <p className="re-note">
            Integrated Gradients scores are model attributions, not
            clinical measures. Absolute attribution magnitude does not
            indicate whether a feature increases or decreases the selected
            class score.
          </p>
        </>
      )}

      {section === "methodComparison" && (
        <>
          {comparisonError && (
            <div className="re-warning">
              Could not load Grad-CAM vs IG comparison: {comparisonError}
            </div>
          )}

          {!comparisonSummary && !comparisonError && (
            <p>Comparison results are not available yet.</p>
          )}

          {comparisonSummary && (
            <>
              <div className="re-metrics">
                <Metric
                  label="Matched joints"
                  value={comparisonSummary.matched_joint_count ?? "—"}
                  detail="Present in both explanation datasets"
                />
                <Metric
                  label="Spearman rank correlation"
                  value={formatNumber(
                    comparisonSummary.spearman_rank_correlation,
                    4
                  )}
                  detail="Similarity of joint rankings"
                />
                <Metric
                  label="Top-five overlap"
                  value={`${
                    comparisonSummary.top_5_overlap_count ?? "—"
                  } / 5`}
                  detail="Shared top-ranked joints"
                />
                <Metric
                  label="Shared top-five joints"
                  value={
                    comparisonSummary.top_5_overlap_joints?.length ?? 0
                  }
                  detail="Joint names listed below"
                />
              </div>

              <Panel
                title="Grad-CAM vs Integrated Gradients"
                subtitle="Compare normalized joint importance and ranking across the joints present in both datasets."
              >
                <p>
                  <strong>Shared top-five joints:</strong>{" "}
                  {comparisonSummary.top_5_overlap_joints?.join(", ") ||
                    "None"}
                </p>

                <div className="re-controls">
                  <span>
                    {comparisonRows.length} matched joint records
                  </span>
                  <DownloadCsv
                    filename="gradcam-vs-ig-joint-comparison.csv"
                    rows={comparisonRows}
                  />
                </div>

                {comparisonChartRows.length ? (
                  <>
                    <div className="re-chart">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={comparisonChartRows}
                          margin={{
                            left: 10,
                            right: 20,
                            top: 10,
                            bottom: 85,
                          }}
                        >
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis
                            dataKey="jointLabel"
                            interval={0}
                            angle={-40}
                            textAnchor="end"
                            height={100}
                            tick={{ fontSize: 10 }}
                          />
                          <YAxis
                            tickFormatter={(value) =>
                              formatNumber(value, 3)
                            }
                          />
                          <Tooltip
                            labelFormatter={(label) => `Joint: ${label}`}
                            formatter={(value, name) => [
                              formatNumber(value, 6),
                              name,
                            ]}
                          />
                          <Legend />
                          <Bar
                            dataKey="gradcam_score"
                            name="Grad-CAM normalized score"
                            fill="#6d75df"
                          />
                          <Bar
                            dataKey="ig_score"
                            name="IG normalized score"
                            fill="#d97757"
                          />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>

                    <div className="re-table-wrap">
                      <table>
                        <thead>
                          <tr>
                            <th>Joint</th>
                            <th>Grad-CAM score</th>
                            <th>Grad-CAM rank</th>
                            <th>IG score</th>
                            <th>IG rank</th>
                          </tr>
                        </thead>
                        <tbody>
                          {comparisonRows.map((row) => (
                            <tr key={row.joint}>
                              <td>{row.joint}</td>
                              <td>
                                {formatNumber(row.gradcam_score, 6)}
                              </td>
                              <td>{row.gradcam_rank}</td>
                              <td>{formatNumber(row.ig_score, 6)}</td>
                              <td>{row.ig_rank}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </>
                ) : (
                  <p>
                    No matching joints were returned. Check the
                    comparison API response and ensure the joint names
                    match between the two datasets.
                  </p>
                )}

                <p className="re-note">
                  IG scores are averaged across the available integration
                  step counts. Both methods' scores are normalized over
                  the matched joints by the backend. Rank agreement
                  indicates similarity between these saved explanations;
                  it does not establish explanation faithfulness or
                  clinical validity.
                </p>
              </Panel>
            </>
          )}
        </>
      )}

      {section === "consistency" && (
        <Panel
          title="Joint consistency ranking"
          subtitle="Sorted by mean score from the saved search19 fold1 report."
        >
          <div className="re-controls">
            <span>{consistencyRows.length} joints</span>
            <DownloadCsv
              filename="joint-consistency.csv"
              rows={consistencyRows}
            />
          </div>

          <div className="re-chart">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={consistencyRows.slice(0, 15)}
                layout="vertical"
                margin={{ left: 15, right: 24, top: 5, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis
                  dataKey="joint"
                  type="category"
                  width={145}
                  tick={{ fontSize: 11 }}
                />
                <Tooltip />
                <Bar
                  dataKey="mean_score"
                  name="Mean score"
                  fill="#159a91"
                />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="re-table-wrap">
            <table>
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
                {consistencyRows.map((row) => (
                  <tr key={row.joint}>
                    <td>{row.joint}</td>
                    <td>{formatNumber(row.mean_score, 6)}</td>
                    <td>{formatNumber(row.std_score, 6)}</td>
                    <td>{formatNumber(row.mean_rank, 2)}</td>
                    <td>{row.top_5_windows}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <p className="re-note">
            Scope: search19, fold 1. These results do not establish
            seven-fold consistency.
          </p>
        </Panel>
      )}

      {section === "velocity" && (
        <Panel
          title="Velocity perturbation experiments"
          subtitle="Compare recorded score changes by joint, perturbation factor, and movement window."
        >
          <div className="re-controls">
            <label>
              Joint
              <select
                value={selectedJoint}
                onChange={(event) => setSelectedJoint(event.target.value)}
              >
                <option>All joints</option>
                {velocityJoints.map((joint) => (
                  <option key={joint}>{joint}</option>
                ))}
              </select>
            </label>

            <label>
              Movement window
              <select
                value={velocityWindow}
                onChange={(event) => setVelocityWindow(event.target.value)}
              >
                {[1, 2, 3, 4].map((number) => (
                  <option key={number} value={String(number)}>
                    Window {number}
                  </option>
                ))}
              </select>
            </label>

            <DownloadCsv
              filename="filtered-velocity-perturbations.csv"
              rows={filteredVelocity}
            />
          </div>

          <div className="re-metrics">
            <Metric
              label="Experiments shown"
              value={filteredVelocity.length}
            />
            <Metric
              label="Largest mean absolute change"
              value={formatNumber(
                Math.max(
                  0,
                  ...filteredVelocity.map(
                    (row) =>
                      Number(row["Mean absolute score change"]) || 0
                  )
                )
              )}
            />
            <Metric
              label="Selected window"
              value={`Window ${velocityWindow}`}
            />
          </div>

          <div className="re-chart">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={velocityChart}
                margin={{ left: 8, right: 20, top: 10, bottom: 10 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="factorLabel" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="scoreChange"
                  name="Mean absolute score change"
                  stroke="#6d75df"
                  strokeWidth={2}
                />
                <Line
                  type="monotone"
                  dataKey="windowDelta"
                  name={`Window ${velocityWindow} score delta`}
                  stroke="#159a91"
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="re-table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Joint</th>
                  <th>Factor</th>
                  <th>Mean abs. change</th>
                  <th>Max abs. change</th>
                  <th>Window score</th>
                  <th>Window delta</th>
                </tr>
              </thead>
              <tbody>
                {velocityChart.map((row, index) => (
                  <tr key={`${row.Joint}-${row.Factor}-${index}`}>
                    <td>{row.Joint}</td>
                    <td>{row.Factor}</td>
                    <td>
                      {formatNumber(
                        row["Mean absolute score change"],
                        7
                      )}
                    </td>
                    <td>
                      {formatNumber(
                        row["Max absolute score change"],
                        7
                      )}
                    </td>
                    <td>{formatNumber(row.windowScore, 7)}</td>
                    <td>{formatNumber(row.windowDelta, 7)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <p className="re-note">
            These are recorded perturbation results. The dashboard does
            not run new model inference when a control changes.
          </p>
        </Panel>
      )}

      {section === "windows" && (
        <Panel
          title="Movement-window sensitivity"
          subtitle="Compare the saved mean and maximum absolute changes for each window."
        >
          <div className="re-controls">
            <DownloadCsv
              filename="window-sensitivity.csv"
              rows={data.windows}
            />
          </div>

          <div className="re-chart">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={windowRows}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="windowLabel" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar
                  dataKey="meanChange"
                  name="Mean absolute change"
                  fill="#159a91"
                />
                <Bar
                  dataKey="maxChange"
                  name="Maximum absolute change"
                  fill="#6d75df"
                />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="re-table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Window</th>
                  <th>Mean absolute change</th>
                  <th>Maximum absolute change</th>
                </tr>
              </thead>
              <tbody>
                {windowRows.map((row) => (
                  <tr key={row.Window}>
                    <td>{row.Window}</td>
                    <td>{formatNumber(row.meanChange, 8)}</td>
                    <td>{formatNumber(row.maxChange, 8)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      )}

      <footer className="re-disclaimer">
        Research prototype only. Results reflect the supplied saved files
        and should not be interpreted as a clinical diagnosis or validated
        individual CP-risk probability.
      </footer>
    </div>
  );
}
