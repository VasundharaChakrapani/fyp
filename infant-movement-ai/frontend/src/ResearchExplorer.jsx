
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
};


async function fetchRecords(endpoint) {
  const response = await fetch(`${API}${endpoint}`);

  if (!response.ok) {
    throw new Error(
      `Request failed (${response.status}): ${endpoint}`
    );
  }

  const json = await response.json();

  // Support both response formats used by the API.
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
    <button className="re-button" onClick={download} disabled={!rows.length}>
      Download CSV
    </button>
  );
}

function formatNumber(value, digits = 5) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "—";
}

export default function ResearchExplorer() {
  const [data, setData] = useState({
    explainability: [],
    consistency: [],
    velocity: [],
    windows: [],
  });
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

      const results = await Promise.allSettled(
        Object.entries(ENDPOINTS).map(async ([key, endpoint]) => [
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
          failures.push(result.reason?.message || "Unknown request error");
        }
      }

      setData((previous) => ({ ...previous, ...nextData }));

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
        (row) => selectedJoint === "All joints" || row.Joint === selectedJoint
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

  if (loading) {
    return <div className="re-status">Loading research datasets…</div>;
  }

  return (
    <div className="research-explorer">
      <header className="re-header">
        <div>
          <p className="re-eyebrow">RESEARCH WORKSPACE</p>
          <h2>Explainability & Perturbation Explorer</h2>
          <p>
            Explore recorded joint-level explanations, kinematic perturbations,
            and movement-window sensitivity.
          </p>
        </div>
        <span className="re-status-pill">Saved experiment results</span>
      </header>

      {error && <div className="re-warning">{error}</div>}

      <div className="re-tabs">
        {[
          ["explainability", "Joint explanations"],
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
                    <Tooltip formatter={(value) => formatNumber(value, 6)} />
                    <Bar dataKey="score" name="Grad-CAM score" fill="#6d75df" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <p>No explanation scores are available for this window.</p>
            )}
            <p className="re-note">
              These are saved model explanation scores, not clinical measures.
              The chart displays the 15 highest-scoring joints for the selected
              window.
            </p>
          </Panel>
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
                <Bar dataKey="mean_score" name="Mean score" fill="#159a91" />
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
                  ...filteredVelocity.map((row) =>
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
                      {formatNumber(row["Mean absolute score change"], 7)}
                    </td>
                    <td>
                      {formatNumber(row["Max absolute score change"], 7)}
                    </td>
                    <td>{formatNumber(row.windowScore, 7)}</td>
                    <td>{formatNumber(row.windowDelta, 7)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="re-note">
            These are recorded perturbation results. The dashboard does not run
            new model inference when a control changes.
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
        Research prototype only. Results reflect the supplied saved files and
        should not be interpreted as a clinical diagnosis or validated
        individual CP-risk probability.
      </footer>
    </div>
  );
}
