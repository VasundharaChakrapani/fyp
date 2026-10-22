
import { useEffect, useMemo, useState } from "react";

const API = (
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/$/, "");

const JOINTS = [
  "head_top", "nose", "right_ear", "left_ear", "upper_neck",
  "right_shoulder", "right_elbow", "right_wrist", "thorax",
  "left_shoulder", "left_elbow", "left_wrist", "pelvis",
  "right_hip", "right_knee", "right_ankle",
  "left_hip", "left_knee", "left_ankle",
  "right_little_finger", "right_index_finger",
  "left_little_finger", "left_index_finger",
  "right_heel", "right_little_toe", "right_big_toe",
  "left_heel", "left_little_toe", "left_big_toe",
];

// Display connections. Confirm these against the ST-GCN graph
// before treating the drawing as the model's exact skeleton topology.
const BONES = [
  ["head_top", "nose"],
  ["head_top", "right_ear"],
  ["head_top", "left_ear"],
  ["nose", "upper_neck"],
  ["upper_neck", "thorax"],
  ["upper_neck", "right_shoulder"],
  ["right_shoulder", "right_elbow"],
  ["right_elbow", "right_wrist"],
  ["upper_neck", "left_shoulder"],
  ["left_shoulder", "left_elbow"],
  ["left_elbow", "left_wrist"],
  ["thorax", "pelvis"],
  ["pelvis", "right_hip"],
  ["right_hip", "right_knee"],
  ["right_knee", "right_ankle"],
  ["pelvis", "left_hip"],
  ["left_hip", "left_knee"],
  ["left_knee", "left_ankle"],
  ["right_wrist", "right_index_finger"],
  ["right_wrist", "right_little_finger"],
  ["left_wrist", "left_index_finger"],
  ["left_wrist", "left_little_finger"],
  ["right_ankle", "right_heel"],
  ["right_ankle", "right_little_toe"],
  ["right_ankle", "right_big_toe"],
  ["left_ankle", "left_heel"],
  ["left_ankle", "left_little_toe"],
  ["left_ankle", "left_big_toe"],
];

function colorForScore(score, min, max) {
  if (!Number.isFinite(score)) return "#94a3b8";
  const t = max > min ? (score - min) / (max - min) : 0.5;
  const hue = 240 * (1 - Math.max(0, Math.min(1, t)));
  return `hsl(${hue}, 90%, 48%)`;
}

function humanize(value) {
  return String(value).replaceAll("_", " ").replace(/\b\w/g, c => c.toUpperCase());
}

async function fetchJson(path) {
  const response = await fetch(`${API}${path}`);
  if (!response.ok) {
    let detail = "";
    try {
      detail = (await response.json()).detail || "";
    } catch {
      // Ignore non-JSON error bodies.
    }
    throw new Error(detail || `${path} returned HTTP ${response.status}`);
  }
  return response.json();
}

export default function GradCAMSkeleton() {
  const [frames, setFrames] = useState([]);
  const [scoreRows, setScoreRows] = useState([]);
  const [frameIndex, setFrameIndex] = useState(0);
  const [windowNumber, setWindowNumber] = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setLoading(true);
      setError("");

      try {
        const [coordinateData, scoreData] = await Promise.all([
          fetchJson("/api/movement/coordinates"),
          fetchJson("/api/explainability/joint-scores"),
        ]);

        const coordinateRows = coordinateData.results;
        const attributionRows = scoreData.results;

        if (!Array.isArray(coordinateRows) || coordinateRows.length === 0) {
          throw new Error("The coordinates endpoint returned no frames.");
        }

        if (!Array.isArray(attributionRows) || attributionRows.length === 0) {
          throw new Error("The Grad-CAM endpoint returned no joint scores.");
        }

        if (!cancelled) {
          setFrames(coordinateRows);
          setScoreRows(attributionRows);
          setFrameIndex(0);

          const availableWindows = attributionRows
            .map(row => Number(row.window))
            .filter(Number.isFinite);

          if (availableWindows.length) {
            setWindowNumber(Math.min(...availableWindows));
          }
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Unable to load data.");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const availableWindows = useMemo(
    () => [...new Set(
      scoreRows.map(row => Number(row.window)).filter(Number.isFinite)
    )].sort((a, b) => a - b),
    [scoreRows]
  );

  const scoreMap = useMemo(() => {
    const row = scoreRows.find(
      item => Number(item.window) === Number(windowNumber)
    );

    if (!row) return {};

    return Object.fromEntries(
      JOINTS.map(joint => {
        const value = Number(row[joint]);
        return [joint, row[joint] != null && row[joint] !== "" &&
          Number.isFinite(value) ? value : null];
      })
    );
  }, [scoreRows, windowNumber]);

  const validScores = Object.values(scoreMap).filter(Number.isFinite);
  const minScore = validScores.length ? Math.min(...validScores) : 0;
  const maxScore = validScores.length ? Math.max(...validScores) : 1;

  const frame = frames[frameIndex];

  const pointFor = name => {
    if (!frame) return null;

    const x = Number(frame[`${name}_x`]);
    const y = Number(frame[`${name}_y`]);
    const confidence = Number(frame[`${name}_conf`] ?? 1);

    if (!Number.isFinite(x) || !Number.isFinite(y)) return null;

    // CSV coordinates are normalized. Preserve their relative positions.
    return { x: x * 600, y: y * 600, confidence };
  };

  if (loading) {
    return <section className="panel gc-panel">Loading skeleton and Grad-CAM data…</section>;
  }

  if (error) {
    return (
      <section className="panel gc-panel">
        <h2>Skeleton Grad-CAM visualization</h2>
        <p className="error-text">{error}</p>
        <p className="chart-note">
          Confirm that FastAPI is running and both API endpoints return data.
        </p>
      </section>
    );
  }

  if (!frame) {
    return <section className="panel gc-panel">No frame data available.</section>;
  }

  return (
    <section className="panel gc-panel">
      <div className="panel-heading">
        <div>
          <h2>Visual Grad-CAM joint attribution</h2>
          <p>Skeleton coordinates with saved window-level joint scores</p>
        </div>
      </div>

      <div className="gc-controls">
        <label>
          Attribution window
          <select
            value={windowNumber}
            onChange={event => setWindowNumber(Number(event.target.value))}
          >
            {availableWindows.map(window => (
              <option key={window} value={window}>Window {window}</option>
            ))}
          </select>
        </label>

        <label>
          Frame {frame.frame ?? frameIndex + 1} of {frames.length}
          <input
            type="range"
            min="0"
            max={frames.length - 1}
            value={frameIndex}
            onChange={event => setFrameIndex(Number(event.target.value))}
          />
        </label>
      </div>

      <div className="gc-canvas">
        <svg viewBox="0 0 600 600" role="img"
          aria-label={`Skeleton at frame ${frame.frame ?? frameIndex + 1}, colored by window ${windowNumber} joint attribution`}>
          {BONES.map(([from, to]) => {
            const a = pointFor(from);
            const b = pointFor(to);
            if (!a || !b || a.confidence < 0.1 || b.confidence < 0.1) return null;

            return (
              <line
                key={`${from}-${to}`}
                x1={a.x} y1={a.y}
                x2={b.x} y2={b.y}
                stroke="#64748b"
                strokeWidth="4"
                strokeLinecap="round"
              />
            );
          })}

          {JOINTS.map(joint => {
            const point = pointFor(joint);
            if (!point || point.confidence < 0.1) return null;

            const score = scoreMap[joint];

            return (
              <g key={joint}>
                <circle
                  cx={point.x}
                  cy={point.y}
                  r="9"
                  fill={colorForScore(score, minScore, maxScore)}
                  stroke="white"
                  strokeWidth="2"
                >
                  <title>
                    {humanize(joint)}: {Number.isFinite(score) ? score : "No matching score"}
                  </title>
                </circle>
              </g>
            );
          })}
        </svg>
      </div>

      <div className="gc-legend">
        <span>Lower attribution</span>
        <div className="gc-gradient" />
        <span>Higher attribution</span>
      </div>

      <p className="chart-note">
        Colors show the selected window's saved Grad-CAM joint scores. The
        same window-level scores are displayed on each frame; they are not
        per-frame Grad-CAM maps. Gray joints indicate missing scores.
        This research visualization is not clinically validated.
      </p>
    </section>
  );
}
