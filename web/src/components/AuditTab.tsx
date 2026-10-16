import { useEffect, useId, useMemo, useRef, useState, type DragEvent } from "react";
import { parseCsv, toCsv } from "../lib/csv";
import { fromBrowserAudit, loadDemo, type AuditData } from "../lib/data";
import { DEFAULT_OPTIONS, type BrowserAuditOptions } from "../lib/pipeline";
import type { AuditRequest, AuditResponse } from "../worker";
import { Workspace } from "./Workspace";

const MAX_ROWS = 50_000;
const MAX_CLASSES = 50;
const MAX_BYTES = 100 * 1024 * 1024;
const MIN_ROWS = 20;
const SMALL_CLASS = 5;

interface LoadedFile {
  name: string;
  header: string[];
  rows: string[][];
}

type RunState =
  | { status: "idle" }
  | { status: "running"; stage: string; done: number; total: number }
  | { status: "cancelled" }
  | { status: "error"; message: string };

const TEXT_HINT = /^(text|sentence|content|review|comment|body|message|document|title|utterance)s?$/i;
const LABEL_HINT = /^(label|class|category|target|sentiment|tag|intent|topic|y)s?$/i;

function guessColumns(file: LoadedFile): [number, number] {
  const width = file.header.length;
  const sample = file.rows.slice(0, 500);
  const avgLength = (c: number) => sample.reduce((s, r) => s + (r[c]?.length ?? 0), 0) / Math.max(1, sample.length);
  const distinct = (c: number) => new Set(sample.map((r) => r[c] ?? "")).size;
  const columns = Array.from({ length: width }, (_, c) => c);
  const text =
    columns.find((c) => TEXT_HINT.test(file.header[c].trim())) ??
    columns.reduce((best, c) => (avgLength(c) > avgLength(best) ? c : best), 0);
  const candidates = columns.filter((c) => c !== text);
  const label =
    candidates.find((c) => LABEL_HINT.test(file.header[c].trim())) ??
    candidates.filter((c) => distinct(c) > 1).sort((a, b) => distinct(a) - distinct(b))[0] ??
    candidates[0] ??
    text;
  return [text, label];
}

function readCsvFile(name: string, text: string): LoadedFile {
  const [header = [], ...rest] = parseCsv(text);
  const rows = rest.filter((r) => r.some((field) => field.trim() !== ""));
  return { name, header: header.map((h, i) => h.trim() || `Column ${i + 1}`), rows };
}

export function AuditTab() {
  const [file, setFile] = useState<LoadedFile | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const [textCol, setTextCol] = useState(0);
  const [labelCol, setLabelCol] = useState(1);
  const [options, setOptions] = useState<BrowserAuditOptions>(DEFAULT_OPTIONS);
  const [run, setRun] = useState<RunState>({ status: "idle" });
  const [result, setResult] = useState<AuditData | null>(null);
  const [dragging, setDragging] = useState(false);
  const workerRef = useRef<Worker | null>(null);
  const resultsRef = useRef<HTMLHeadingElement>(null);
  const ids = { file: useId(), text: useId(), label: useId() };

  useEffect(() => () => workerRef.current?.terminate(), []);
  useEffect(() => {
    if (result) resultsRef.current?.focus();
  }, [result]);

  const accept = (loaded: LoadedFile) => {
    workerRef.current?.terminate();
    setFile(loaded);
    setResult(null);
    setRun({ status: "idle" });
    const [t, l] = guessColumns(loaded);
    setTextCol(t);
    setLabelCol(l);
  };

  const openFile = async (picked: File | undefined) => {
    if (!picked) return;
    setFileError(null);
    if (picked.size > MAX_BYTES) {
      setFileError(`${picked.name} is larger than 100 MB. Split it, or use the Python pipeline for large datasets.`);
      return;
    }
    try {
      const loaded = readCsvFile(picked.name, await picked.text());
      if (loaded.header.length < 2) throw new Error("The file needs at least two columns: one for text, one for labels.");
      accept(loaded);
    } catch (e) {
      setFile(null);
      setFileError(e instanceof Error ? e.message : `Could not read ${picked.name}.`);
    }
  };

  const loadSample = async () => {
    setFileError(null);
    try {
      const demo = await loadDemo(`${import.meta.env.BASE_URL}demo.json`);
      accept({
        name: "ag-news-sample.csv",
        header: ["text", "label"],
        rows: demo.rows.map((r) => [r.text, demo.classes[r.given]]),
      });
    } catch (e) {
      setFileError(e instanceof Error ? e.message : String(e));
    }
  };

  const onDrop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    setDragging(false);
    void openFile(event.dataTransfer.files[0]);
  };

  const prepared = useMemo(() => {
    if (!file) return null;
    const errors: string[] = [];
    const warnings: string[] = [];
    if (file.rows.length > MAX_ROWS) {
      errors.push(
        `This file has ${file.rows.length.toLocaleString()} rows. The in-browser audit handles up to ${MAX_ROWS.toLocaleString()} so it stays fast and fits in memory. Split the file, or run the Python pipeline for larger datasets.`,
      );
    }
    if (textCol === labelCol) errors.push("Pick two different columns for text and label.");

    const rowIds: number[] = [];
    const texts: string[] = [];
    const rawLabels: string[] = [];
    file.rows.forEach((r, i) => {
      const text = (r[textCol] ?? "").trim();
      const label = (r[labelCol] ?? "").trim();
      if (text && label) {
        rowIds.push(i);
        texts.push(text);
        rawLabels.push(label);
      }
    });
    const counts = new Map<string, number>();
    for (const l of rawLabels) counts.set(l, (counts.get(l) ?? 0) + 1);
    const classes = [...counts.keys()].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
    const index = new Map(classes.map((c, i) => [c, i]));
    const labels = rawLabels.map((l) => index.get(l)!);

    const skipped = file.rows.length - texts.length;
    if (skipped > 0) {
      warnings.push(
        `${skipped.toLocaleString()} ${skipped === 1 ? "row has" : "rows have"} an empty text or label. They are left out of the audit and kept unchanged in the export.`,
      );
    }
    if (classes.length > MAX_CLASSES) {
      errors.push(
        `The label column has ${classes.length.toLocaleString()} different values; the audit supports up to ${MAX_CLASSES} classes. Is it the right column?`,
      );
    } else if (classes.length < 2) {
      errors.push("The label column needs at least two different values.");
    }
    if (texts.length < MIN_ROWS) errors.push(`At least ${MIN_ROWS} rows with both a text and a label are needed.`);
    const small = classes.filter((c) => counts.get(c)! < SMALL_CLASS);
    if (small.length && classes.length <= MAX_CLASSES) {
      warnings.push(
        `${small.length === 1 ? "One class has" : `${small.length} classes have`} fewer than ${SMALL_CLASS} examples (${small.slice(0, 5).join(", ")}${small.length > 5 ? ", …" : ""}). Rows in tiny classes tend to look suspicious simply because the model sees few examples of them.`,
      );
    }
    return { rowIds, texts, labels, classes, counts, errors, warnings };
  }, [file, textCol, labelCol]);

  const start = () => {
    if (!prepared || prepared.errors.length) return;
    workerRef.current?.terminate();
    const worker = new Worker(new URL("../worker.ts", import.meta.url), { type: "module" });
    workerRef.current = worker;
    setResult(null);
    setRun({ status: "running", stage: "Starting", done: 0, total: 1 });
    const { rowIds, texts, labels, classes } = prepared;
    const runOptions = options;
    worker.onmessage = ({ data }: MessageEvent<AuditResponse>) => {
      if (data.type === "progress") {
        setRun({ status: "running", stage: data.stage, done: data.done, total: data.total });
      } else if (data.type === "done") {
        worker.terminate();
        workerRef.current = null;
        setRun({ status: "idle" });
        setResult(fromBrowserAudit(rowIds, texts, labels, classes, data.result, runOptions));
      } else {
        worker.terminate();
        workerRef.current = null;
        setRun({ status: "error", message: data.message });
      }
    };
    worker.onerror = (event) => {
      worker.terminate();
      workerRef.current = null;
      setRun({ status: "error", message: event.message || "The audit stopped unexpectedly." });
    };
    const request: AuditRequest = { texts, labels, numClasses: classes.length, options: runOptions };
    worker.postMessage(request);
  };

  const cancel = () => {
    workerRef.current?.terminate();
    workerRef.current = null;
    setRun({ status: "cancelled" });
  };

  const changeSetup = (apply: () => void) => {
    apply();
    setResult(null);
  };

  const running = run.status === "running";
  const setOption = (key: keyof BrowserAuditOptions, min: number, max: number) => (value: string) =>
    changeSetup(() => setOptions((o) => ({ ...o, [key]: Math.max(min, Math.min(max, Math.round(Number(value) || min))) })));

  return (
    <>
      <div className="privacy" role="note">
        <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
          <path
            d="M7 10V7a5 5 0 0 1 10 0v3M5.5 10h13a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1h-13a1 1 0 0 1-1-1v-9a1 1 0 0 1 1-1Z"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
          />
        </svg>
        <p>
          <strong>Your data never leaves this browser.</strong> The file is read on your device and audited in a
          background thread of this tab. Nothing is uploaded, and the page loads no third-party scripts or analytics.
        </p>
      </div>

      <ol className="steps">
        <li className="step">
          <h3>Choose a CSV file</h3>
          <input
            id={ids.file}
            className="visually-hidden-input"
            type="file"
            accept=".csv,.tsv,text/csv,text/tab-separated-values"
            disabled={running}
            onChange={(event) => void openFile(event.target.files?.[0])}
          />
          <label
            htmlFor={ids.file}
            className="dropzone"
            data-dragging={dragging || undefined}
            onDragOver={(event) => {
              event.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
          >
            <span className="dropzone-title">{file ? file.name : "Drop a CSV here or browse"}</span>
            <span className="muted">
              {file
                ? `${file.rows.length.toLocaleString()} rows · ${file.header.length} columns`
                : `One row per example, with a header row. Up to ${MAX_ROWS.toLocaleString()} rows and ${MAX_CLASSES} classes.`}
            </span>
          </label>
          <p className="hint">
            No file handy?{" "}
            <button type="button" className="link-button" onClick={loadSample} disabled={running}>
              Use the AG News sample as a CSV
            </button>
          </p>
          {fileError && (
            <p className="notice notice-error" role="alert">
              {fileError}
            </p>
          )}
        </li>

        {file && prepared && (
          <li className="step">
            <h3>Pick the columns</h3>
            <div className="field-row">
              <div className="field">
                <label htmlFor={ids.text}>Text column</label>
                <select
                  id={ids.text}
                  value={textCol}
                  disabled={running}
                  onChange={(event) => changeSetup(() => setTextCol(Number(event.target.value)))}
                >
                  {file.header.map((h, i) => (
                    <option key={i} value={i}>
                      {h}
                    </option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label htmlFor={ids.label}>Label column</label>
                <select
                  id={ids.label}
                  value={labelCol}
                  disabled={running}
                  onChange={(event) => changeSetup(() => setLabelCol(Number(event.target.value)))}
                >
                  {file.header.map((h, i) => (
                    <option key={i} value={i}>
                      {h}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {prepared.errors.length === 0 && (
              <>
                <div className="table-wrap">
                  <table className="preview">
                    <caption>First rows</caption>
                    <thead>
                      <tr>
                        <th scope="col">{file.header[textCol]}</th>
                        <th scope="col">{file.header[labelCol]}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {prepared.texts.slice(0, 3).map((t, i) => (
                        <tr key={i}>
                          <td>{t.length > 140 ? `${t.slice(0, 140)}…` : t}</td>
                          <td>{prepared.classes[prepared.labels[i]]}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="class-summary">
                  {prepared.texts.length.toLocaleString()} rows in {prepared.classes.length} classes:{" "}
                  {prepared.classes.slice(0, 12).map((c) => (
                    <span key={c} className="chip">
                      {c} <span className="muted">{prepared.counts.get(c)!.toLocaleString()}</span>
                    </span>
                  ))}
                  {prepared.classes.length > 12 && <span className="muted"> and {prepared.classes.length - 12} more</span>}
                </p>
              </>
            )}
            {prepared.errors.map((message) => (
              <p key={message} className="notice notice-error" role="alert">
                {message}
              </p>
            ))}
            {prepared.warnings.map((message) => (
              <p key={message} className="notice">
                {message}
              </p>
            ))}
          </li>
        )}

        {file && prepared && prepared.errors.length === 0 && (
          <li className="step">
            <h3>Run the audit</h3>
            <details className="settings">
              <summary>Training settings</summary>
              <div className="field-row">
                {(
                  [
                    ["folds", "Cross-validation folds", 2, 10],
                    ["cvEpochs", "Epochs per fold", 1, 30],
                    ["seeds", "Trajectory seeds", 1, 6],
                    ["trajectoryEpochs", "Trajectory epochs", 3, 40],
                  ] as const
                ).map(([key, label, min, max]) => (
                  <div className="field" key={key}>
                    <label htmlFor={`${ids.file}-${key}`}>{label}</label>
                    <input
                      id={`${ids.file}-${key}`}
                      type="number"
                      min={min}
                      max={max}
                      value={options[key]}
                      disabled={running}
                      onChange={(event) => setOption(key, min, max)(event.target.value)}
                    />
                  </div>
                ))}
              </div>
            </details>
            {running ? (
              <div className="progress" role="status" aria-live="polite">
                <div className="progress-head">
                  <span>{run.stage}</span>
                  <span className="num">{Math.round((run.done / run.total) * 100)}%</span>
                </div>
                <progress max={run.total} value={run.done} aria-label="Audit progress" />
                <button type="button" className="button-secondary" onClick={cancel}>
                  Cancel
                </button>
              </div>
            ) : (
              <button type="button" className="button-primary" onClick={start}>
                {result ? "Run again" : "Audit this file"}
              </button>
            )}
            {run.status === "cancelled" && (
              <p className="notice" role="status">
                Audit cancelled. Nothing was kept.
              </p>
            )}
            {run.status === "error" && (
              <p className="notice notice-error" role="alert">
                The audit failed: {run.message}
              </p>
            )}
          </li>
        )}
      </ol>

      {result && file && (
        <section className="audit-results" aria-labelledby="results-heading">
          <h2 id="results-heading" ref={resultsRef} tabIndex={-1}>
            Results for {file.name}
          </h2>
          <p className="muted">
            TF-IDF features and logistic regression trained in this tab: {result.folds}-fold confident learning and loss
            curves over {result.seeds} seeds × {result.epochs} epochs.
          </p>
          <Workspace
            data={result}
            fileName={file.name.replace(/\.(csv|tsv)$/i, "") + "-cleaned.csv"}
            buildCsv={(removed) => toCsv([file.header, ...file.rows.filter((_, i) => !removed.has(i))])}
          />
        </section>
      )}
    </>
  );
}
