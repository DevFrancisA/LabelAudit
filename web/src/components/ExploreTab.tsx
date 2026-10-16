import { useCallback, useEffect, useState } from "react";
import { toCsv } from "../lib/csv";
import { loadDemo, type DemoData } from "../lib/data";
import { Workspace } from "./Workspace";

const DEMO_URL = `${import.meta.env.BASE_URL}demo.json`;

export function ExploreTab() {
  const [data, setData] = useState<DemoData | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    setError(null);
    loadDemo(DEMO_URL).then(setData, (e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  }, []);
  useEffect(load, [load]);

  if (error) {
    return (
      <div className="notice notice-error" role="alert">
        <p>{error}</p>
        <button type="button" className="button-secondary" onClick={load}>
          Try again
        </button>
      </div>
    );
  }
  if (!data) {
    return (
      <p className="loading" role="status">
        Loading the sample…
      </p>
    );
  }

  const buildCsv = (removed: Set<number>) =>
    toCsv([["text", "label"], ...data.rows.filter((r) => !removed.has(r.id)).map((r) => [r.text, data.classes[r.given]])]);

  return (
    <>
      <div className="intro">
        <p>
          A random sample of {data.rows.length.toLocaleString()} news articles from the{" "}
          {data.auditedExamples.toLocaleString()} AG News training examples the Python pipeline audited. Each article is
          labeled World, Sports, Business, or Sci/Tech. Before the audit we flipped some labels on purpose, choosing
          confusions people actually make, such as Business for Sci/Tech.
        </p>
        <p className="muted">
          Python version: frozen <code>{data.encoder.split("/").pop()}</code> sentence embeddings, an MLP head,{" "}
          {data.folds}-fold confident learning, and loss curves over {data.seeds} seeds × {data.epochs} epochs. Scores
          here are the pipeline's own output.
        </p>
      </div>
      <Workspace data={data} buildCsv={buildCsv} fileName="ag-news-sample-cleaned.csv" revealable />
    </>
  );
}
