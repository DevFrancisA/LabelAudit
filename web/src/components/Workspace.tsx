import { useId, useMemo, useState } from "react";
import type { AuditData } from "../lib/data";
import { combine, flagRates, ranking, selectFlags } from "../lib/scoring";
import { FlagRateTable } from "./FlagRateTable";
import { RowCard } from "./RowCard";

type RankBy = "combined" | "a" | "b";

interface WorkspaceProps {
  data: AuditData;
  /** Build the cleaned CSV given the ids of rows to drop. */
  buildCsv: (removed: Set<number>) => string;
  fileName: string;
  /** Offer the "reveal injected labels" switch (demo only). */
  revealable?: boolean;
}

const PAGE = 25;

const RANK_OPTIONS: { value: RankBy; label: string; hint: string }[] = [
  { value: "combined", label: "Combined", hint: "Blend of both signals" },
  { value: "a", label: "Signal A only", hint: "Confident learning" },
  { value: "b", label: "Signal B only", hint: "Loss trajectory" },
];

function quantile(values: number[], q: number): number {
  if (values.length === 0) return 1;
  const sorted = values.slice().sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
}

export function Workspace({ data, buildCsv, fileName, revealable = false }: WorkspaceProps) {
  const { rows, classes } = data;
  const n = rows.length;
  const labels = useMemo(() => rows.map((r) => r.given), [rows]);
  const defaultFlags = Math.min(n, data.suggestedFlagCount);

  const [rankBy, setRankBy] = useState<RankBy>("combined");
  const [blend, setBlend] = useState(Math.round(data.defaultBlend * 100));
  const [flagCount, setFlagCount] = useState(defaultFlags);
  const [capOn, setCapOn] = useState(data.defaultClassCap !== null);
  const [capPercent, setCapPercent] = useState(Math.round((data.defaultClassCap ?? 0.5) * 100));
  const [classFilter, setClassFilter] = useState(-1);
  const [showUnflagged, setShowUnflagged] = useState(false);
  const [reveal, setReveal] = useState(false);
  const [overrides, setOverrides] = useState<Map<number, boolean>>(new Map());
  const [visible, setVisible] = useState(PAGE);
  const ids = { blend: useId(), flags: useId(), cap: useId(), cls: useId() };

  const order = useMemo(() => {
    const a = rows.map((r) => r.signalA);
    const b = rows.map((r) => r.signalB);
    const scores = rankBy === "a" ? a : rankBy === "b" ? b : combine(a, b, blend, 100 - blend);
    return ranking(scores);
  }, [rows, rankBy, blend]);

  const classCap = capOn ? capPercent / 100 : null;
  const flagged = useMemo(
    () => new Set(selectFlags(order, labels, classes.length, flagCount, classCap)),
    [order, labels, classes.length, flagCount, classCap],
  );
  const rates = useMemo(() => flagRates(labels, flagged, classes.length), [labels, flagged, classes.length]);
  const lossMax = useMemo(() => Math.max(0.5, quantile(rows.flatMap((r) => r.trajectory), 0.99)), [rows]);

  const isRemoved = (index: number) => overrides.get(rows[index].id) ?? flagged.has(index);
  const removedIds = useMemo(() => {
    const ids = new Set<number>();
    rows.forEach((r, i) => (overrides.get(r.id) ?? flagged.has(i)) && ids.add(r.id));
    return ids;
  }, [rows, overrides, flagged]);

  const listed = useMemo(
    () => order.filter((i) => (classFilter < 0 || labels[i] === classFilter) && (showUnflagged || flagged.has(i))),
    [order, classFilter, labels, showUnflagged, flagged],
  );
  const rankOf = useMemo(() => {
    const ranks = new Int32Array(n);
    order.forEach((i, r) => (ranks[i] = r + 1));
    return ranks;
  }, [order, n]);

  const toggle = (id: number, remove: boolean) =>
    setOverrides((current) => new Map(current).set(id, remove));

  const download = () => {
    const blob = new Blob([buildCsv(removedIds)], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = fileName;
    link.click();
    URL.revokeObjectURL(url);
  };

  const resetPage = () => setVisible(PAGE);

  return (
    <div className="workspace">
      <aside className="controls" aria-label="Ranking and cleaning controls">
        <fieldset className="control">
          <legend>Signal used for ranking</legend>
          <div className="segmented">
            {RANK_OPTIONS.map((option) => (
              <label key={option.value} title={option.hint}>
                <input
                  type="radio"
                  name="rank-by"
                  value={option.value}
                  checked={rankBy === option.value}
                  onChange={() => {
                    setRankBy(option.value);
                    resetPage();
                  }}
                />
                <span>{option.label}</span>
              </label>
            ))}
          </div>
        </fieldset>

        {rankBy === "combined" && (
          <div className="control">
            <label htmlFor={ids.blend}>Blend</label>
            <input
              id={ids.blend}
              type="range"
              min={0}
              max={100}
              step={5}
              value={blend}
              aria-valuetext={`${blend}% signal A, ${100 - blend}% signal B`}
              onChange={(event) => setBlend(Number(event.target.value))}
            />
            <div className="range-ends">
              <span>A {blend}%</span>
              <span>B {100 - blend}%</span>
            </div>
          </div>
        )}

        <div className="control">
          <label id={`${ids.flags}-label`} htmlFor={ids.flags}>
            Flag the top
          </label>
          <div className="range-with-number">
            <input
              type="range"
              min={0}
              max={n}
              value={flagCount}
              aria-labelledby={`${ids.flags}-label`}
              onChange={(event) => {
                setFlagCount(Number(event.target.value));
                resetPage();
              }}
            />
            <input
              id={ids.flags}
              type="number"
              min={0}
              max={n}
              value={flagCount}
              onChange={(event) => setFlagCount(Math.max(0, Math.min(n, Number(event.target.value) || 0)))}
            />
          </div>
          <p className="hint">
            Starts at confident learning's estimate of how many labels are wrong ({defaultFlags.toLocaleString()}).{" "}
            {flagCount !== defaultFlags && (
              <button type="button" className="link-button" onClick={() => setFlagCount(defaultFlags)}>
                Reset
              </button>
            )}
          </p>
        </div>

        <div className="control">
          <label className="check">
            <input type="checkbox" checked={capOn} onChange={(event) => setCapOn(event.target.checked)} />
            <span>Cap flags per class</span>
          </label>
          {capOn && (
            <div className="inline-field">
              <label htmlFor={ids.cap}>At most</label>
              <input
                id={ids.cap}
                type="number"
                min={1}
                max={100}
                value={capPercent}
                onChange={(event) => setCapPercent(Math.max(1, Math.min(100, Number(event.target.value) || 1)))}
              />
              <span>% of any class</span>
            </div>
          )}
          <p className="hint">Keeps rare classes from being stripped out.</p>
        </div>

        <div className="control">
          <label htmlFor={ids.cls}>Class</label>
          <select
            id={ids.cls}
            value={classFilter}
            onChange={(event) => {
              setClassFilter(Number(event.target.value));
              resetPage();
            }}
          >
            <option value={-1}>All classes</option>
            {classes.map((name, c) => (
              <option key={name} value={c}>
                {name}
              </option>
            ))}
          </select>
          <label className="check">
            <input
              type="checkbox"
              checked={showUnflagged}
              onChange={(event) => {
                setShowUnflagged(event.target.checked);
                resetPage();
              }}
            />
            <span>Also list unflagged rows</span>
          </label>
        </div>

        {revealable && (
          <div className="control control-reveal">
            <label className="switch">
              <input type="checkbox" role="switch" checked={reveal} onChange={(event) => setReveal(event.target.checked)} />
              <span>Reveal injected labels</span>
            </label>
            <p className="hint">Marks rows whose label we flipped before the audit. The ranking never saw this.</p>
          </div>
        )}

        <div className="control control-export">
          <p>
            <strong>{removedIds.size.toLocaleString()}</strong> rows marked for removal,{" "}
            <strong>{(n - removedIds.size).toLocaleString()}</strong> kept.
          </p>
          <button type="button" className="button-primary" onClick={download}>
            Download cleaned CSV
          </button>
          {overrides.size > 0 && (
            <button type="button" className="link-button" onClick={() => setOverrides(new Map())}>
              Undo {overrides.size} manual {overrides.size === 1 ? "choice" : "choices"}
            </button>
          )}
        </div>
      </aside>

      <section className="results" aria-label="Ranked examples">
        <FlagRateTable
          rates={rates}
          classes={classes}
          capCount={(total) => (classCap === null ? null : Math.floor(classCap * total))}
        />
        <h3 className="list-heading" aria-live="polite">
          {listed.length === 0
            ? "No rows to show"
            : `Showing ${Math.min(visible, listed.length).toLocaleString()} of ${listed.length.toLocaleString()} ${showUnflagged ? "rows" : "flagged rows"}, most suspicious first`}
        </h3>
        {listed.length === 0 ? (
          <p className="empty">
            Nothing is flagged {classFilter >= 0 ? `in ${classes[classFilter]} ` : ""}at this setting. Raise the
            flag count or list unflagged rows too.
          </p>
        ) : (
          <ol className="rows">
            {listed.slice(0, visible).map((i) => (
              <RowCard
                key={rows[i].id}
                row={rows[i]}
                rank={rankOf[i]}
                classes={classes}
                flagged={flagged.has(i)}
                removed={isRemoved(i)}
                onToggle={toggle}
                showInjected={reveal}
                lossMax={lossMax}
                epochs={data.epochs}
              />
            ))}
          </ol>
        )}
        {visible < listed.length && (
          <button type="button" className="button-secondary more" onClick={() => setVisible(visible + PAGE)}>
            Show {Math.min(PAGE, listed.length - visible)} more
          </button>
        )}
      </section>
    </div>
  );
}
