import { useId, useState } from "react";
import type { AuditRow } from "../lib/data";
import { Sparkline } from "./Sparkline";

interface RowCardProps {
  row: AuditRow;
  rank: number;
  classes: string[];
  flagged: boolean;
  removed: boolean;
  onToggle: (id: number, remove: boolean) => void;
  showInjected: boolean;
  lossMax: number;
  epochs: number;
}

const LONG_TEXT = 320;

export function RowCard({ row, rank, classes, flagged, removed, onToggle, showInjected, lossMax, epochs }: RowCardProps) {
  const [expanded, setExpanded] = useState(false);
  const textId = useId();
  const isLong = row.text.length > LONG_TEXT;
  const agrees = row.suggested === row.given;
  const neverLearned = row.forgetting >= epochs;

  return (
    <li className="row" data-flagged={flagged || undefined} data-removed={removed || undefined}>
      <div className="row-rank" aria-label={`Rank ${rank}`}>
        {rank}
      </div>
      <article className="row-body" aria-labelledby={textId}>
        <div className="row-labels">
          <span className="chip">
            <span className="chip-key">Given</span> {classes[row.given]}
          </span>
          {agrees ? (
            <span className="chip chip-quiet">Model agrees</span>
          ) : (
            <span className="chip chip-suggest">
              <span className="chip-key">Suggested</span> {classes[row.suggested]}
            </span>
          )}
          {flagged && <span className="chip chip-flag">Flagged</span>}
          {showInjected && row.injected && <span className="chip chip-injected">Flipped on purpose</span>}
        </div>
        <p id={textId} className="row-text" data-clamped={isLong && !expanded ? true : undefined}>
          {row.text}
        </p>
        {isLong && (
          <button type="button" className="link-button" aria-expanded={expanded} onClick={() => setExpanded(!expanded)}>
            {expanded ? "Show less" : "Show full text"}
          </button>
        )}
        <dl className="signals">
          <div className="signal signal-a">
            <dt>
              <span className="key-swatch" aria-hidden="true" /> A · confidence in given label
            </dt>
            <dd>
              <span className="meter" aria-hidden="true">
                <span style={{ width: `${Math.max(2, row.selfConfidence * 100)}%` }} />
              </span>
              <span className="num">{(row.selfConfidence * 100).toFixed(1)}%</span>
              <span className="muted">suspicion {row.signalA.toFixed(2)}</span>
            </dd>
          </div>
          <div className="signal signal-b">
            <dt>
              <span className="key-swatch" aria-hidden="true" /> B · training loss per epoch
            </dt>
            <dd>
              <Sparkline values={row.trajectory} max={lossMax} chance={Math.log(classes.length)} />
              <span className="signal-b-stats">
                <span className="num">mean {row.meanLoss.toFixed(2)}</span>
                <span className="num">late {row.lateLoss.toFixed(2)}</span>
                <span className="num">{neverLearned ? "never learned" : `forgetting ${row.forgetting.toFixed(1)}`}</span>
                <span className="muted">suspicion {row.signalB.toFixed(2)}</span>
              </span>
            </dd>
          </div>
        </dl>
      </article>
      <label className="row-toggle">
        <input type="checkbox" checked={removed} onChange={(event) => onToggle(row.id, event.target.checked)} />
        <span>Remove</span>
      </label>
    </li>
  );
}
