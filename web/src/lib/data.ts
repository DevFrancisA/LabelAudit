import type { BrowserAuditOptions, BrowserAuditResult } from "./pipeline";

/** One audited example, as both tabs display it. */
export interface AuditRow {
  /** Position in the source data (row index for uploads, sample index for the demo). */
  id: number;
  text: string;
  given: number;
  suggested: number;
  selfConfidence: number;
  meanLoss: number;
  lateLoss: number;
  forgetting: number;
  /** Rank-normalized suspicion from confident learning, in [0, 1]. */
  signalA: number;
  /** Rank-normalized suspicion from loss trajectories, in [0, 1]. */
  signalB: number;
  trajectory: number[];
  /** Whether the label was flipped on purpose; only known for the demo sample. */
  injected: boolean | null;
}

export interface AuditData {
  classes: string[];
  rows: AuditRow[];
  /** Default number of rows to flag: confident learning's estimate of label issues. */
  suggestedFlagCount: number;
  /** Share of signal A in the combined ranking, in [0, 1]. */
  defaultBlend: number;
  defaultClassCap: number | null;
  epochs: number;
  seeds: number;
  folds: number;
}

interface DemoFile {
  version: number;
  meta: {
    classes: string[];
    folds: number;
    trajectorySeeds: number;
    trajectoryEpochs: number;
    weightA: number;
    weightB: number;
    classCap: number | null;
    suggestedFlagCount: number;
    auditedExamples: number;
    encoder: string;
  };
  rows: {
    text: string[];
    given: number[];
    suggested: number[];
    selfConfidence: number[];
    meanLoss: number[];
    lateLoss: number[];
    forgetting: number[];
    signalA: number[];
    signalB: number[];
    trajectory: number[][];
    injected: number[];
  };
}

export interface DemoData extends AuditData {
  auditedExamples: number;
  encoder: string;
}

export function fromDemo(file: DemoFile): DemoData {
  if (file.version !== 1) throw new Error(`Unsupported demo data version ${file.version}`);
  const { meta, rows } = file;
  return {
    classes: meta.classes,
    rows: rows.text.map((text, i) => ({
      id: i,
      text,
      given: rows.given[i],
      suggested: rows.suggested[i],
      selfConfidence: rows.selfConfidence[i],
      meanLoss: rows.meanLoss[i],
      lateLoss: rows.lateLoss[i],
      forgetting: rows.forgetting[i],
      signalA: rows.signalA[i],
      signalB: rows.signalB[i],
      trajectory: rows.trajectory[i],
      injected: rows.injected[i] === 1,
    })),
    suggestedFlagCount: meta.suggestedFlagCount,
    defaultBlend: meta.weightA / (meta.weightA + meta.weightB),
    defaultClassCap: meta.classCap,
    epochs: meta.trajectoryEpochs,
    seeds: meta.trajectorySeeds,
    folds: meta.folds,
    auditedExamples: meta.auditedExamples,
    encoder: meta.encoder,
  };
}

export async function loadDemo(url: string): Promise<DemoData> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load the demo data (HTTP ${response.status}).`);
  return fromDemo((await response.json()) as DemoFile);
}

export function fromBrowserAudit(
  ids: number[],
  texts: string[],
  labels: number[],
  classes: string[],
  result: BrowserAuditResult,
  options: BrowserAuditOptions,
): AuditData {
  return {
    classes,
    rows: texts.map((text, i) => ({
      id: ids[i],
      text,
      given: labels[i],
      suggested: result.suggested[i],
      selfConfidence: result.selfConfidence[i],
      meanLoss: result.meanLoss[i],
      lateLoss: result.lateLoss[i],
      forgetting: result.forgetting[i],
      signalA: result.signalA[i],
      signalB: result.signalB[i],
      trajectory: result.trajectory[i],
      injected: null,
    })),
    suggestedFlagCount: result.estimatedIssues,
    defaultBlend: 0.5,
    defaultClassCap: 0.5,
    epochs: options.trajectoryEpochs,
    seeds: options.seeds,
    folds: options.folds,
  };
}
