// Scoring math shared by both tabs. It mirrors the Python package
// (labelaudit.confident, loss_signals, ranking, cleaning), and both are tested
// against the same hand-computed fixture in tests/fixtures/toy.json.

/** Round half up, matching the Python side's math.floor(x + 0.5). */
export const roundHalfUp = (x: number): number => Math.floor(x + 0.5);

// ---------- Signal A: confident learning (Northcutt et al., 2021) ----------

export interface ConfidentLearning {
  thresholds: number[];
  /** Most probable class among those clearing their threshold; -1 if none does. */
  confidentLabel: number[];
  /** Rows are given labels, columns are confident labels. */
  joint: number[][];
  calibratedJoint: number[][];
  suggested: number[];
  selfConfidence: number[];
  normalizedMargin: number[];
  estimatedIssuesPerClass: number[];
  estimatedIssues: number;
}

export function perClassThresholds(probs: number[][], labels: number[], numClasses: number): number[] {
  const sums = new Array<number>(numClasses).fill(0);
  const counts = new Array<number>(numClasses).fill(0);
  labels.forEach((y, i) => {
    sums[y] += probs[i][y];
    counts[y] += 1;
  });
  return sums.map((s, j) => {
    if (counts[j] === 0) throw new Error(`class ${j} has no labeled examples`);
    return s / counts[j];
  });
}

export function confidentLabels(probs: number[][], thresholds: number[]): number[] {
  return probs.map((row) => {
    let best = -1;
    for (let j = 0; j < row.length; j++) {
      if (row[j] >= thresholds[j] && (best < 0 || row[j] > row[best])) best = j;
    }
    return best;
  });
}

export function confidentJoint(labels: number[], confident: number[], numClasses: number): number[][] {
  const joint = Array.from({ length: numClasses }, () => new Array<number>(numClasses).fill(0));
  labels.forEach((y, i) => {
    if (confident[i] >= 0) joint[y][confident[i]] += 1;
  });
  return joint;
}

/** Rescale row i so it sums to the number of examples labeled i. */
export function calibrateJoint(joint: number[][], labels: number[]): number[][] {
  const k = joint.length;
  const counts = new Array<number>(k).fill(0);
  for (const y of labels) counts[y] += 1;
  return joint.map((row, i) => {
    const total = row.reduce((a, b) => a + b, 0);
    if (total === 0) return row.map((_, j) => (j === i ? counts[i] : 0));
    return row.map((v) => (v / total) * counts[i]);
  });
}

export function selfConfidence(probs: number[][], labels: number[]): number[] {
  return labels.map((y, i) => probs[i][y]);
}

export function normalizedMargin(probs: number[][], labels: number[]): number[] {
  return labels.map((y, i) => {
    let other = -Infinity;
    probs[i].forEach((p, j) => {
      if (j !== y && p > other) other = p;
    });
    return probs[i][y] - other;
  });
}

function argmax(row: number[]): number {
  let best = 0;
  for (let j = 1; j < row.length; j++) if (row[j] > row[best]) best = j;
  return best;
}

export function confidentLearning(probs: number[][], labels: number[], numClasses: number): ConfidentLearning {
  const thresholds = perClassThresholds(probs, labels, numClasses);
  const confident = confidentLabels(probs, thresholds);
  const joint = confidentJoint(labels, confident, numClasses);
  const calibrated = calibrateJoint(joint, labels);
  const perClass = calibrated.map((row, i) => row.reduce((a, b) => a + b, 0) - row[i]);
  return {
    thresholds,
    confidentLabel: confident,
    joint,
    calibratedJoint: calibrated,
    suggested: confident.map((c, i) => (c >= 0 ? c : argmax(probs[i]))),
    selfConfidence: selfConfidence(probs, labels),
    normalizedMargin: normalizedMargin(probs, labels),
    estimatedIssuesPerClass: perClass,
    estimatedIssues: roundHalfUp(perClass.reduce((a, b) => a + b, 0)),
  };
}

// ---------- Signal B: loss-trajectory statistics ----------

export interface LossStats {
  meanLoss: number[];
  lateLoss: number[];
  forgetting: number[];
  /** Per example, loss at each epoch averaged over seeds. */
  trajectory: number[][];
}

export function lateEpochCount(epochs: number, lateFraction: number): number {
  return Math.max(1, Math.min(epochs, roundHalfUp(epochs * lateFraction)));
}

/** losses[seed][epoch][example]; correct[seed][epoch][example] says whether the top class was the given label. */
export function lossStatistics(
  losses: ArrayLike<number>[][],
  correct: ArrayLike<number | boolean>[][],
  lateFraction: number,
): LossStats {
  const seeds = losses.length;
  const epochs = losses[0].length;
  const n = losses[0][0].length;
  const late = lateEpochCount(epochs, lateFraction);
  const meanLoss = new Array<number>(n).fill(0);
  const lateLoss = new Array<number>(n).fill(0);
  const forgetting = new Array<number>(n).fill(0);
  const trajectory = Array.from({ length: n }, () => new Array<number>(epochs).fill(0));

  for (let s = 0; s < seeds; s++) {
    for (let i = 0; i < n; i++) {
      let events = 0;
      let everCorrect = false;
      for (let e = 0; e < epochs; e++) {
        const loss = losses[s][e][i];
        meanLoss[i] += loss;
        if (e >= epochs - late) lateLoss[i] += loss;
        trajectory[i][e] += loss / seeds;
        const now = Boolean(correct[s][e][i]);
        if (now) everCorrect = true;
        if (e > 0 && Boolean(correct[s][e - 1][i]) && !now) events += 1;
      }
      // Never learned in this seed: count as more forgetting than any learnable example can reach.
      forgetting[i] += (everCorrect ? events : epochs) / seeds;
    }
  }
  return {
    meanLoss: meanLoss.map((v) => v / (seeds * epochs)),
    lateLoss: lateLoss.map((v) => v / (seeds * late)),
    forgetting,
    trajectory,
  };
}

// ---------- Ranking ----------

/** Ranks scaled to [0, 1] with ties sharing their average rank; the smallest value maps to 0. */
export function rankNormalize(values: ArrayLike<number>): number[] {
  const n = values.length;
  const out = new Array<number>(n).fill(0);
  if (n <= 1) return out;
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => values[a] - values[b] || a - b);
  let start = 0;
  while (start < n) {
    let end = start + 1;
    while (end < n && values[order[end]] === values[order[start]]) end++;
    const average = (start + end - 1) / 2;
    for (let r = start; r < end; r++) out[order[r]] = average / (n - 1);
    start = end;
  }
  return out;
}

/** Suspicion from a confident-learning score where low values are suspicious. */
export const signalA = (confidence: ArrayLike<number>): number[] =>
  rankNormalize(Array.from(confidence, (v) => -v));

/** Suspicion from loss statistics: high mean loss, late loss and forgetting are suspicious. */
export function signalB(stats: Pick<LossStats, "meanLoss" | "lateLoss" | "forgetting">): number[] {
  const parts = [stats.meanLoss, stats.lateLoss, stats.forgetting].map(rankNormalize);
  return rankNormalize(parts[0].map((_, i) => (parts[0][i] + parts[1][i] + parts[2][i]) / 3));
}

export function combine(a: ArrayLike<number>, b: ArrayLike<number>, weightA: number, weightB: number): number[] {
  if (weightA < 0 || weightB < 0 || weightA + weightB === 0) {
    throw new Error("weights must be non-negative and not both zero");
  }
  return Array.from(a, (v, i) => (weightA * v + weightB * b[i]) / (weightA + weightB));
}

/** Indices from most to least suspicious; ties keep the lower index first. */
export function ranking(suspicion: ArrayLike<number>): number[] {
  return Array.from({ length: suspicion.length }, (_, i) => i).sort(
    (a, b) => suspicion[b] - suspicion[a] || a - b,
  );
}

// ---------- Cleaning ----------

/**
 * Walk the ranking and flag up to nFlag examples. With a cap, at most
 * floor(classCap * n_c) examples of class c are flagged; full classes are skipped.
 */
export function selectFlags(
  order: number[],
  labels: number[],
  numClasses: number,
  nFlag: number,
  classCap: number | null,
): number[] {
  const counts = new Array<number>(numClasses).fill(0);
  for (const y of labels) counts[y] += 1;
  const limits = classCap === null ? counts : counts.map((c) => Math.floor(classCap * c));
  const used = new Array<number>(numClasses).fill(0);
  const flagged: number[] = [];
  for (const i of order) {
    if (flagged.length >= nFlag) break;
    const c = labels[i];
    if (used[c] < limits[c]) {
      used[c] += 1;
      flagged.push(i);
    }
  }
  return flagged;
}

export interface ClassFlagRate {
  label: number;
  total: number;
  flagged: number;
}

export function flagRates(labels: number[], flagged: Iterable<number>, numClasses: number): ClassFlagRate[] {
  const rates = Array.from({ length: numClasses }, (_, label) => ({ label, total: 0, flagged: 0 }));
  for (const y of labels) rates[y].total += 1;
  for (const i of flagged) rates[labels[i]].flagged += 1;
  return rates;
}
