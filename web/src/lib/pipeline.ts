// The in-browser audit: TF-IDF features, K-fold cross-validated probabilities for
// confident learning, and per-epoch loss trajectories over several seeds, blended
// with the same scoring code the Python pipeline uses.

import { lossAndCorrect, SoftmaxRegression } from "./logreg";
import { mulberry32, shuffle } from "./rng";
import { confidentLearning, lossStatistics, signalA, signalB } from "./scoring";
import { tfidf } from "./tfidf";

export interface BrowserAuditOptions {
  folds: number;
  cvEpochs: number;
  trajectoryEpochs: number;
  seeds: number;
  lateFraction: number;
  learningRate: number;
  l2: number;
  maxFeatures: number;
  minDf: number;
  seed: number;
}

export const DEFAULT_OPTIONS: BrowserAuditOptions = {
  folds: 5,
  cvEpochs: 8,
  trajectoryEpochs: 12,
  seeds: 3,
  lateFraction: 1 / 3,
  learningRate: 0.5,
  l2: 1e-4,
  maxFeatures: 30_000,
  minDf: 2,
  seed: 0,
};

export interface BrowserAuditResult {
  suggested: number[];
  selfConfidence: number[];
  normalizedMargin: number[];
  meanLoss: number[];
  lateLoss: number[];
  forgetting: number[];
  trajectory: number[][];
  signalA: number[];
  signalB: number[];
  estimatedIssues: number;
}

export type ProgressHandler = (stage: string, done: number, total: number) => void;

/** Fold id per example; each class is spread evenly across folds in a seeded order. */
export function stratifiedFolds(labels: number[], folds: number, seed: number): number[] {
  const random = mulberry32(seed);
  const assignment = new Array<number>(labels.length).fill(0);
  const byClass = new Map<number, number[]>();
  labels.forEach((y, i) => {
    const members = byClass.get(y);
    if (members) members.push(i);
    else byClass.set(y, [i]);
  });
  let offset = 0;
  for (const y of [...byClass.keys()].sort((a, b) => a - b)) {
    const members = shuffle(byClass.get(y)!, random);
    members.forEach((i, r) => (assignment[i] = (r + offset) % folds));
    offset += members.length;
  }
  return assignment;
}

export function runBrowserAudit(
  texts: string[],
  labels: number[],
  numClasses: number,
  options: BrowserAuditOptions = DEFAULT_OPTIONS,
  onProgress: ProgressHandler = () => {},
): BrowserAuditResult {
  const n = texts.length;
  const totalEpochs = options.folds * options.cvEpochs + options.seeds * options.trajectoryEpochs;
  let done = 0;

  onProgress("Building TF-IDF features", 0, totalEpochs);
  const { matrix } = tfidf(texts, { maxFeatures: options.maxFeatures, minDf: options.minDf });
  const sgd = (seed: number, epochs: number) => ({
    epochs,
    learningRate: options.learningRate,
    l2: options.l2,
    seed,
  });

  // Signal A: every example is scored by a model that never trained on it.
  const fold = stratifiedFolds(labels, options.folds, options.seed);
  const probs: number[][] = Array.from({ length: n }, () => []);
  const out = new Float64Array(numClasses);
  for (let f = 0; f < options.folds; f++) {
    const train: number[] = [];
    const held: number[] = [];
    fold.forEach((g, i) => (g === f ? held : train).push(i));
    const model = new SoftmaxRegression(matrix.cols, numClasses);
    model.fit(matrix, train, labels, sgd(options.seed + f, options.cvEpochs), () =>
      onProgress(`Cross-validation: fold ${f + 1} of ${options.folds}`, ++done, totalEpochs),
    );
    for (const i of held) probs[i] = Array.from(model.predictRow(matrix, i, out));
  }
  const cl = confidentLearning(probs, labels, numClasses);

  // Signal B: train on everything and record each example's loss after every epoch.
  const losses: Float64Array[][] = [];
  const correct: Uint8Array[][] = [];
  for (let s = 0; s < options.seeds; s++) {
    const seedLosses: Float64Array[] = [];
    const seedCorrect: Uint8Array[] = [];
    const model = new SoftmaxRegression(matrix.cols, numClasses);
    model.fit(matrix, Array.from({ length: n }, (_, i) => i), labels, sgd(options.seed + 10_000 + s, options.trajectoryEpochs), () => {
      const epoch = lossAndCorrect(model, matrix, labels);
      seedLosses.push(epoch.loss);
      seedCorrect.push(epoch.correct);
      onProgress(`Loss trajectories: seed ${s + 1} of ${options.seeds}`, ++done, totalEpochs);
    });
    losses.push(seedLosses);
    correct.push(seedCorrect);
  }
  const stats = lossStatistics(losses, correct, options.lateFraction);

  return {
    suggested: cl.suggested,
    selfConfidence: cl.selfConfidence,
    normalizedMargin: cl.normalizedMargin,
    meanLoss: stats.meanLoss,
    lateLoss: stats.lateLoss,
    forgetting: stats.forgetting,
    trajectory: stats.trajectory,
    signalA: signalA(cl.selfConfidence),
    signalB: signalB(stats),
    estimatedIssues: cl.estimatedIssues,
  };
}
