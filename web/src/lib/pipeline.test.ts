import { describe, expect, it } from "vitest";
import { lossAndCorrect, SoftmaxRegression } from "./logreg";
import { mulberry32 } from "./rng";
import { DEFAULT_OPTIONS, runBrowserAudit, stratifiedFolds } from "./pipeline";
import { ranking, combine } from "./scoring";
import { tfidf, tokenize } from "./tfidf";

const TOPICS = [
  ["goal", "match", "league", "coach", "season", "striker", "referee"],
  ["market", "shares", "profit", "investor", "earnings", "bank", "stocks"],
  ["software", "chip", "browser", "laptop", "server", "code", "network"],
];

/** A corpus where each topic's words identify its class. */
function corpus(perClass: number, seed: number): { texts: string[]; labels: number[] } {
  const random = mulberry32(seed);
  const texts: string[] = [];
  const labels: number[] = [];
  for (let i = 0; i < perClass; i++) {
    TOPICS.forEach((words, c) => {
      const picked = Array.from({ length: 6 }, () => words[Math.floor(random() * words.length)]);
      texts.push(picked.join(" "));
      labels.push(c);
    });
  }
  return { texts, labels };
}

describe("tfidf", () => {
  it("tokenizes words and numbers, dropping single letters", () => {
    expect(tokenize("Apple's iPod, 2 a B-side!")).toEqual(["apple", "ipod", "2", "side"]);
  });

  it("produces unit-length rows and down-weights common terms", () => {
    const { matrix, vocabulary } = tfidf(["red apple", "red pear", "red fig", "green apple"], {
      maxFeatures: 100,
      minDf: 1,
    });
    for (let i = 0; i < matrix.rows; i++) {
      let sum = 0;
      for (let p = matrix.indptr[i]; p < matrix.indptr[i + 1]; p++) sum += matrix.values[p] ** 2;
      expect(sum).toBeCloseTo(1, 12);
    }
    const weight = (row: number, term: string) => {
      const j = vocabulary.indexOf(term);
      for (let p = matrix.indptr[row]; p < matrix.indptr[row + 1]; p++) {
        if (matrix.indices[p] === j) return matrix.values[p];
      }
      return 0;
    };
    expect(weight(1, "pear")).toBeGreaterThan(weight(1, "red"));
  });

  it("caps the vocabulary by document frequency, breaking ties alphabetically", () => {
    // df: a1 = 3; b1 = 2 and the bigram "a1 b1" = 2; c1 and "a1 c1" = 1.
    const { vocabulary } = tfidf(["a1 b1", "a1 c1", "a1 b1"], { maxFeatures: 3, minDf: 1 });
    expect(vocabulary).toEqual(["a1", "a1 b1", "b1"]);
  });

  it("drops terms below the minimum document frequency", () => {
    const { vocabulary } = tfidf(["a1 b1", "a1 c1"], { maxFeatures: 10, minDf: 2 });
    expect(vocabulary).toEqual(["a1"]);
  });
});

describe("SoftmaxRegression", () => {
  it("fits separable data and returns probabilities", () => {
    const { texts, labels } = corpus(10, 1);
    const { matrix } = tfidf(texts, { maxFeatures: 1000, minDf: 1 });
    const model = new SoftmaxRegression(matrix.cols, 3).fit(
      matrix,
      labels.map((_, i) => i),
      labels,
      { epochs: 5, learningRate: 0.5, l2: 1e-4, seed: 0 },
    );
    const probs = model.predictRow(matrix, 0, new Float64Array(3));
    expect(probs.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    expect(Array.from(lossAndCorrect(model, matrix, labels).correct).every(Boolean)).toBe(true);
  });
});

describe("runBrowserAudit", () => {
  const options = { ...DEFAULT_OPTIONS, folds: 3, cvEpochs: 4, trajectoryEpochs: 5, seeds: 2 };

  it("spreads every class across folds", () => {
    const labels = [0, 0, 0, 0, 0, 0, 1, 1, 1];
    const folds = stratifiedFolds(labels, 3, 0);
    for (const c of [0, 1]) {
      const counts = [0, 0, 0];
      labels.forEach((y, i) => y === c && counts[folds[i]]++);
      expect(Math.max(...counts) - Math.min(...counts)).toBeLessThanOrEqual(1);
    }
  });

  it("is deterministic and returns one value per example", () => {
    const { texts, labels } = corpus(12, 2);
    const a = runBrowserAudit(texts, labels, 3, options);
    const b = runBrowserAudit(texts, labels, 3, options);
    expect(a).toEqual(b);
    expect(a.signalA).toHaveLength(texts.length);
    expect(a.trajectory[0]).toHaveLength(options.trajectoryEpochs);
  });

  it("puts a planted wrong label at the top of the combined ranking", () => {
    const { texts, labels } = corpus(12, 3);
    const noisy = labels.slice();
    noisy[4] = (labels[4] + 1) % 3;
    const result = runBrowserAudit(texts, noisy, 3, options);
    expect(ranking(combine(result.signalA, result.signalB, 0.5, 0.5))[0]).toBe(4);
    expect(result.suggested[4]).toBe(labels[4]);
  });

  it("reports progress up to the total number of epochs", () => {
    const { texts, labels } = corpus(6, 4);
    const seen: number[] = [];
    runBrowserAudit(texts, labels, 3, options, (_, done, total) => {
      seen.push(done);
      expect(done).toBeLessThanOrEqual(total);
    });
    expect(seen.at(-1)).toBe(options.folds * options.cvEpochs + options.seeds * options.trajectoryEpochs);
  });
});
