// Multinomial logistic regression on sparse rows, trained with plain SGD.
// L2 regularization uses the scaled-weights trick (true weights = scale * W) so a
// decay step costs O(1) instead of touching every weight.

import { mulberry32, shuffle } from "./rng";
import type { SparseMatrix } from "./tfidf";

export interface SgdOptions {
  epochs: number;
  learningRate: number;
  l2: number;
  seed: number;
}

export class SoftmaxRegression {
  readonly weights: Float64Array; // [feature * numClasses + class]
  readonly bias: Float64Array;
  private scale = 1;

  constructor(
    readonly numFeatures: number,
    readonly numClasses: number,
  ) {
    this.weights = new Float64Array(numFeatures * numClasses);
    this.bias = new Float64Array(numClasses);
  }

  /** Writes class probabilities for row `i` of `x` into `out`. */
  predictRow(x: SparseMatrix, i: number, out: Float64Array): Float64Array {
    const k = this.numClasses;
    out.set(this.bias);
    for (let p = x.indptr[i]; p < x.indptr[i + 1]; p++) {
      const v = x.values[p] * this.scale;
      const base = x.indices[p] * k;
      for (let c = 0; c < k; c++) out[c] += this.weights[base + c] * v;
    }
    let max = -Infinity;
    for (let c = 0; c < k; c++) if (out[c] > max) max = out[c];
    let sum = 0;
    for (let c = 0; c < k; c++) {
      out[c] = Math.exp(out[c] - max);
      sum += out[c];
    }
    for (let c = 0; c < k; c++) out[c] /= sum;
    return out;
  }

  /**
   * Train on the given rows. `afterEpoch` runs after every epoch, e.g. to record
   * per-example losses.
   */
  fit(
    x: SparseMatrix,
    rows: number[],
    labels: ArrayLike<number>,
    options: SgdOptions,
    afterEpoch?: (epoch: number) => void,
  ): this {
    const k = this.numClasses;
    const random = mulberry32(options.seed);
    const order = rows.slice();
    const probs = new Float64Array(k);
    let step = 0;
    for (let epoch = 0; epoch < options.epochs; epoch++) {
      shuffle(order, random);
      for (const i of order) {
        const eta = options.learningRate / (1 + options.learningRate * options.l2 * step++);
        this.predictRow(x, i, probs);
        probs[labels[i]] -= 1; // gradient of cross-entropy w.r.t. the logits
        this.scale *= 1 - eta * options.l2;
        const factor = eta / this.scale;
        for (let p = x.indptr[i]; p < x.indptr[i + 1]; p++) {
          const v = x.values[p] * factor;
          const base = x.indices[p] * k;
          for (let c = 0; c < k; c++) this.weights[base + c] -= probs[c] * v;
        }
        for (let c = 0; c < k; c++) this.bias[c] -= eta * probs[c];
        if (this.scale < 1e-9) this.rescale();
      }
      afterEpoch?.(epoch);
    }
    return this;
  }

  private rescale(): void {
    for (let j = 0; j < this.weights.length; j++) this.weights[j] *= this.scale;
    this.scale = 1;
  }
}

const MIN_PROB = 1e-12;

/** Cross-entropy under the given label, and whether the top class equals it. */
export function lossAndCorrect(
  model: SoftmaxRegression,
  x: SparseMatrix,
  labels: ArrayLike<number>,
): { loss: Float64Array; correct: Uint8Array } {
  const n = x.rows;
  const loss = new Float64Array(n);
  const correct = new Uint8Array(n);
  const probs = new Float64Array(model.numClasses);
  for (let i = 0; i < n; i++) {
    model.predictRow(x, i, probs);
    const y = labels[i];
    loss[i] = -Math.log(Math.max(probs[y], MIN_PROB));
    let top = 0;
    for (let c = 1; c < probs.length; c++) if (probs[c] > probs[top]) top = c;
    correct[i] = top === y ? 1 : 0;
  }
  return { loss, correct };
}
