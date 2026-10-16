import { describe, expect, it } from "vitest";
import toy from "../../../tests/fixtures/toy.json";
import {
  calibrateJoint,
  combine,
  confidentLearning,
  flagRates,
  lateEpochCount,
  lossStatistics,
  rankNormalize,
  ranking,
  selectFlags,
  signalA,
  signalB,
} from "./scoring";

const close = (actual: number[], expected: number[]) => {
  expect(actual).toHaveLength(expected.length);
  actual.forEach((v, i) => expect(v).toBeCloseTo(expected[i], 9));
};

describe("confident learning on the shared toy example", () => {
  const cl = confidentLearning(toy.pred_probs, toy.labels, toy.num_classes);
  const expected = toy.confident_learning;

  it("computes per-class thresholds", () => close(cl.thresholds, expected.thresholds));

  it("assigns confident labels, taking the argmax among classes over threshold", () => {
    expect(cl.confidentLabel).toEqual(expected.confident_label);
    expect(cl.confidentLabel[12]).toBe(0);
    expect(cl.confidentLabel[10]).toBe(1);
  });

  it("builds the confident joint", () => expect(cl.joint).toEqual(expected.confident_joint));

  it("calibrates the joint to label counts", () => {
    cl.calibratedJoint.forEach((row, i) => close(row, expected.calibrated_joint[i]));
  });

  it("estimates issues, rounding half up", () => {
    close(cl.estimatedIssuesPerClass, expected.estimated_issues_per_class);
    expect(cl.estimatedIssues).toBe(expected.estimated_issues);
  });

  it("scores and suggests labels", () => {
    close(cl.selfConfidence, expected.self_confidence);
    close(cl.normalizedMargin, expected.normalized_margin);
    expect(cl.suggested).toEqual(expected.suggested_labels);
  });

  it("ranks by self-confidence", () => {
    expect(ranking(signalA(cl.selfConfidence))).toEqual(expected.self_confidence_order);
  });

  it("keeps an empty calibrated row on the diagonal", () => {
    expect(calibrateJoint([[2, 0], [0, 0]], [0, 0, 1])).toEqual([[2, 0], [0, 1]]);
  });
});

describe("ranking", () => {
  it("rank-normalizes with averaged ties", () => {
    close(rankNormalize(toy.rank_normalize.input), toy.rank_normalize.expected);
  });

  it("handles empty, single, and all-tied input", () => {
    expect(rankNormalize([])).toEqual([]);
    expect(rankNormalize([4.2])).toEqual([0]);
    expect(rankNormalize([7, 7, 7])).toEqual([0.5, 0.5, 0.5]);
  });

  it("combines the two signals into the same order as Python", () => {
    const r = toy.ranking;
    const a = signalA(toy.confident_learning.self_confidence);
    const b = rankNormalize(r.mean_loss);
    close(a, r.signal_a_twelfths.map((v) => v / 12));
    close(b, r.loss_only_twelfths.map((v) => v / 12));
    const combined = combine(a, b, r.weight_a, r.weight_b);
    close(combined, r.combined_twelfths.map((v) => v / 12));
    expect(ranking(combined)).toEqual(r.combined_order);
  });

  it("rejects all-zero weights", () => {
    expect(() => combine([0], [0], 0, 0)).toThrow();
  });
});

describe("loss trajectories", () => {
  const t = toy.trajectories;
  const stats = lossStatistics(t.losses, t.correct, t.late_fraction);

  it("computes mean loss, late loss, forgetting, and the trajectory", () => {
    close(stats.meanLoss, t.mean_loss);
    close(stats.lateLoss, t.late_loss);
    close(stats.forgetting, t.forgetting);
    stats.trajectory.forEach((row, i) => close(row, t.trajectory[i]));
  });

  it("combines them into signal B", () => close(signalB(stats), t.signal_b));

  it("counts late epochs the same way as Python", () => {
    expect(lateEpochCount(15, 1 / 3)).toBe(5);
    expect(lateEpochCount(4, 0.5)).toBe(2);
    expect(lateEpochCount(3, 0.01)).toBe(1);
  });
});

describe("cleaning", () => {
  it.each(toy.cleaning.map((c, i) => [i, c] as const))("selects flags for case %i", (_, c) => {
    const flagged = selectFlags(toy.ranking.combined_order, toy.labels, toy.num_classes, c.n_flag, c.class_cap);
    expect(flagged).toEqual(c.flagged);
  });

  it("reports per-class flag rates", () => {
    const rates = flagRates(toy.labels, toy.cleaning[0].flagged, toy.num_classes);
    expect(rates).toEqual(toy.flag_rates_for_first_cleaning_case);
  });
});
