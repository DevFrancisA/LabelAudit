# LabelAudit

Find training examples whose labels are probably wrong.

**Live demo: https://DevFrancisA.github.io/LabelAudit/**

LabelAudit ranks every example in a labeled text dataset by how likely its label is a mistake, shows the evidence for each one, and lets you remove the flagged examples and export a cleaned CSV. It comes as two pieces that share the same scoring math:

- a **Python pipeline** (PyTorch + scikit-learn) that audits a public dataset end to end, and
- a **static web app** that shows that audit, and can also audit your own CSV entirely in your browser.

## The problem

Real labeled datasets contain mistakes: a sports story filed under Business, a review tagged positive that is plainly negative. A model trained on that data learns the mistakes too, and evaluation on mislabeled test data gives misleading scores. Reading every example by hand doesn't scale, so the useful question is *which* examples to read first.

LabelAudit answers that by asking two independent questions about every example and ranking the ones where the answers look worst. Its output is a list of leads, not verdicts. Some flagged examples are genuinely ambiguous rather than wrong, so a person should read them before deleting anything.

## How it works

### Signal A: confident learning

Confident learning (Northcutt, Jiang & Chuang, 2021) looks for examples where a model, scoring data it was **not** trained on, disagrees with the given label.

1. Split the data into K folds. Train on K−1 folds and predict class probabilities for the held-out fold, so every example gets an out-of-sample prediction.
2. For each class *j*, compute a **confidence threshold** *t<sub>j</sub>*: the average predicted probability of class *j* over the examples labeled *j*. It is how confident the model typically is when that class is correct.
3. Build the **confident joint**: an example with given label *i* is counted in cell (*i*, *j*) when *j* is the most probable class among those whose probability clears their own threshold. Off-diagonal cells are likely label errors, and *j* becomes the example's **suggested label**.
4. **Calibrate** the joint by rescaling each row to the number of examples with that given label. Its off-diagonal total estimates how many labels are wrong, and that estimate sets the default number of examples to flag.
5. Rank examples by **self-confidence**, the out-of-sample probability of the given label (lower is more suspicious). The **normalized margin** variant, p(given) − max p(other), is also implemented and can be selected in the config.

This repository implements the method directly in `src/labelaudit/confident.py` and `web/src/lib/scoring.ts`. It does not use the `cleanlab` package.

### Signal B: loss trajectories

Correct labels agree with the rest of the data, so a model learns them early and keeps them. Wrong labels fight the rest of the data: their loss stays high, falls only late in training once the model starts memorizing, and the model often flips back and forth on them.

LabelAudit trains on all examples several times with different seeds, records every example's loss after every epoch, and computes per example:

- **mean loss** over all seeds and epochs;
- **late-training loss**, the mean over the last third of epochs;
- a **forgetting** statistic in the spirit of Toneva et al. (2019): how often the model went from predicting the given label to not predicting it between consecutive epochs. An example that is never predicted correctly gets the maximum score.

The three are rank-normalized and averaged into one suspicion score.

### Combining them

Each signal is converted to ranks scaled to [0, 1], where ties share their average rank, so that probabilities, losses, and counts can be blended without one dominating through its scale. The **combined ranking** is a weighted average of the two (weights are configurable). A **loss-only ranking** (mean training loss alone) is exposed as a baseline.

### Cleaning

Cleaning removes the top of the ranking. By default it removes as many examples as confident learning estimates are wrong, or you can set a fixed fraction. An optional **per-class cap** limits how much of any one class can be removed, so a rare class isn't stripped out just because it is hard. The audit prints and saves a per-class flag-rate table.

## Python version vs. browser version

|                          | Python pipeline                                               | Browser ("Audit your own data")                          |
| ------------------------ | ------------------------------------------------------------- | -------------------------------------------------------- |
| Features                 | Frozen `all-MiniLM-L6-v2` sentence embeddings                  | TF-IDF over word unigrams and bigrams                    |
| Model                    | Small MLP head in PyTorch (AdamW, dropout)                     | Multinomial logistic regression trained with SGD         |
| Signal A                 | K-fold out-of-sample probabilities → confident learning        | Same                                                     |
| Signal B                 | Per-epoch losses over several seeds                            | Same                                                     |
| Ranking, flags, cap      | `ranking.py`, `cleaning.py`                                    | `scoring.ts`: same formulas, tested on the same fixture  |
| Where it runs            | Your machine (CPU is enough)                                   | A Web Worker in your browser tab; nothing is uploaded    |

Sentence embeddings capture meaning ("stocks slid" ≈ "shares fell"), while TF-IDF only sees which words appear, so the browser version works best when the classes differ in vocabulary. All of the TypeScript math is hand-written with no ML library. Both implementations are tested against one hand-computed example (`tests/fixtures/toy.json`), which pins thresholds, the confident joint, rank normalization, the combined ranking, and the per-class cap, so they must agree.

## Run it locally

### Python pipeline

Requires [uv](https://docs.astral.sh/uv/). The project pins CPU builds of PyTorch, so no GPU is needed.

```sh
uv sync
uv run labelaudit prepare --config configs/agnews-demo.toml   # download AG News, take a subset, inject noise
uv run labelaudit audit                                       # both signals, ranking, cleaned CSV
uv run labelaudit evaluate                                    # compare rankings against the injected ground truth
uv run labelaudit retrain                                     # corrupted vs. cleaned training, scored on the clean held-out split
uv run labelaudit export-site                                 # write web/public/demo.json
```

Every command takes `--run-dir` (default `runs/agnews`). `prepare` saves the resolved config into the run directory and every later stage reads it from there, so a run is self-describing. `evaluate` and `retrain` print to the terminal and write nothing.

A run directory looks like this:

```
runs/agnews/
  config.json          resolved config for this run
  data/                what the detector sees: train.jsonl (noisy labels), test.jsonl, classes.json
  truth/injected.json  which training labels were flipped, read only by evaluate and export-site
  audit/               scores.npz, flags.csv, flag_rates.csv, cleaned.csv
```

Embeddings are cached under `.cache/labelaudit/` keyed by a hash of the encoder name and the texts.

### Configuration

Without `--config`, the defaults in `src/labelaudit/config.py` apply. A TOML file overrides any subset:

```toml
seed = 0

[data]
n_train = 20000
n_test = 4000
class_proportions = [0.4, 0.3, 0.2, 0.1]   # omit for equal counts per class

[noise]
mode = "class_dependent"                    # or "uniform"
rate = 0.1
flip_targets = [[0, 0.2, 0.5, 0.3], [0.6, 0, 0.2, 0.2], [0.4, 0.1, 0, 0.5], [0.2, 0.1, 0.7, 0]]

[audit]
folds = 5
trajectory_seeds = 3
trajectory_epochs = 15
cl_score = "self_confidence"                # or "normalized_margin"
weight_a = 0.5
weight_b = 0.5
flag_fraction = 0.08                        # omit to use the confident-learning estimate
class_cap = 0.5                             # remove at most this share of any class; 1.0 disables the cap
```

With the same seed, machine, and library versions, every stage produces identical output: data sampling, noise, fold splits, weight initialization, and minibatch order are all seeded, and PyTorch runs with deterministic algorithms. Different CPUs or library versions can change floating-point results slightly.

### Web app

```sh
cd web
npm ci
npm run dev        # http://localhost:5173/LabelAudit/
npm test           # vitest
npm run typecheck
npm run build      # static site in web/dist
```

### Reproducing the demo data

The sample on the site's first tab was produced by a real run of the pipeline:

```sh
uv run labelaudit prepare --config configs/agnews-demo.toml
uv run labelaudit audit
uv run labelaudit export-site
```

`export-site` writes a seeded random sample of the audited training set to `web/public/demo.json`. For each example it includes the text, the given and suggested labels, both signals, the per-epoch loss trajectory, and whether its label was injected noise. Nothing on the site is hand-written or adjusted.

## Deployment

On every push to `main`, GitHub Actions runs the Python tests and lint, the web tests, the type-check, and the production build, then deploys `web/dist` to GitHub Pages (Settings → Pages → Source: GitHub Actions).

## Limitations

- **Hard is not the same as wrong.** Both signals flag examples a model finds difficult. Ambiguous, multi-topic, or unusual examples can be flagged even when their label is right, and a mislabeled example that happens to look like its wrong class can slip through.
- **Systematic errors are hard to see.** If a whole region of the data is mislabeled the same way, the model learns that pattern and both signals treat it as normal.
- **The estimate depends on calibration.** Confident learning's count of label issues assumes the model's probabilities are roughly calibrated. An underfit or overconfident model shifts the estimate, and with it the default flag count.
- **Rare classes look suspicious.** Classes with few examples are harder to learn, so their members rank higher. The per-class cap limits the damage but doesn't fix the cause.
- **The demo noise is synthetic.** The AG News sample has labels flipped on purpose (class-dependent, following plausible confusions), and real annotation errors can be subtler. AG News itself also contains some original label noise, which the pipeline may flag but the injected ground truth doesn't count.
- **The browser version is weaker by design.** TF-IDF with a linear model misses paraphrase and context that sentence embeddings capture. It handles up to 50,000 rows and 50 classes, single-label text only.
- **English-first.** The demo encoder is an English model. The browser tokenizer handles any Unicode script, but has no language-specific processing.

## References

- Curtis G. Northcutt, Lu Jiang, Isaac L. Chuang. *Confident Learning: Estimating Uncertainty in Dataset Labels.* Journal of Artificial Intelligence Research 70 (2021): 1373–1411. [arXiv:1911.00068](https://arxiv.org/abs/1911.00068)
- Mariya Toneva, Alessandro Sordoni, Remi Tachet des Combes, Adam Trischler, Yoshua Bengio, Geoffrey J. Gordon. *An Empirical Study of Example Forgetting during Deep Neural Network Learning.* ICLR 2019. [arXiv:1812.05159](https://arxiv.org/abs/1812.05159)
- Xiang Zhang, Junbo Zhao, Yann LeCun. *Character-level Convolutional Networks for Text Classification.* NeurIPS 2015 (the AG News benchmark).
- Nils Reimers, Iryna Gurevych. *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP 2019 (the sentence-transformers library behind `all-MiniLM-L6-v2`).

## License

MIT. See [LICENSE](LICENSE).
