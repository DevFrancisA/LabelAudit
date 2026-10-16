export function Explainer() {
  return (
    <section className="explainer" aria-labelledby="how-heading">
      <h2 id="how-heading">How it decides what looks wrong</h2>
      <p className="explainer-lede">
        Labeled datasets always contain some mistakes, and a model trained on them learns the mistakes too. LabelAudit
        asks two independent questions about every example and ranks the ones where the answers look worst. They are
        leads to check, not verdicts: always read the example.
      </p>
      <div className="explainer-grid">
        <article className="signal-card signal-card-a">
          <h3>
            <span className="key-swatch" aria-hidden="true" /> Signal A · Confident learning
          </h3>
          <p>
            Split the data into folds. Train on all but one fold and score the held-out one, so the model never grades
            examples it studied. For each class, note how confident the model usually is when that class is correct;
            that is the class's threshold.
          </p>
          <p>
            An example is suspicious when the model gives its label a low probability, and especially when a different
            class clears its own threshold. That other class becomes the <em>suggested</em> label. Counting these
            disagreements across the dataset gives an estimate of how many labels are wrong, which sets the default flag
            count.
          </p>
        </article>
        <article className="signal-card signal-card-b">
          <h3>
            <span className="key-swatch" aria-hidden="true" /> Signal B · Loss trajectory
          </h3>
          <p>
            Train on everything, several times with different seeds, and record each example's loss after every epoch.
            Correct labels are learned early and stay learned. Wrong labels fight the rest of the data: their loss stays
            high, drops only late when the model starts memorizing, and the model often flips back and forth on them.
          </p>
          <p>
            Each row shows that curve. The faint line marks the loss of a blind guess. The signal combines mean loss,
            late-training loss, and a forgetting count (how often the model stopped predicting the label after getting
            it right; examples it never got right score highest).
          </p>
        </article>
      </div>
      <p className="explainer-foot">
        Both signals are turned into ranks between 0 (least suspicious) and 1 (most suspicious) and blended, so neither
        dominates because of its scale. Method after Northcutt, Jiang &amp; Chuang,{" "}
        <cite>Confident Learning: Estimating Uncertainty in Dataset Labels</cite> (JAIR, 2021).
      </p>
    </section>
  );
}
