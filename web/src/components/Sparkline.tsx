import { useState, type PointerEvent } from "react";

interface SparklineProps {
  values: number[];
  /** Shared upper bound so every row is drawn on the same scale; larger values clip. */
  max: number;
  /** Loss of a uniform guess, drawn as a hairline for reference. */
  chance: number;
}

const W = 132;
const H = 36;
const PAD = 5;

export function Sparkline({ values, max, chance }: SparklineProps) {
  const [hover, setHover] = useState<number | null>(null);
  const x = (i: number) => PAD + (i * (W - 2 * PAD)) / Math.max(1, values.length - 1);
  const y = (v: number) => PAD + (1 - Math.min(v, max) / max) * (H - 2 * PAD);
  const points = values.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`);
  const line = `M${points.join("L")}`;
  const area = `${line}L${x(values.length - 1).toFixed(1)},${H - PAD}L${PAD},${H - PAD}Z`;
  const last = values.length - 1;
  const shown = hover ?? last;

  const onMove = (event: PointerEvent<SVGSVGElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    const fraction = (event.clientX - box.left) / box.width;
    const i = Math.round(((fraction * W - PAD) / (W - 2 * PAD)) * last);
    setHover(Math.max(0, Math.min(last, i)));
  };

  const summary = `Training loss by epoch, from ${values[0].toFixed(2)} at epoch 1 to ${values[last].toFixed(2)} at epoch ${last + 1}`;

  return (
    <span className="spark">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        width={W}
        height={H}
        role="img"
        aria-label={summary}
        onPointerMove={onMove}
        onPointerLeave={() => setHover(null)}
      >
        <line className="spark-chance" x1={PAD} x2={W - PAD} y1={y(chance)} y2={y(chance)} />
        <path className="spark-area" d={area} />
        <path className="spark-line" d={line} />
        <circle className="spark-dot" cx={x(shown)} cy={y(values[shown])} r={4} />
      </svg>
      <span className="spark-readout" aria-hidden="true">
        {hover === null ? "" : `epoch ${hover + 1}: ${values[hover].toFixed(2)}`}
      </span>
    </span>
  );
}
