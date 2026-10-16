import type { ClassFlagRate } from "../lib/scoring";

interface FlagRateTableProps {
  rates: ClassFlagRate[];
  classes: string[];
  capCount: (total: number) => number | null;
}

export function FlagRateTable({ rates, classes, capCount }: FlagRateTableProps) {
  const rateOf = (r: ClassFlagRate) => (r.total ? r.flagged / r.total : 0);
  const scaleMax = Math.max(0.05, ...rates.map(rateOf));
  return (
    <div className="table-wrap">
      <table className="rates">
        <caption>Flag rate by given label</caption>
        <thead>
          <tr>
            <th scope="col">Class</th>
            <th scope="col" className="num">
              Flagged
            </th>
            <th scope="col" className="num">
              Rows
            </th>
            <th scope="col">Rate</th>
          </tr>
        </thead>
        <tbody>
          {rates.map((r) => {
            const cap = capCount(r.total);
            const atCap = cap !== null && r.flagged >= cap && r.flagged > 0;
            return (
              <tr key={r.label}>
                <th scope="row">{classes[r.label]}</th>
                <td className="num">{r.flagged.toLocaleString()}</td>
                <td className="num">{r.total.toLocaleString()}</td>
                <td>
                  <span className="rate-cell">
                    <span className="rate-bar" aria-hidden="true">
                      <span style={{ width: `${(rateOf(r) / scaleMax) * 100}%` }} />
                    </span>
                    <span className="num">{(rateOf(r) * 100).toFixed(1)}%</span>
                    {atCap && <span className="cap-note">at cap</span>}
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
