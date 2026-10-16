// TF-IDF features: lowercase word unigrams and bigrams, sublinear term frequency,
// smoothed inverse document frequency, and L2-normalized rows in CSR form.

export interface SparseMatrix {
  rows: number;
  cols: number;
  indptr: Int32Array;
  indices: Int32Array;
  values: Float64Array;
}

export interface TfidfOptions {
  maxFeatures: number;
  minDf: number;
}

const WORD = /[\p{L}\p{N}]+/gu;

export function tokenize(text: string): string[] {
  return (text.toLowerCase().match(WORD) ?? []).filter((t) => t.length > 1 || /\p{N}/u.test(t));
}

export function terms(tokens: string[]): string[] {
  const out = tokens.slice();
  for (let i = 1; i < tokens.length; i++) out.push(`${tokens[i - 1]} ${tokens[i]}`);
  return out;
}

export function tfidf(texts: string[], options: TfidfOptions): { matrix: SparseMatrix; vocabulary: string[] } {
  const n = texts.length;
  const docCounts: Map<string, number>[] = [];
  const df = new Map<string, number>();
  for (const text of texts) {
    const counts = new Map<string, number>();
    for (const term of terms(tokenize(text))) counts.set(term, (counts.get(term) ?? 0) + 1);
    for (const term of counts.keys()) df.set(term, (df.get(term) ?? 0) + 1);
    docCounts.push(counts);
  }

  const vocabulary = [...df.entries()]
    .filter(([, d]) => d >= options.minDf)
    .sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
    .slice(0, options.maxFeatures)
    .map(([term]) => term);
  const column = new Map(vocabulary.map((term, j) => [term, j]));
  const idf = vocabulary.map((term) => Math.log((1 + n) / (1 + (df.get(term) ?? 0))) + 1);

  const indptr = new Int32Array(n + 1);
  const indices: number[] = [];
  const values: number[] = [];
  docCounts.forEach((counts, i) => {
    const entries: [number, number][] = [];
    for (const [term, count] of counts) {
      const j = column.get(term);
      if (j !== undefined) entries.push([j, (1 + Math.log(count)) * idf[j]]);
    }
    entries.sort((a, b) => a[0] - b[0]);
    const norm = Math.sqrt(entries.reduce((s, [, v]) => s + v * v, 0)) || 1;
    for (const [j, v] of entries) {
      indices.push(j);
      values.push(v / norm);
    }
    indptr[i + 1] = indices.length;
  });

  return {
    matrix: {
      rows: n,
      cols: vocabulary.length,
      indptr,
      indices: Int32Array.from(indices),
      values: Float64Array.from(values),
    },
    vocabulary,
  };
}
