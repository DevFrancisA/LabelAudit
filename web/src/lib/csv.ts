// Minimal RFC 4180 CSV reading and writing: quoted fields, doubled quotes,
// embedded newlines, CRLF or LF line endings, and an optional UTF-8 BOM.

const CANDIDATE_DELIMITERS = [",", ";", "\t"] as const;

/** Pick the delimiter that splits the header line into the most fields. */
export function detectDelimiter(text: string): string {
  let best: string = ",";
  let bestCount = 0;
  for (const delimiter of CANDIDATE_DELIMITERS) {
    let count = 0;
    let quoted = false;
    for (const ch of text) {
      if (ch === '"') quoted = !quoted;
      else if (!quoted && (ch === "\n" || ch === "\r")) break;
      else if (!quoted && ch === delimiter) count++;
    }
    if (count > bestCount) {
      best = delimiter;
      bestCount = count;
    }
  }
  return best;
}

export function parseCsv(input: string, delimiter = detectDelimiter(input)): string[][] {
  const text = input.charCodeAt(0) === 0xfeff ? input.slice(1) : input;
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let quoted = false;
  let i = 0;

  while (i < text.length) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i += 2;
          continue;
        }
        quoted = false;
      } else {
        field += ch;
      }
      i++;
      continue;
    }
    if (ch === '"' && field === "") {
      quoted = true;
    } else if (ch === delimiter) {
      row.push(field);
      field = "";
    } else if (ch === "\n" || ch === "\r") {
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
      if (ch === "\r" && text[i + 1] === "\n") i++;
    } else {
      field += ch;
    }
    i++;
  }
  if (field !== "" || row.length > 0) {
    row.push(field);
    rows.push(row);
  }
  return rows;
}

function quote(field: string): string {
  return /[",\r\n]|^\s|\s$/.test(field) ? `"${field.replace(/"/g, '""')}"` : field;
}

export function toCsv(rows: string[][]): string {
  return rows.map((row) => row.map(quote).join(",")).join("\r\n") + "\r\n";
}
