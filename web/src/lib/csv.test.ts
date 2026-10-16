import { describe, expect, it } from "vitest";
import { detectDelimiter, parseCsv, toCsv } from "./csv";

describe("parseCsv", () => {
  it("handles quotes, doubled quotes, embedded newlines and CRLF", () => {
    const text = 'text,label\r\n"Hello, world",a\r\n"She said ""hi""\nthen left",b\r\n';
    expect(parseCsv(text)).toEqual([
      ["text", "label"],
      ["Hello, world", "a"],
      ['She said "hi"\nthen left', "b"],
    ]);
  });

  it("strips a BOM and keeps a final row without a newline", () => {
    expect(parseCsv("﻿a,b\n1,2")).toEqual([
      ["a", "b"],
      ["1", "2"],
    ]);
  });

  it("keeps empty fields", () => {
    expect(parseCsv("a,,c\n,,\n")).toEqual([
      ["a", "", "c"],
      ["", "", ""],
    ]);
  });

  it("detects semicolon and tab delimiters from the header", () => {
    expect(detectDelimiter('text;label\n"a,b";c')).toBe(";");
    expect(parseCsv("text\tlabel\nx\ty")).toEqual([
      ["text", "label"],
      ["x", "y"],
    ]);
  });
});

describe("toCsv", () => {
  it("round-trips awkward fields", () => {
    const rows = [
      ["text", "label"],
      ['comma, "quote"', "x"],
      ["line\nbreak", " padded "],
    ];
    expect(parseCsv(toCsv(rows))).toEqual(rows);
  });
});
