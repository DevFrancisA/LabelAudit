// Runs the in-browser audit off the main thread. Cancelling terminates the worker.

import { runBrowserAudit, type BrowserAuditOptions, type BrowserAuditResult } from "./lib/pipeline";

export interface AuditRequest {
  texts: string[];
  labels: number[];
  numClasses: number;
  options: BrowserAuditOptions;
}

export type AuditResponse =
  | { type: "progress"; stage: string; done: number; total: number }
  | { type: "done"; result: BrowserAuditResult }
  | { type: "error"; message: string };

const scope = self as unknown as {
  onmessage: ((event: MessageEvent<AuditRequest>) => void) | null;
  postMessage(message: AuditResponse): void;
};

scope.onmessage = ({ data }) => {
  try {
    let lastReport = 0;
    const result = runBrowserAudit(data.texts, data.labels, data.numClasses, data.options, (stage, done, total) => {
      const now = performance.now();
      if (done === 0 || done === total || now - lastReport > 50) {
        lastReport = now;
        scope.postMessage({ type: "progress", stage, done, total });
      }
    });
    scope.postMessage({ type: "done", result });
  } catch (error) {
    scope.postMessage({ type: "error", message: error instanceof Error ? error.message : String(error) });
  }
};
