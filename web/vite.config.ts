import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  base: "/LabelAudit/",
  plugins: [react()],
  test: {
    include: ["src/**/*.test.ts"],
  },
});
