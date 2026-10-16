import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { AuditTab } from "./components/AuditTab";
import { Explainer } from "./components/Explainer";
import { ExploreTab } from "./components/ExploreTab";

const TABS = [
  { id: "explore", label: "Explore sample" },
  { id: "audit", label: "Audit your own data" },
] as const;
type TabId = (typeof TABS)[number]["id"];

type Theme = "light" | "dark";

function storedTheme(): Theme | null {
  try {
    const value = localStorage.getItem("labelaudit-theme");
    return value === "light" || value === "dark" ? value : null;
  } catch {
    return null;
  }
}

function ThemeToggle() {
  const systemDark = () => window.matchMedia("(prefers-color-scheme: dark)").matches;
  const [theme, setTheme] = useState<Theme>(() => storedTheme() ?? (systemDark() ? "dark" : "light"));
  const next: Theme = theme === "dark" ? "light" : "dark";

  const choose = (value: Theme) => {
    setTheme(value);
    document.documentElement.dataset.theme = value;
    try {
      localStorage.setItem("labelaudit-theme", value);
    } catch {
      // Storage can be unavailable (private mode); the choice still applies to this visit.
    }
  };

  return (
    <button type="button" className="theme-toggle" onClick={() => choose(next)} aria-label={`Switch to ${next} theme`}>
      <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
        {theme === "dark" ? (
          <path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5Z" fill="currentColor" />
        ) : (
          <g stroke="currentColor" strokeWidth="2" strokeLinecap="round" fill="none">
            <circle cx="12" cy="12" r="4" fill="currentColor" stroke="none" />
            <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
          </g>
        )}
      </svg>
      <span>{theme === "dark" ? "Dark" : "Light"}</span>
    </button>
  );
}

function initialTab(): TabId {
  return window.location.hash === "#audit" ? "audit" : "explore";
}

export function App() {
  const [tab, setTab] = useState<TabId>(initialTab);
  const tabRefs = useRef<Record<TabId, HTMLButtonElement | null>>({ explore: null, audit: null });

  useEffect(() => {
    history.replaceState(null, "", tab === "audit" ? "#audit" : window.location.pathname);
  }, [tab]);

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const index = TABS.findIndex((t) => t.id === tab);
    const moves: Record<string, number> = { ArrowRight: index + 1, ArrowLeft: index - 1, Home: 0, End: TABS.length - 1 };
    if (!(event.key in moves)) return;
    event.preventDefault();
    const target = TABS[(moves[event.key] + TABS.length) % TABS.length].id;
    setTab(target);
    tabRefs.current[target]?.focus();
  };

  return (
    <>
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <header className="masthead">
        <div className="masthead-inner">
          <p className="wordmark">LabelAudit</p>
          <ThemeToggle />
        </div>
        <h1>
          Find the training labels that are probably{" "}
          <span className="circled">
            wrong
            <svg viewBox="0 0 120 52" preserveAspectRatio="none" aria-hidden="true">
              <path d="M8 30C6 14 40 5 66 6c28 1 50 9 48 23-2 15-38 20-62 19C26 47 7 41 9 27c1-7 9-12 17-15" />
            </svg>
          </span>
          .
        </h1>
        <p className="tagline">
          Rank every example by how likely its label is a mistake, see the evidence, and export a cleaned CSV. Everything
          on this page runs in your browser.
        </p>
      </header>

      <main id="main">
        <Explainer />
        <div className="tabs" role="tablist" aria-label="Mode" onKeyDown={onKeyDown}>
          {TABS.map((t) => (
            <button
              key={t.id}
              ref={(el) => {
                tabRefs.current[t.id] = el;
              }}
              type="button"
              role="tab"
              id={`tab-${t.id}`}
              aria-selected={tab === t.id}
              aria-controls={`panel-${t.id}`}
              tabIndex={tab === t.id ? 0 : -1}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </div>
        {TABS.map((t) => (
          <section
            key={t.id}
            id={`panel-${t.id}`}
            role="tabpanel"
            aria-labelledby={`tab-${t.id}`}
            hidden={tab !== t.id}
            className="panel"
          >
            {t.id === "explore" ? <ExploreTab /> : <AuditTab />}
          </section>
        ))}
      </main>

      <footer className="footer">
        <p>
          LabelAudit is open source under the MIT license.{" "}
          <a href="https://github.com/DevFrancisA/LabelAudit">Source, Python pipeline, and method notes on GitHub</a>.
        </p>
      </footer>
    </>
  );
}
