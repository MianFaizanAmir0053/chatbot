"use client";

import { useSyncExternalStore } from "react";
import { MoonIcon, SunIcon } from "./icons";

type Theme = "light" | "dark" | "system";

const ORDER: Theme[] = ["system", "light", "dark"];
const LABEL: Record<Theme, string> = { system: "System", light: "Light", dark: "Dark" };

/**
 * The current theme lives on <html data-theme>, which the inline script in the
 * root layout stamps from localStorage before first paint. Subscribing to that
 * attribute rather than mirroring it into component state keeps the DOM the
 * single source of truth and avoids a render pass that corrects itself.
 */
const listeners = new Set<() => void>();

function subscribe(fn: () => void): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

function getSnapshot(): Theme {
  const attr = document.documentElement.getAttribute("data-theme");
  return attr === "light" || attr === "dark" ? attr : "system";
}

/** The server cannot know the stored preference; "system" is what it renders. */
function getServerSnapshot(): Theme {
  return "system";
}

function applyTheme(next: Theme) {
  const root = document.documentElement;
  if (next === "system") {
    root.removeAttribute("data-theme");
    try {
      window.localStorage.removeItem("theme");
    } catch {
      /* storage can be blocked; the in-session choice still applies */
    }
  } else {
    root.setAttribute("data-theme", next);
    try {
      window.localStorage.setItem("theme", next);
    } catch {
      /* as above */
    }
  }
  listeners.forEach((fn) => fn());
}

export function ThemeToggle() {
  const theme = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
  const next = ORDER[(ORDER.indexOf(theme) + 1) % ORDER.length];

  return (
    <button
      type="button"
      onClick={() => applyTheme(next)}
      title={`Theme: ${LABEL[theme]} — switch to ${LABEL[next]}`}
      aria-label={`Theme: ${LABEL[theme]}. Switch to ${LABEL[next]}.`}
      className="press grid h-8 w-8 place-items-center rounded-md border border-border text-muted-foreground hover:bg-accent hover:text-foreground"
    >
      {/* Keyed so React remounts the icon and the fade plays on every change. */}
      <span key={theme} className="grid animate-pop place-items-center">
        {theme === "dark" ? <MoonIcon className="w-4 h-4" /> : <SunIcon className="w-4 h-4" />}
      </span>
    </button>
  );
}
