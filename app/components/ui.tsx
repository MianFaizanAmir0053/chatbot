"use client";

import React, { useEffect, useRef, useState } from "react";

/** Honours the OS "reduce motion" setting for JS-driven animation. */
function prefersReducedMotion(): boolean {
  return (
    typeof window !== "undefined" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}

/**
 * Counts from the previous value to the next one.
 *
 * A dashboard number that jumps has no direction: 24 replacing 18 and 24
 * replacing 31 look identical. Rolling the digits shows which way it moved
 * and how far, which is the only reason the number is on screen at all.
 */
export function useCountUp(target: number, duration = 650): number {
  // null means "nothing in flight, render the target as given" — so a value
  // that arrives while motion is reduced, or that never moves, needs no state
  // write at all.
  const [tween, setTween] = useState<number | null>(null);
  const shownRef = useRef(target);
  const previousRef = useRef(target);

  shownRef.current = tween ?? target;

  useEffect(() => {
    const from = previousRef.current;
    previousRef.current = target;
    if (from === target || prefersReducedMotion()) return;

    // Start from whatever is on screen, so a value that changes again mid-roll
    // continues from there instead of snapping back.
    const start = performance.now();
    const origin = shownRef.current;
    let frame = requestAnimationFrame(function tick(now: number) {
      const t = Math.min(1, (now - start) / duration);
      // Ease-out cubic: quick off the mark, settled at the end.
      const eased = 1 - Math.pow(1 - t, 3);
      if (t < 1) {
        setTween(Math.round(origin + (target - origin) * eased));
        frame = requestAnimationFrame(tick);
      } else {
        setTween(null);
      }
    });

    return () => cancelAnimationFrame(frame);
  }, [target, duration]);

  return tween ?? target;
}

/* ------------------------------------------------------------------ *
 * Shared presentational primitives
 *
 * Kept deliberately small: a tone scale, a card, a stat tile and a few
 * pieces of chrome. Everything else composes these so spacing, radius
 * and colour stay consistent between the chat and the dashboard.
 * ------------------------------------------------------------------ */

export type Tone = "neutral" | "accent" | "success" | "warn" | "danger" | "info";

const TONE_FG: Record<Tone, string> = {
  neutral: "var(--foreground)",
  accent: "var(--accent-color)",
  success: "var(--success)",
  warn: "var(--warn)",
  danger: "var(--danger)",
  info: "var(--info)",
};

const TONE_BG: Record<Tone, string> = {
  neutral: "var(--surface-muted)",
  accent: "var(--accent-soft)",
  success: "var(--success-soft)",
  warn: "var(--warn-soft)",
  danger: "var(--danger-soft)",
  info: "var(--info-soft)",
};

export function toneFor(ok: boolean | undefined, pendingTone: Tone = "warn"): Tone {
  if (ok === undefined) return pendingTone;
  return ok ? "success" : "danger";
}

/** A small status chip. `dot` adds a leading indicator; `pulse` animates it. */
export function Pill({
  tone = "neutral",
  dot = false,
  pulse = false,
  children,
  className = "",
  title,
}: {
  tone?: Tone;
  dot?: boolean;
  pulse?: boolean;
  children: React.ReactNode;
  className?: string;
  title?: string;
}) {
  return (
    <span
      title={title}
      className={`inline-flex items-center gap-1.5 rounded-full px-2 py-1 text-[11px] font-semibold leading-none whitespace-nowrap transition-colors ${className}`}
      style={{ background: TONE_BG[tone], color: TONE_FG[tone] }}
    >
      {dot && (
        <span
          className={`w-1.5 h-1.5 rounded-full shrink-0 ${pulse ? "animate-pulse-dot" : ""}`}
          style={{ background: TONE_FG[tone] }}
        />
      )}
      {children}
    </span>
  );
}

export function Card({
  children,
  className = "",
  padded = true,
  interactive = false,
}: {
  children: React.ReactNode;
  className?: string;
  padded?: boolean;
  /** Adds hover feedback. Only for cards that lead somewhere or do something. */
  interactive?: boolean;
}) {
  return (
    <section
      className={`rounded-md border border-border bg-card ${interactive ? "lift" : ""} ${
        padded ? "p-4" : ""
      } ${className}`}
    >
      {children}
    </section>
  );
}

export function CardHeader({
  icon,
  title,
  subtitle,
  action,
}: {
  icon?: React.ReactNode;
  title: string;
  subtitle?: string;
  action?: React.ReactNode;
}) {
  return (
    <header className="flex items-start justify-between gap-3 mb-4">
      <div className="flex items-center gap-2.5 min-w-0">
        {icon && (
          <span className="shrink-0 grid place-items-center w-8 h-8 rounded-md bg-secondary text-foreground">
            {icon}
          </span>
        )}
        <div className="min-w-0">
          <h2 className="font-display text-[13px] font-semibold text-foreground truncate">
            {title}
          </h2>
          {subtitle && <p className="mt-0.5 text-[11px] text-muted-foreground truncate">{subtitle}</p>}
        </div>
      </div>
      {action && <div className="shrink-0">{action}</div>}
    </header>
  );
}

/** A headline metric. `hint` sits under the value, `trailing` to its right. */
export function Stat({
  label,
  value,
  hint,
  icon,
  tone = "neutral",
  trailing,
}: {
  label: string;
  value: React.ReactNode;
  hint?: React.ReactNode;
  icon?: React.ReactNode;
  tone?: Tone;
  trailing?: React.ReactNode;
}) {
  return (
    <div className="rounded-md border border-border bg-card p-4">
      <div className="flex items-center justify-between gap-2">
        <span className="eyebrow">{label}</span>
        {icon && (
          <span className="shrink-0" style={{ color: TONE_FG[tone] }}>
            {icon}
          </span>
        )}
      </div>
      <div className="mt-4 flex items-baseline gap-2">
        <span className="tnum font-display text-2xl font-semibold text-foreground leading-none">
          {typeof value === "number" ? <CountUp value={value} /> : value}
        </span>
        {trailing}
      </div>
      {hint && <div className="mt-1 text-[11px] text-muted-foreground truncate">{hint}</div>}
    </div>
  );
}

/** Renders a number that rolls to its new value when it changes. */
export function CountUp({ value }: { value: number }) {
  return <>{useCountUp(value).toLocaleString()}</>;
}

/** A label/value row inside a subsystem card. */
export function Row({
  label,
  children,
  mono = false,
}: {
  label: string;
  children: React.ReactNode;
  mono?: boolean;
}) {
  return (
    <div className="flex items-center justify-between gap-3 py-2 border-b border-border last:border-0">
      <span className="text-[11px] text-muted-foreground shrink-0">{label}</span>
      <span
        className={`text-[11px] text-foreground text-right min-w-0 truncate ${mono ? "font-mono" : "font-medium"}`}
      >
        {children}
      </span>
    </div>
  );
}

/** Proportional bar used for share-of-index in the document table. */
export function Meter({ value, tone = "accent" }: { value: number; tone?: Tone }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  // Painted at zero on mount so the first render animates out to its share
  // rather than appearing already full.
  const [grown, setGrown] = useState(false);
  useEffect(() => {
    const frame = requestAnimationFrame(() => setGrown(true));
    return () => cancelAnimationFrame(frame);
  }, []);

  return (
    <div className="h-1.5 w-full rounded-full overflow-hidden bg-secondary" role="presentation">
      <div
        className="h-full rounded-full transition-[width] duration-700 ease-[cubic-bezier(0.32,0.72,0,1)]"
        style={{ width: grown ? `${pct}%` : "0%", background: TONE_FG[tone] }}
      />
    </div>
  );
}

type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "ghost" | "danger";
  size?: "sm" | "md";
};

export function Button({
  variant = "secondary",
  size = "md",
  className = "",
  style,
  ...rest
}: ButtonProps) {
  const sizing = size === "sm" ? "h-8 px-3 text-[12px] gap-1.5" : "h-9 px-3.5 text-[13px] gap-2";
  const base =
    "press inline-flex items-center justify-center rounded-md font-medium " +
    "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring " +
    "disabled:opacity-45 disabled:pointer-events-none select-none";

  const variants: Record<string, string> = {
    primary: "bg-primary text-primary-foreground hover:opacity-90",
    secondary: "border border-border bg-secondary/60 text-foreground hover:bg-accent",
    ghost: "text-muted-foreground hover:bg-accent hover:text-foreground",
    danger: "text-[var(--danger)] hover:bg-[var(--danger-soft)]",
  };

  return (
    <button
      {...rest}
      className={`${base} ${sizing} ${variants[variant]} ${className}`}
      style={style}
    />
  );
}

/** Section heading used between groups of cards on the dashboard. */
export function SectionTitle({
  children,
  action,
}: {
  children: React.ReactNode;
  action?: React.ReactNode;
}) {
  return (
    <div className="flex items-end justify-between gap-3 mb-3">
      <h2 className="eyebrow font-semibold">{children}</h2>
      {action}
    </div>
  );
}

export function Skeleton({ className = "h-4 w-full" }: { className?: string }) {
  return <div className={`shimmer rounded-md ${className}`} />;
}

export function EmptyState({
  icon,
  title,
  body,
  action,
}: {
  icon?: React.ReactNode;
  title: string;
  body?: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="text-center py-10 px-6">
      {icon && (
        <span className="mx-auto mb-3 grid place-items-center w-10 h-10 rounded-md bg-secondary text-muted-foreground">
          {icon}
        </span>
      )}
      <p className="font-display text-[13px] font-semibold text-foreground">{title}</p>
      {body && (
        <p className="mt-1 text-[11px] text-muted-foreground max-w-xs mx-auto leading-relaxed">
          {body}
        </p>
      )}
      {action && <div className="mt-4 flex justify-center">{action}</div>}
    </div>
  );
}

/* ------------------------------------------------------------------ *
 * Formatting
 * ------------------------------------------------------------------ */

export function compactNumber(n: number): string {
  if (!Number.isFinite(n)) return "—";
  if (n < 1000) return String(n);
  if (n < 1_000_000) return `${(n / 1000).toFixed(n < 10_000 ? 1 : 0)}k`;
  return `${(n / 1_000_000).toFixed(1)}M`;
}

export function relativeTime(iso: string | null): string {
  if (!iso) return "—";
  const delta = Math.round((Date.now() - new Date(iso).getTime()) / 1000);
  if (delta < 5) return "just now";
  if (delta < 60) return `${delta}s ago`;
  if (delta < 3600) return `${Math.floor(delta / 60)}m ago`;
  return `${Math.floor(delta / 3600)}h ago`;
}
