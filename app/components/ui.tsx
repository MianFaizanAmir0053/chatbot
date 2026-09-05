"use client";

import React from "react";

/* ------------------------------------------------------------------ *
 * Shared presentational primitives
 *
 * Kept deliberately small: a tone scale, a card, a stat tile and a few
 * pieces of chrome. Everything else composes these so spacing, radius
 * and colour stay consistent between the chat and the dashboard.
 * ------------------------------------------------------------------ */

export type Tone = "neutral" | "accent" | "success" | "warn" | "danger" | "info";

const TONE_FG: Record<Tone, string> = {
  neutral: "var(--ink-2)",
  accent: "var(--accent)",
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
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-medium leading-none whitespace-nowrap ${className}`}
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
}: {
  children: React.ReactNode;
  className?: string;
  padded?: boolean;
}) {
  return (
    <section
      className={`rounded-xl border border-line bg-surface shadow-xs ${padded ? "p-5" : ""} ${className}`}
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
          <span
            className="shrink-0 grid place-items-center w-8 h-8 rounded-lg"
            style={{ background: "var(--accent-soft)", color: "var(--accent)" }}
          >
            {icon}
          </span>
        )}
        <div className="min-w-0">
          <h2 className="text-sm font-semibold tracking-tight text-ink truncate">{title}</h2>
          {subtitle && <p className="text-xs text-ink-3 truncate">{subtitle}</p>}
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
    <div className="rounded-xl border border-line bg-surface p-4 shadow-xs">
      <div className="flex items-center justify-between gap-2 mb-2.5">
        <span className="text-[11px] font-medium uppercase tracking-[0.07em] text-ink-3">
          {label}
        </span>
        {icon && (
          <span className="shrink-0" style={{ color: TONE_FG[tone] }}>
            {icon}
          </span>
        )}
      </div>
      <div className="flex items-baseline gap-2">
        <span className="tnum text-2xl font-semibold tracking-tight text-ink leading-none">
          {value}
        </span>
        {trailing}
      </div>
      {hint && <div className="mt-2 text-xs text-ink-3 truncate">{hint}</div>}
    </div>
  );
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
    <div className="flex items-center justify-between gap-3 py-2 border-b border-line last:border-0">
      <span className="text-xs text-ink-3 shrink-0">{label}</span>
      <span
        className={`text-xs text-ink-2 text-right min-w-0 truncate ${mono ? "font-mono" : "font-medium"}`}
      >
        {children}
      </span>
    </div>
  );
}

/** Proportional bar used for share-of-index in the document table. */
export function Meter({ value, tone = "accent" }: { value: number; tone?: Tone }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div
      className="h-1.5 w-full rounded-full overflow-hidden"
      style={{ background: "var(--surface-hover)" }}
      role="presentation"
    >
      <div
        className="h-full rounded-full transition-[width] duration-500"
        style={{ width: `${pct}%`, background: TONE_FG[tone] }}
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
  const sizing = size === "sm" ? "h-8 px-3 text-xs gap-1.5" : "h-9 px-3.5 text-[13px] gap-2";
  const base =
    "inline-flex items-center justify-center rounded-lg font-medium transition-colors duration-150 " +
    "disabled:opacity-45 disabled:pointer-events-none select-none";

  const variants: Record<string, string> = {
    primary: "text-[var(--accent-ink)] hover:brightness-[1.06]",
    secondary: "border border-line text-ink-2 hover:bg-surface-hover hover:text-ink",
    ghost: "text-ink-3 hover:bg-surface-hover hover:text-ink",
    danger: "text-[var(--danger)] hover:bg-[var(--danger-soft)]",
  };

  return (
    <button
      {...rest}
      className={`${base} ${sizing} ${variants[variant]} ${className}`}
      style={{
        ...(variant === "primary" ? { background: "var(--accent)" } : null),
        ...(variant === "secondary" ? { background: "var(--surface)" } : null),
        ...style,
      }}
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
      <h2 className="text-[11px] font-semibold uppercase tracking-[0.09em] text-ink-3">
        {children}
      </h2>
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
        <span
          className="mx-auto mb-3 grid place-items-center w-11 h-11 rounded-xl"
          style={{ background: "var(--surface-muted)", color: "var(--ink-3)" }}
        >
          {icon}
        </span>
      )}
      <p className="text-sm font-medium text-ink">{title}</p>
      {body && <p className="mt-1 text-xs text-ink-3 max-w-xs mx-auto leading-relaxed">{body}</p>}
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
