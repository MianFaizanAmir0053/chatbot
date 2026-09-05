import type { Metadata, Viewport } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import { AppShell } from "./components/app-shell";
import { KnowledgeProvider } from "./components/knowledge-context";
import "./globals.css";

// Loaded through next/font rather than an @import in the stylesheet: the CSS
// import blocks first paint on a third-party round-trip and reintroduces the
// flash of unstyled text that self-hosting exists to remove.
const sans = Inter({
  variable: "--font-ui",
  subsets: ["latin"],
  display: "swap",
});

const mono = JetBrains_Mono({
  variable: "--font-code",
  subsets: ["latin"],
  weight: ["400", "500"],
  display: "swap",
});

export const metadata: Metadata = {
  title: {
    default: "Agentic RAG — Retrieval workspace",
    template: "%s · Agentic RAG",
  },
  description:
    "Upload documents and ask grounded questions. The agent plans, retrieves, verifies and cites every claim.",
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f5f7f9" },
    { media: "(prefers-color-scheme: dark)", color: "#0a0d12" },
  ],
};

/**
 * Applies the stored theme before first paint.
 *
 * Without this the page renders in the system theme and then snaps to the
 * user's choice a frame later, which reads as a bug.
 */
const THEME_SCRIPT = `try{var t=localStorage.getItem("theme");if(t==="light"||t==="dark")document.documentElement.setAttribute("data-theme",t)}catch(e){}`;

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body className={`${sans.variable} ${mono.variable} antialiased`}>
        <KnowledgeProvider>
          <AppShell>{children}</AppShell>
        </KnowledgeProvider>
      </body>
    </html>
  );
}
