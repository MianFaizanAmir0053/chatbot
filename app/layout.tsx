import type { Metadata, Viewport } from "next";
import { IBM_Plex_Sans, Inter, JetBrains_Mono } from "next/font/google";
import { AppShell } from "./components/app-shell";
import { ConversationsProvider } from "./components/conversations-context";
import { KnowledgeProvider } from "./components/knowledge-context";
import "./globals.css";

// Loaded through next/font rather than an @import in the stylesheet: the CSS
// import blocks first paint on a third-party round-trip and reintroduces the
// flash of unstyled text that self-hosting exists to remove.
// Inter carries every piece of running text: it was drawn for screen UI at
// small sizes, which is most of this interface. IBM Plex Sans sets headings —
// it has enough character to separate them from body copy while keeping the
// open apertures that make a headline legible at a glance.
const display = IBM_Plex_Sans({
  variable: "--font-display-family",
  subsets: ["latin"],
  weight: ["500", "600", "700"],
  display: "swap",
});

const body = Inter({
  variable: "--font-body-family",
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
    { media: "(prefers-color-scheme: light)", color: "#f4f1ea" },
    { media: "(prefers-color-scheme: dark)", color: "#1c1a18" },
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
      <body className={`${display.variable} ${body.variable} ${mono.variable} antialiased`}>
        {/*
          Conversations sit outside knowledge, because documents are scoped to a
          conversation: the knowledge provider reads the active conversation to
          decide which documents it is describing, so it has to be the inner one.
        */}
        <ConversationsProvider>
          <KnowledgeProvider>
            <AppShell>{children}</AppShell>
          </KnowledgeProvider>
        </ConversationsProvider>
      </body>
    </html>
  );
}
