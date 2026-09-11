"use client";

import { memo, type ReactNode } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { visit } from "unist-util-visit";
import type { Parent, Root, RootContent, Text } from "mdast";

/* ------------------------------------------------------------------ *
 * Answer rendering
 *
 * Answers are markdown, and they were previously rendered by a handful of
 * regular expressions. That handled bold, italics, `###` and unordered lists,
 * which is a small fraction of what the models actually write — so numbered
 * lists arrived as a paragraph of runs ("1. Drive chain inspection: Every
 * 1,000 km 2. ..."), tables came through as rows of pipes, links stayed as
 * `[text](url)`, and every heading level but one was shown as literal hashes.
 *
 * Two of those regexes did active damage. `\n` became `<br />` across the whole
 * string, including inside code blocks, so any fenced code lost its layout; and
 * list items were emitted as bare `<li>` with no `<ul>` around them, which
 * browsers render inconsistently and which the same `<br />` rule then spaced
 * out as though each bullet were its own paragraph.
 *
 * A real parser replaces all of it. `react-markdown` builds React elements
 * rather than an HTML string, so there is no `dangerouslySetInnerHTML` here and
 * no way for document content to inject markup — which matters more than usual,
 * because what is rendered is a model's text quoting passages from files the
 * user uploaded.
 * ------------------------------------------------------------------ */

/** Where a citation's synthetic link points, so the renderer can recognise it. */
const CITATION_PREFIX = "#citation-";

const CITATION = /\[(\d+)\]/g;

/**
 * Turn `[1]` into something the renderer can style as a superscript.
 *
 * Done as a syntax-tree pass rather than a search-and-replace on the text,
 * because the markers have to be found *after* parsing: a `[3]` inside a code
 * span or a fenced block is code, not a citation, and a string-level
 * replacement cannot tell the difference. Visiting text nodes gets that for
 * free, since a parser has already decided what is code.
 *
 * The marker becomes an ordinary link node rather than a bespoke one. Custom
 * node types need a matching handler in the mdast-to-hast step to render at
 * all, while a link is understood by everything in the pipeline and is turned
 * into a superscript by the component override below.
 */
function remarkCitations() {
  return (tree: Root) => {
    visit(tree, "text", (node: Text, index: number | undefined, parent: Parent | undefined) => {
      if (!parent || index === undefined) return;
      // A real link's own text is not a citation, and nesting a link inside a
      // link produces invalid markup.
      if (parent.type === "link" || parent.type === "linkReference") return;
      if (!node.value.includes("[")) return;

      const parts: RootContent[] = [];
      let last = 0;
      CITATION.lastIndex = 0;
      for (let m = CITATION.exec(node.value); m; m = CITATION.exec(node.value)) {
        if (m.index > last) parts.push({ type: "text", value: node.value.slice(last, m.index) });
        parts.push({
          type: "link",
          url: `${CITATION_PREFIX}${m[1]}`,
          children: [{ type: "text", value: m[1] }],
        });
        last = m.index + m[0].length;
      }
      if (parts.length === 0) return;
      if (last < node.value.length) parts.push({ type: "text", value: node.value.slice(last) });

      parent.children.splice(index, 1, ...parts);
      // Resume past what was just inserted: the new text nodes have already
      // been scanned, and revisiting them would loop.
      return index + parts.length;
    });
  };
}

const PLUGINS = [remarkGfm, remarkCitations];

/**
 * Overrides are written to take only the props they need, never `...rest`.
 *
 * react-markdown hands every override the syntax-tree node it came from, and
 * spreading that onto the element renders `node="[object Object]"` into the
 * page. Naming the handful of attributes markdown can actually produce avoids
 * that by construction rather than by remembering to delete one prop.
 */
const COMPONENTS: Components = {
  a({ href, title, children }) {
    if (typeof href === "string" && href.startsWith(CITATION_PREFIX)) {
      return <sup className="citation-ref">{children as ReactNode}</sup>;
    }
    // Answers cite the open web, so a link leaves this app. Opened in a new tab
    // so a half-read answer is not navigated away from, and with `noreferrer`
    // because the destination is chosen by a model reading untrusted pages.
    return (
      <a href={href} title={title} target="_blank" rel="noopener noreferrer">
        {children as ReactNode}
      </a>
    );
  },
  // Wrapped so a wide table scrolls inside the message instead of stretching
  // the conversation column.
  table({ children }) {
    return (
      <div className="prose-table-wrap">
        <table>{children as ReactNode}</table>
      </div>
    );
  },
};

/**
 * Memoised on content because this re-parses on every streamed token: the
 * answer grows one chunk at a time and each chunk re-renders the message.
 */
export const Markdown = memo(function Markdown({ content }: { content: string }) {
  return (
    <div className="prose-content">
      <ReactMarkdown remarkPlugins={PLUGINS} components={COMPONENTS}>
        {content}
      </ReactMarkdown>
    </div>
  );
});
