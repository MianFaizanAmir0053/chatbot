/**
 * Does an answer survive the trip from markdown to the screen?
 *
 * Rendering used to be a handful of regular expressions, which covered bold,
 * italics, `###` and unordered lists. Everything else a model writes came
 * through as literal syntax — numbered lists flattened into one paragraph,
 * tables shown as rows of pipes, links left as `[text](url)`. Two of the rules
 * also did damage: `\n` became `<br />` across the whole string including
 * inside fenced code, and list items were emitted without any list around them.
 *
 * These checks render the component the way the app does and assert on the
 * markup, because the failure was never an exception — it was an answer that
 * arrived looking wrong.
 *
 * Usage: npx tsx --env-file=.env scripts/markdown-render-test.ts
 */

import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { Markdown } from "../app/components/markdown";

let failures = 0;
function check(name: string, ok: boolean, detail = "") {
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${name}${detail ? `  — ${detail}` : ""}`);
  if (!ok) failures++;
}
function banner(s: string) {
  console.log(`\n${"=".repeat(72)}\n${s}\n${"=".repeat(72)}`);
}

function render(markdown: string): string {
  return renderToStaticMarkup(createElement(Markdown, { content: markdown }));
}

function has(name: string, markdown: string, ...fragments: string[]) {
  const html = render(markdown);
  const missing = fragments.filter((f) => !html.includes(f));
  check(name, missing.length === 0, missing.length ? `missing ${missing.join(", ")} in ${html.slice(0, 160)}` : "");
}

function lists() {
  banner("1. Lists");

  // The observed failure: a numbered list arrived as one run of prose.
  has(
    "an ordered list becomes <ol>",
    "1. Drive chain inspection: every 1,000 km\n2. Oil change: every 6,000 km",
    "<ol>",
    "<li>Drive chain inspection: every 1,000 km</li>",
    "<li>Oil change: every 6,000 km</li>",
  );
  has("an unordered list is wrapped in <ul>", "- first\n- second", "<ul>", "<li>first</li>");
  has(
    "a nested list nests",
    "- outer\n  - inner",
    "<ul>",
    "inner",
  );
  const flat = render("1. one\n2. two");
  check("and carries no stray <br>", !flat.includes("<br"), flat.slice(0, 120));
}

function blocks() {
  banner("2. Block elements");

  has("a table renders as a table", "| a | b |\n| - | - |\n| 1 | 2 |", "<table", "<th>a</th>", "<td>1</td>");
  // react-markdown hands every component override the syntax-tree node. Spread
  // onward it becomes `node="[object Object]"` on the element — invalid, and
  // visible in the page source.
  const table = render("| a | b |\n| - | - |\n| 1 | 2 |");
  check("and no syntax-tree node leaks into the markup", !table.includes("node="), table.slice(0, 160));
  check(
    "and scrolls in its own box",
    render("| a | b |\n| - | - |\n| 1 | 2 |").includes('class="prose-table-wrap"'),
  );

  has("every heading level renders", "# One\n\n## Two\n\n#### Four", "<h1>One</h1>", "<h2>Two</h2>", "<h4>Four</h4>");
  has("a blockquote renders", "> quoted advice", "<blockquote>", "quoted advice");
  has("a horizontal rule renders", "above\n\n---\n\nbelow", "<hr");

  // The rule that flattened `\n` to `<br />` ran over the whole document, so a
  // fenced block lost its line structure entirely.
  const code = render("```js\nconst a = 1;\nconst b = 2;\n```");
  check("fenced code keeps its newlines", code.includes("const a = 1;\nconst b = 2;"), code.slice(0, 160));
  check("and contains no <br>", !code.includes("<br"), code.slice(0, 160));
}

function inline() {
  banner("3. Inline elements");

  has("bold", "**3.4 litres**", "<strong>3.4 litres</strong>");
  has("italic", "*approximately*", "<em>approximately</em>");
  has("inline code", "use `search_documents`", "<code>search_documents</code>");
  has("strikethrough (GFM)", "~~withdrawn~~", "<del>withdrawn</del>");

  const link = render("see the [manual](https://example.com/manual)");
  check(
    "a link renders and opens safely",
    link.includes('href="https://example.com/manual"') &&
      link.includes('target="_blank"') &&
      link.includes("noopener"),
    link.slice(0, 200),
  );
}

function citations() {
  banner("4. Citations");

  const cited = render("The capacity is 3.4 litres [1] and the grade is 10W-40 [2].");
  check(
    "a marker becomes a superscript anchor",
    cited.includes('<sup class="citation-ref">1</sup>') &&
      cited.includes('<sup class="citation-ref">2</sup>'),
    cited.slice(0, 200),
  );
  check("and leaves the prose intact", cited.includes("The capacity is 3.4 litres"), "");

  // A number in brackets inside code is code, not provenance. The old
  // string-level replacement could not tell the difference; parsing first can.
  const inlineCode = render("use `arr[0]` to read it");
  check(
    "a bracketed number in inline code is left alone",
    !inlineCode.includes("citation-ref"),
    inlineCode.slice(0, 180),
  );
  const fenced = render("```python\nvalues[2] = 3\n```");
  check(
    "and one in a fenced block too",
    !fenced.includes("citation-ref"),
    fenced.slice(0, 180),
  );

  // A real markdown link whose text is a number must stay a link.
  const numbered = render("see [1](https://example.com)");
  check(
    "a link with a numeric label stays a link",
    numbered.includes('href="https://example.com"') && !numbered.includes("citation-ref"),
    numbered.slice(0, 180),
  );
}

function safety() {
  banner("5. Untrusted content");

  // What is rendered is a model's text quoting passages from uploaded files, so
  // markup in either must never become live markup here.
  const injected = render('Normal text <img src=x onerror="alert(1)"> and <script>alert(2)</script> more');
  // Escaped, so the markup is shown as the characters the model wrote rather
  // than becoming elements. Asserted on the opening angle brackets: the
  // attribute text survives as `onerror=&quot;`, which is inert.
  check(
    "raw HTML in an answer is not executed",
    !injected.includes("<script") && !injected.includes("<img"),
    injected.slice(0, 200),
  );
  check(
    "and is shown as text instead",
    injected.includes("&lt;script&gt;"),
    injected.slice(0, 200),
  );

  const scheme = render("[click me](javascript:alert(1))");
  check(
    "a javascript: link is defused",
    !scheme.includes("javascript:alert"),
    scheme.slice(0, 160),
  );
}

function main() {
  lists();
  blocks();
  inline();
  citations();
  safety();

  banner("Summary");
  console.log(failures === 0 ? "  ALL CHECKS PASSED" : `  ${failures} CHECK(S) FAILED`);
  process.exit(failures === 0 ? 0 : 1);
}

main();

export {};
