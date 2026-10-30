/**
 * In-place editing for posts. The floating button switches the article into
 * edit mode (headings, paragraphs and table cells become editable); pressing
 * it again converts the changed blocks back to Markdown, splices them into
 * the .md file at the offsets from utils/remarkSourceMap.ts, and commits the
 * file to GitHub, which redeploys the site.
 */
import { forget, getToken, hasVault, lock } from "./vault";

const REPO = "lxy-alexander/VoidIndex";
const BRANCH = "main";
const UNITS = "#article [data-md-s], #article [data-new]";
const ZWSP = "\u200b";

// The innermost editable blocks, in page order. A loose list item renders as
// <li><p> with the same offsets on both; only the <p> is edited.
function units(withDeleted = false): HTMLElement[] {
  return [...document.querySelectorAll<HTMLElement>(UNITS)].filter(
    el =>
      !el.querySelector("[data-md-s], [data-new]") &&
      (withDeleted || el.dataset.deleted === undefined)
  );
}

// Inline Markdown of each existing block when edit mode started.
let before = new Map<HTMLElement, string>();

// Inline HTML back to Markdown: bold, italic, code and links survive.
function toMarkdown(node: Node): string {
  if (node.nodeType === Node.TEXT_NODE) {
    return (
      (node.textContent ?? "")
        .replace(/ /g, " ")
        .replace(/​/g, "")
        // A literal asterisk in the text must stay literal in Markdown.
        .replace(/\*/g, "\\*")
    );
  }
  if (!(node instanceof HTMLElement)) return "";
  const inner = () => [...node.childNodes].map(toMarkdown).join("");
  if (node.classList.contains("heading-link")) return "";
  switch (node.tagName) {
    case "STRONG":
    case "B":
      return `**${inner()}**`;
    case "EM":
    case "I":
      return `*${inner()}*`;
    case "CODE":
      return `\`${node.textContent ?? ""}\``;
    case "MARK":
      return `<mark>${inner()}</mark>`;
    case "A":
      return `[${inner()}](${node.getAttribute("href") ?? ""})`;
    case "BR":
      return " ";
    case "DIV":
    case "P":
      return ` ${inner()}`;
    default:
      return inner();
  }
}

function blockMarkdown(el: HTMLElement): string {
  return [...el.childNodes]
    .map(toMarkdown)
    .join("")
    .replace(/\s+/g, " ")
    .trim();
}

// Markdown for one editable block. A code block is the whole fence; every
// other block is its inline content (block markers live outside the range).
function unitMarkdown(el: HTMLElement): string {
  if (el.tagName === "PRE") {
    const code = (el.textContent ?? "")
      .replace(/\u200b/g, "")
      .replace(/\n$/, "");
    return `\`\`\`${el.dataset.lang ?? ""}\n${code}\n\`\`\``;
  }
  return blockMarkdown(el);
}

// Block-level Markdown that goes in front of a new or converted block.
function blockPrefix(el: HTMLElement, listPrefix: string): string {
  const level = /^H([1-6])$/.exec(el.tagName)?.[1];
  if (level) return `${"#".repeat(Number(level))} `;
  if (el.tagName === "PRE") return "";
  if (el.tagName === "LI") {
    return listPrefix || (el.parentElement?.tagName === "OL" ? "1. " : "- ");
  }
  if (el.parentElement?.tagName === "BLOCKQUOTE") return "> ";
  return "";
}

async function sha1(text: string): Promise<string> {
  const digest = await crypto.subtle.digest(
    "SHA-1",
    new TextEncoder().encode(text)
  );
  return [...new Uint8Array(digest)]
    .map(b => b.toString(16).padStart(2, "0"))
    .join("");
}

function decodeBase64(b64: string): string {
  const bin = atob(b64.replace(/\n/g, ""));
  return new TextDecoder().decode(Uint8Array.from(bin, c => c.charCodeAt(0)));
}

function encodeBase64(text: string): string {
  const bytes = new TextEncoder().encode(text);
  let bin = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return btoa(bin);
}

function toast(message: string): void {
  const el = document.createElement("div");
  el.className = "edit-toast";
  el.textContent = message;
  document.body.append(el);
  setTimeout(() => el.remove(), 5000);
}

type Piece = { el: HTMLElement; start: number; length: number };
type Edit = { s: number; e: number; text: string; pieces: Piece[] };

// Group blocks by the existing block they follow: each group rewrites that
// block's source range with its new text plus any new blocks after it.
function buildEdits(body: string, raw: string, bodyStart: number): Edit[] {
  const edits: Edit[] = [];
  let current: Edit | null = null;
  let anchorPrefix = "";
  let prev: HTMLElement | null = null;

  const flush = () => {
    if (current && current.pieces.some(p => p.el.dataset.dirty !== undefined)) {
      edits.push(current);
    }
  };

  for (const el of units(true)) {
    // A block merged into the one above: drop its source up to its end.
    if (el.dataset.deleted !== undefined) {
      if (current) current.e = Number(el.dataset.mdE);
      continue;
    }
    if (el.dataset.new === undefined) {
      flush();
      const converted = el.dataset.was !== undefined;
      let s = Number(el.dataset.mdS);
      const e = Number(el.dataset.mdE);
      // A heading's range starts after its "## "; widen it to the line start
      // when the block type changes so the old marker is replaced.
      if (converted && /^H[1-6]$/.test(el.dataset.was ?? "")) {
        s = raw.lastIndexOf("\n", bodyStart + s - 1) + 1 - bodyStart;
      }
      const inline = unitMarkdown(el);
      const changed = converted || inline !== before.get(el);
      if (changed) el.dataset.dirty = "";
      else delete el.dataset.dirty;
      const prefix = converted ? blockPrefix(el, "") : "";
      const text = changed ? prefix + inline : body.slice(s, e);
      current = {
        s,
        e,
        text,
        pieces: [
          {
            el,
            start: changed ? prefix.length : 0,
            length: changed ? inline.length : e - s,
          },
        ],
      };
      // Reuse the list marker ("- ", "  1. ") for new items after this one.
      const lineStart = raw.lastIndexOf("\n", bodyStart + s - 1) + 1;
      anchorPrefix =
        el.tagName === "LI" ? raw.slice(lineStart, bodyStart + s) : "";
      prev = el;
      continue;
    }
    const inline = unitMarkdown(el);
    el.dataset.dirty = "";
    if (!current) continue;
    const sameList =
      el.tagName === "LI" &&
      prev?.tagName === "LI" &&
      prev.parentElement === el.parentElement;
    const prefix = blockPrefix(el, sameList ? anchorPrefix : "");
    const sep = sameList ? "\n" : "\n\n";
    current.text += sep + prefix;
    current.pieces.push({
      el,
      start: current.text.length,
      length: inline.length,
    });
    current.text += inline;
    prev = el;
  }
  flush();
  return edits;
}

type Result = "saved" | "unchanged" | "cancelled";

async function save(path: string): Promise<Result> {
  const dirty =
    document.querySelector("#article [data-deleted]") !== null ||
    units().some(
      el =>
        el.dataset.new !== undefined ||
        el.dataset.was !== undefined ||
        unitMarkdown(el) !== before.get(el)
    );
  if (!dirty) return "unchanged";

  const token = await getToken();
  if (!token) return "cancelled";
  const api = `https://api.github.com/repos/${REPO}/contents/${path}`;
  const headers = {
    Authorization: `Bearer ${token}`,
    Accept: "application/vnd.github+json",
  };

  const res = await fetch(`${api}?ref=${BRANCH}`, { headers });
  if (res.status === 401) forget();
  if (!res.ok) throw new Error(`Reading the file failed (${res.status})`);
  const file = (await res.json()) as { content: string; sha: string };
  const raw = decodeBase64(file.content);

  // Offsets are relative to the trimmed body after the frontmatter.
  const fm = raw.match(/^---\r?\n[\s\S]*?\r?\n---/)?.[0] ?? "";
  const rest = raw.slice(fm.length);
  const bodyStart = fm.length + (rest.length - rest.trimStart().length);
  const body = rest.trim();
  const hashEl = document.querySelector<HTMLElement>("#md-source");
  if ((await sha1(body)) !== hashEl?.dataset.hash) {
    throw new Error(
      "The note changed since this page was built. Reload after the deploy finishes, then edit again."
    );
  }

  const edits = buildEdits(body, raw, bodyStart);
  let next = raw;
  for (const { s, e, text } of [...edits].sort((a, b) => b.s - a.s)) {
    next = next.slice(0, bodyStart + s) + text + next.slice(bodyStart + e);
  }

  const title =
    document
      .querySelector("main h1")
      ?.textContent?.replace(/\s+/g, " ")
      .trim() ?? path;
  const put = await fetch(api, {
    method: "PUT",
    headers,
    body: JSON.stringify({
      message: `Edit: ${title}`,
      content: encodeBase64(next),
      sha: file.sha,
      branch: BRANCH,
    }),
  });
  if (put.status === 401) forget();
  if (!put.ok) throw new Error(`Saving failed (${put.status})`);

  // Recompute offsets so the next save on this page lands in the right place.
  const delta = (x: Edit) => x.text.length - (x.e - x.s);
  const shiftBefore = (pos: number, self?: Edit) =>
    edits
      .filter(x => x !== self && x.e <= pos)
      .reduce((n, x) => n + delta(x), 0);
  const placed = new Set<HTMLElement>();
  for (const x of edits) {
    const newS = x.s + shiftBefore(x.s, x);
    for (const piece of x.pieces) {
      piece.el.dataset.mdS = String(newS + piece.start);
      piece.el.dataset.mdE = String(newS + piece.start + piece.length);
      placed.add(piece.el);
    }
  }
  document.querySelectorAll<HTMLElement>("#article [data-md-s]").forEach(el => {
    if (placed.has(el)) return;
    const inner = el.querySelector<HTMLElement>("[data-md-s]");
    if (inner) return;
    const s = Number(el.dataset.mdS);
    const e = Number(el.dataset.mdE);
    const shift = shiftBefore(s);
    el.dataset.mdS = String(s + shift);
    el.dataset.mdE = String(e + shift);
  });
  // A loose <li> mirrors its inner <p>.
  document
    .querySelectorAll<HTMLElement>("#article li[data-md-s]")
    .forEach(li => {
      const inner = li.querySelector<HTMLElement>(":scope > p[data-md-s]");
      if (inner) {
        li.dataset.mdS = inner.dataset.mdS;
        li.dataset.mdE = inner.dataset.mdE;
      }
    });
  document.querySelectorAll("#article [data-deleted]").forEach(t => t.remove());
  for (const el of units()) {
    delete el.dataset.new;
    delete el.dataset.was;
    delete el.dataset.dirty;
  }
  hashEl.dataset.hash = await sha1(next.slice(fm.length).trim());
  return "saved";
}

// Markdown typed inline turns into formatting as soon as it is closed,
// like Typora or a live Markdown preview.
const SHORTCUTS: [RegExp, (m: RegExpMatchArray) => HTMLElement][] = [
  [/\[([^\]]+)\]\(([^)\s]+)\)$/, m => el("a", m[1], m[2])],
  [/\*\*([^*]+)\*\*$/, m => el("strong", m[1])],
  [/(?<!\*)\*([^*\s][^*]*)\*$/, m => el("em", m[1])],
  [/`([^`]+)`$/, m => el("code", m[1])],
  [/==([^=]+)==$/, m => el("mark", m[1])],
];

function el(tag: string, text: string, href?: string): HTMLElement {
  const node = document.createElement(tag);
  node.textContent = text;
  if (href) node.setAttribute("href", href);
  return node;
}

function caretUnit(): HTMLElement | null {
  const node = getSelection()?.anchorNode;
  const el = node instanceof Element ? node : node?.parentElement;
  const unit = el?.closest<HTMLElement>("[contenteditable='true']");
  return unit && units().includes(unit) ? unit : null;
}

// A replacement block keeps the source range of the block it replaces, and
// remembers that block's original type so the save rewrites its marker.
function makeUnit(tag: string, from?: HTMLElement): HTMLElement {
  const el = document.createElement(tag);
  el.contentEditable = "true";
  if (from?.dataset.mdS !== undefined) {
    el.dataset.mdS = from.dataset.mdS;
    el.dataset.mdE = from.dataset.mdE;
    el.dataset.was = from.dataset.was ?? from.tagName;
  } else {
    el.dataset.new = "";
  }
  return el;
}

function placeCaret(el: Node, atEnd = false): void {
  const range = document.createRange();
  range.selectNodeContents(el);
  range.collapse(!atEnd);
  getSelection()?.removeAllRanges();
  getSelection()?.addRange(range);
}

function applyInlineShortcut(): void {
  const sel = getSelection();
  const node = sel?.anchorNode;
  if (!sel?.isCollapsed || !(node instanceof Text)) return;
  if (node.parentElement?.closest("pre")) return;
  const offset = sel.anchorOffset;
  const typed = node.data.slice(0, offset);
  for (const [pattern, make] of SHORTCUTS) {
    const m = typed.match(pattern);
    if (!m) continue;
    // A zero-width space gives the caret somewhere to sit after the new tag.
    const rest = node.data.slice(offset) || ZWSP;
    node.data = node.data.slice(0, offset - m[0].length);
    const tail = document.createTextNode(rest);
    node.after(make(m), tail);
    sel.collapse(tail, rest === ZWSP ? 1 : 0);
    return;
  }
}

type Kind =
  | "P"
  | "H1"
  | "H2"
  | "H3"
  | "H4"
  | "H5"
  | "H6"
  | "UL"
  | "OL"
  | "QUOTE"
  | "CODE";

// Only plain paragraphs and headings change type; list items, quotes and
// table cells keep theirs.
function convertible(unit: HTMLElement): boolean {
  if (/^H[1-6]$/.test(unit.tagName)) return true;
  if (unit.tagName !== "P") return false;
  const wrapper = unit.parentElement;
  return (
    !unit.closest("li, td") &&
    (wrapper?.tagName !== "BLOCKQUOTE" || wrapper.dataset.wrap !== undefined)
  );
}

// Turn a block into another type, keeping its content and caret.
function convert(unit: HTMLElement, kind: Kind, lang = ""): HTMLElement {
  // Pressing the same heading shortcut again goes back to a paragraph.
  if (kind === unit.tagName) kind = "P";
  let target: HTMLElement;
  let outer: HTMLElement;
  if (kind === "UL" || kind === "OL" || kind === "QUOTE") {
    outer = document.createElement(kind === "QUOTE" ? "blockquote" : kind);
    outer.dataset.wrap = "";
    target = makeUnit(kind === "QUOTE" ? "P" : "LI", unit);
    outer.append(target);
  } else if (kind === "CODE") {
    target = outer = makeUnit("PRE", unit);
    target.dataset.lang = lang;
    target.className = "edit-code";
  } else {
    target = outer = makeUnit(kind, unit);
  }
  if (kind === "CODE") {
    target.textContent = (unit.textContent ?? "").replace(/​/g, "") || "\n";
  } else {
    target.append(
      ...[...unit.childNodes].filter(
        n => !(n instanceof HTMLElement && n.classList.contains("heading-link"))
      )
    );
  }
  if (!target.textContent) target.append(ZWSP);
  const wrapper = unit.parentElement;
  const replaced =
    wrapper?.dataset.wrap !== undefined && wrapper.children.length === 1
      ? wrapper
      : unit;
  replaced.replaceWith(outer);
  placeCaret(target, kind !== "CODE");
  return target;
}

// "# ", "## ", "- ", "1. " or "> " typed at the start of a paragraph turns it
// into a heading, list item or quote.
function applyBlockShortcut(): boolean {
  const unit = caretUnit();
  if (
    !unit ||
    unit.tagName !== "P" ||
    !convertible(unit) ||
    unit.parentElement?.tagName === "BLOCKQUOTE"
  ) {
    return false;
  }
  const first = unit.firstChild;
  if (!(first instanceof Text)) return false;
  const m = first.data.match(/^(#{1,6}|[-*+]|\d+\.|>)[\s ]/);
  if (!m) return false;
  first.data = first.data.slice(m[0].length);
  const kind: Kind = m[1].startsWith("#")
    ? (`H${m[1].length}` as Kind)
    : m[1] === ">"
      ? "QUOTE"
      : /\d/.test(m[1])
        ? "OL"
        : "UL";
  const target = convert(unit, kind);
  placeCaret(target);
  return true;
}

function onInput(): void {
  if (!applyBlockShortcut()) applyInlineShortcut();
}

// Wrap the selection in <mark>, or unwrap it when already highlighted.
function toggleHighlight(): void {
  const sel = getSelection();
  if (!sel || sel.rangeCount === 0) return;
  const range = sel.getRangeAt(0);
  const node = range.commonAncestorContainer;
  const existing = (
    node instanceof Element ? node : node.parentElement
  )?.closest("mark");
  if (existing) {
    existing.replaceWith(...existing.childNodes);
    return;
  }
  if (range.collapsed) return;
  const mark = document.createElement("mark");
  mark.append(range.extractContents());
  range.insertNode(mark);
  sel.selectAllChildren(mark);
}

// Whether the caret sits on the first or last visual line of a block.
function onEdgeLine(unit: HTMLElement, edge: "top" | "bottom"): boolean {
  const sel = getSelection();
  if (!sel?.rangeCount) return false;
  const caret = sel.getRangeAt(0).cloneRange();
  caret.collapse(edge === "top");
  const rect = caret.getClientRects()[0];
  const box = unit.getBoundingClientRect();
  if (!rect) return true;
  const line = parseFloat(getComputedStyle(unit).lineHeight) || 24;
  return edge === "top"
    ? rect.top - box.top < line * 0.8
    : box.bottom - rect.bottom < line * 0.8;
}

function atBoundary(unit: HTMLElement, end: boolean): boolean {
  const sel = getSelection();
  if (!sel?.rangeCount || !sel.isCollapsed) return false;
  const probe = document.createRange();
  probe.selectNodeContents(unit);
  const caret = sel.getRangeAt(0);
  if (end) probe.setStart(caret.endContainer, caret.endOffset);
  else probe.setEnd(caret.startContainer, caret.startOffset);
  return probe.toString().replace(/​/g, "") === "";
}

// Each block is its own editing host, so arrow keys must hop between them.
function moveAcross(e: KeyboardEvent, unit: HTMLElement): boolean {
  const back = e.key === "ArrowUp" || e.key === "ArrowLeft";
  const vertical = e.key === "ArrowUp" || e.key === "ArrowDown";
  const leaving = vertical
    ? onEdgeLine(unit, back ? "top" : "bottom")
    : atBoundary(unit, !back);
  if (!leaving || e.shiftKey) return false;
  const all = units();
  const next = all[all.indexOf(unit) + (back ? -1 : 1)];
  if (!next) return false;
  e.preventDefault();
  next.focus();
  placeCaret(next, back);
  return true;
}

const HEADING_KEYS: Record<string, Kind> = {
  "0": "P",
  "1": "H1",
  "2": "H2",
  "3": "H3",
  "4": "H4",
  "5": "H5",
  "6": "H6",
};

// Typora-style shortcuts (Cmd on macOS, Ctrl elsewhere).
function onShortcut(e: KeyboardEvent, unit: HTMLElement): boolean {
  if (!(e.metaKey || e.ctrlKey)) return false;
  const key = e.key.toLowerCase();
  let kind: Kind | undefined;
  if (!e.shiftKey && !e.altKey && HEADING_KEYS[e.key])
    kind = HEADING_KEYS[e.key];
  else if (e.shiftKey && e.code === "BracketRight") kind = "UL";
  else if (e.shiftKey && e.code === "BracketLeft") kind = "OL";
  else if (e.shiftKey && key === "q") kind = "QUOTE";
  else if (e.altKey && e.code === "KeyC") kind = "CODE";
  else if (e.shiftKey && key === "h") {
    e.preventDefault();
    toggleHighlight();
    return true;
  } else if (e.shiftKey && e.code === "Backquote") {
    e.preventDefault();
    const sel = getSelection();
    if (sel && !sel.isCollapsed) {
      const range = sel.getRangeAt(0);
      const code = document.createElement("code");
      code.textContent = range.toString();
      range.deleteContents();
      range.insertNode(code);
    }
    return true;
  }
  if (!kind) return false;
  e.preventDefault();
  if (convertible(unit)) convert(unit, kind);
  return true;
}

// The article-level element a block belongs to (a list, table, quote, ...).
function topLevel(el: HTMLElement): Element {
  let node: Element = el;
  while (node.parentElement && node.parentElement.id !== "article") {
    node = node.parentElement;
  }
  return node;
}

// Blocks can merge only when nothing un-editable (code, diagram) sits between.
function adjacent(prev: HTMLElement, unit: HTMLElement): boolean {
  const a = topLevel(prev);
  const b = topLevel(unit);
  if (a === b) return true;
  let between = a.nextElementSibling;
  while (between && between !== b) {
    if (!(between instanceof HTMLElement && between.hidden)) return false;
    between = between.nextElementSibling;
  }
  return between === b;
}

// Text of a block from its start up to the caret.
function caretText(unit: HTMLElement): string {
  const sel = getSelection();
  if (!sel?.rangeCount) return "";
  const range = document.createRange();
  range.selectNodeContents(unit);
  range.setEnd(sel.getRangeAt(0).startContainer, sel.getRangeAt(0).startOffset);
  return range.toString().replace(/\u200b/g, "");
}

// Insert plain text at the caret. A code block always keeps a trailing
// newline so the caret can sit on its last line.
function insertText(text: string): void {
  const sel = getSelection();
  if (!sel?.rangeCount) return;
  const range = sel.getRangeAt(0);
  range.deleteContents();
  const node = document.createTextNode(text);
  range.insertNode(node);
  const pre = node.parentElement?.closest("pre");
  if (!pre) {
    range.setStartAfter(node);
    range.collapse(true);
    sel.removeAllRanges();
    sel.addRange(range);
    return;
  }
  // Collapse the code into one text node and place the caret by offset, so
  // the browser can't slip the next keystroke in front of the newline.
  const offset = caretTextUpTo(pre, node) + text.length;
  pre.textContent = (pre.textContent ?? "").replace(/\u200b/g, "");
  // A final newline is invisible, so a caret on the last line needs one more.
  if (offset >= pre.textContent.length || !pre.textContent.endsWith("\n")) {
    pre.textContent += "\n";
  }
  sel.collapse(pre.firstChild, offset);
}

function caretTextUpTo(root: HTMLElement, node: Node): number {
  const range = document.createRange();
  range.selectNodeContents(root);
  range.setEndBefore(node);
  return range.toString().replace(/\u200b/g, "").length;
}

function onKeyDown(e: KeyboardEvent): void {
  const unit = caretUnit();
  if (!unit) return;
  if (onShortcut(e, unit)) return;
  if (e.key.startsWith("Arrow") && moveAcross(e, unit)) return;
  const sel = getSelection()!;
  const text = (unit.textContent ?? "").replace(/​/g, "");

  if (unit.tagName === "PRE") {
    if (e.key === "Enter" && !e.isComposing) {
      e.preventDefault();
      const before = caretText(unit);
      const code = (unit.textContent ?? "").replace(/\u200b/g, "");
      // Enter on an empty last line leaves the code block.
      if (before.endsWith("\n") && code.slice(before.length).trim() === "") {
        unit.textContent = before.replace(/\n+$/, "") + "\n";
        const p = makeUnit("P");
        p.append(ZWSP);
        unit.after(p);
        placeCaret(p);
        return;
      }
      insertText("\n");
    } else if (e.key === "Tab") {
      e.preventDefault();
      insertText("  ");
    }
    return;
  }

  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    if (unit.tagName === "TD" || unit.tagName === "TH") return;
    // "```lang" or ">" on its own line, then Enter, starts a code block or quote.
    const fence = text.match(/^```(\S*)$/);
    if (convertible(unit) && fence) {
      const code = convert(unit, "CODE", fence[1]);
      code.textContent = "\n";
      placeCaret(code);
      return;
    }
    if (convertible(unit) && text === ">") {
      unit.textContent = "";
      convert(unit, "QUOTE");
      return;
    }
    // Enter on an empty new list item leaves the list.
    if (unit.tagName === "LI" && unit.dataset.new !== undefined && !text) {
      const list = unit.parentElement!;
      const p = makeUnit("P");
      p.append(ZWSP);
      list.after(p);
      unit.remove();
      if (!list.children.length) list.remove();
      placeCaret(p);
      return;
    }
    // Move whatever follows the caret into a new block after this one.
    const range = sel.getRangeAt(0);
    // Leave a heading's "#" anchor link where it is.
    const anchor = unit.querySelector(":scope > .heading-link");
    if (anchor) range.setEndBefore(anchor);
    else range.setEndAfter(unit.lastChild ?? unit);
    const tail = range.extractContents();
    const isItem =
      unit.tagName === "LI" || unit.parentElement?.tagName === "LI";
    const next = makeUnit(isItem ? "LI" : "P");
    next.append(tail);
    if (!next.textContent) next.append(ZWSP);
    const after = isItem
      ? unit.closest("li")!
      : unit.parentElement?.dataset.wrap !== undefined &&
          unit.parentElement.tagName === "BLOCKQUOTE"
        ? unit
        : unit;
    after.after(next);
    placeCaret(next);
    return;
  }

  // Backspace at the start of a block merges it into the block above.
  if (e.key === "Backspace" && sel.isCollapsed && atBoundary(unit, false)) {
    const all = units();
    const prev = all[all.indexOf(unit) - 1];
    if (!prev || !adjacent(prev, unit) || prev.tagName === "PRE") return;
    e.preventDefault();
    const list = unit.tagName === "LI" ? unit.parentElement : null;
    const joinAt = document.createRange();
    joinAt.selectNodeContents(prev);
    joinAt.collapse(false);
    const anchor = prev.querySelector(":scope > .heading-link");
    if (anchor) joinAt.setStartBefore(anchor);
    // Drop placeholder characters before joining.
    prev.childNodes.forEach(n => {
      if (n instanceof Text) n.data = n.data.replace(/\u200b/g, "");
    });
    const moved = [...unit.childNodes].filter(
      n => !(n instanceof HTMLElement && n.classList.contains("heading-link"))
    );
    const marker = document.createTextNode("");
    joinAt.insertNode(marker);
    marker.after(...moved);
    if (unit.dataset.mdS !== undefined) {
      // Keep a hidden placeholder so the save removes this block's source.
      const tomb = document.createElement("span");
      tomb.hidden = true;
      tomb.dataset.mdS = unit.dataset.mdS;
      tomb.dataset.mdE = unit.dataset.mdE;
      tomb.dataset.deleted = "";
      unit.replaceWith(tomb);
    } else {
      unit.remove();
    }
    if (list && !list.querySelector("li")) list.hidden = true;
    const caret = document.createRange();
    caret.setStartAfter(marker);
    caret.collapse(true);
    sel.removeAllRanges();
    sel.addRange(caret);
  }
}

// Place the button midway between the article and the Content sidebar,
// halfway down the screen; fall back to the corner when there is no sidebar.
function placeButton(btn: HTMLElement): void {
  const article = document.querySelector("#article");
  const toc = document.querySelector("#toc-aside .toc-aside-nav");
  const tocBox = toc?.getBoundingClientRect();
  if (!article || !tocBox || tocBox.width === 0) {
    btn.style.removeProperty("left");
    btn.style.removeProperty("top");
    return;
  }
  const gapStart = article.getBoundingClientRect().right;
  btn.style.left = `${(gapStart + tocBox.left) / 2 - btn.offsetWidth / 2}px`;
  btn.style.top = `calc(50% - ${btn.offsetHeight / 2}px)`;
}

const sleep = (ms: number) => new Promise(r => setTimeout(r, ms));

function setupEdit(): void {
  const btn = document.querySelector<HTMLButtonElement>("#edit-btn");
  const article = document.querySelector<HTMLElement>("#article");
  const path = article?.dataset.path;
  if (!btn || !article || !path) return;

  // Visitors never see the button: it shows only once this browser holds an
  // encrypted token, or when the page is opened with ?edit to set one up.
  const owner = hasVault() || new URLSearchParams(location.search).has("edit");
  if (!owner) return;
  btn.hidden = false;

  placeButton(btn);
  window.addEventListener("resize", () => placeButton(btn));
  article.addEventListener("input", onInput);
  article.addEventListener("keydown", onKeyDown);

  const setMode = (mode: string, label: string) => {
    btn.dataset.mode = mode;
    btn.setAttribute("aria-label", label);
    btn.title = label;
  };

  const setEditing = (editing: boolean) => {
    article.classList.toggle("editing", editing);
    units().forEach(el => {
      if (editing) el.contentEditable = "true";
      else el.removeAttribute("contenteditable");
    });
  };

  // Show a result icon for two seconds, then return to `mode`.
  const flash = async (result: string, label: string, mode: string) => {
    setMode(result, label);
    await sleep(2000);
    setMode(mode, mode === "editing" ? "Save and publish" : "Edit page");
  };

  btn.addEventListener("click", async () => {
    const mode = btn.dataset.mode;
    if (mode === "idle") {
      before = new Map(units().map(el => [el, unitMarkdown(el)]));
      setEditing(true);
      setMode("editing", "Save and publish");
      return;
    }
    if (mode !== "editing") return;
    setMode("saving", "Saving…");
    const started = Date.now();
    try {
      const result = await save(path);
      // Let the ring finish at least one turn.
      await sleep(Math.max(0, 1000 - (Date.now() - started)));
      if (result === "cancelled") {
        setMode("editing", "Save and publish");
        return;
      }
      setEditing(false);
      if (result === "unchanged") {
        await flash("unchanged", "No changes", "idle");
      } else {
        toast("Saved and pushed. The live site updates in about 1–2 minutes.");
        await flash("saved", "Saved", "idle");
      }
    } catch (error) {
      toast((error as Error).message);
      await flash("failed", "Save failed", "editing");
    }
  });

  // Keep links from navigating away while editing.
  article.addEventListener("click", e => {
    if (
      article.classList.contains("editing") &&
      (e.target as HTMLElement).closest("a")
    ) {
      e.preventDefault();
    }
  });
}

document.addEventListener("astro:page-load", setupEdit);
// Wipe the unlocked token when the page is hidden for good.
window.addEventListener("pagehide", lock);

export {};
