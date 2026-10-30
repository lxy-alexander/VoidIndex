/**
 * In-place editing for posts. The floating button switches the article into
 * edit mode (headings, paragraphs and table cells become editable); pressing
 * it again converts the changed blocks back to Markdown, splices them into
 * the .md file at the offsets from utils/remarkSourceMap.ts, and commits the
 * file to GitHub, which redeploys the site.
 */
import { openReview } from "./review";
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
let before = new Map<string, string>();
// Source range of each existing block when editing (re)started, by uid.
let original = new Map<string, { s: number; e: number; cell: boolean }>();

function remember(): void {
  before = new Map(units().map(el => [uid(el), unitMarkdown(el)]));
  original = new Map(
    units()
      .filter(
        el => el.dataset.mdS !== undefined && el.dataset.new === undefined
      )
      .map(el => [
        uid(el),
        {
          s: Number(el.dataset.mdS),
          e: Number(el.dataset.mdE),
          cell: el.tagName === "TD" || el.tagName === "TH",
        },
      ])
  );
}

// A stable id per block; it survives undo, which rebuilds the article HTML.
let nextUid = 0;
function uid(el: HTMLElement): string {
  if (!el.dataset.uid) el.dataset.uid = String(++nextUid);
  return el.dataset.uid;
}

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
    case "S":
    case "STRIKE":
    case "DEL":
      return `~~${inner()}~~`;
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

// A short notice in the middle of the screen; click to dismiss early.
type Tone = "success" | "error" | "info";
const TONE_ICON: Record<Tone, string> = {
  success: '<path d="M5 12l5 5l10 -10" />',
  error: '<path d="M12 8v4" /><path d="M12 16h.01" />',
  info: '<path d="M12 8h.01" /><path d="M11 12h1v4h1" />',
};

function toast(message: string, tone: Tone = "info", detail = ""): void {
  document.querySelector(".edit-toast")?.remove();
  const el = document.createElement("div");
  el.className = `edit-toast edit-toast-${tone}`;
  el.setAttribute("role", tone === "error" ? "alert" : "status");
  el.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">${TONE_ICON[tone]}</svg><div><p class="edit-toast-title"></p><p class="edit-toast-detail"></p></div>`;
  el.querySelector(".edit-toast-title")!.textContent = message;
  const sub = el.querySelector<HTMLElement>(".edit-toast-detail")!;
  if (detail) sub.textContent = detail;
  else sub.remove();
  const dismiss = () => {
    el.classList.add("leaving");
    setTimeout(() => el.remove(), 200);
  };
  el.addEventListener("click", dismiss);
  document.body.append(el);
  setTimeout(dismiss, tone === "error" ? 4000 : 2200);
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

  // New blocks typed above an existing block, with something un-editable
  // (code, a diagram) between them and the block before: they go right in
  // front of the following block instead.
  let pending: { el: HTMLElement; md: string }[] = [];
  const insertBefore = (el: HTMLElement) => {
    if (!pending.length) return;
    const at =
      raw.lastIndexOf("\n", bodyStart + Number(el.dataset.mdS) - 1) +
      1 -
      bodyStart;
    const insert: Edit = { s: at, e: at, text: "", pieces: [] };
    for (const { el: block, md } of pending) {
      const prefix = blockPrefix(block, "");
      insert.pieces.push({
        el: block,
        start: insert.text.length + prefix.length,
        length: md.length,
      });
      insert.text += prefix + md + "\n\n";
    }
    edits.push(insert);
    pending = [];
  };

  for (const el of units(true)) {
    // A block merged into the one above: drop its source up to its end.
    if (el.dataset.deleted !== undefined) {
      if (current) current.e = Number(el.dataset.mdE);
      continue;
    }
    if (el.dataset.new === undefined) {
      flush();
      insertBefore(el);
      const converted = el.dataset.was !== undefined;
      let s = Number(el.dataset.mdS);
      const e = Number(el.dataset.mdE);
      // A heading's range starts after its "## "; widen it to the line start
      // when the block type changes so the old marker is replaced.
      if (converted && /^H[1-6]$/.test(el.dataset.was ?? "")) {
        s = raw.lastIndexOf("\n", bodyStart + s - 1) + 1 - bodyStart;
      }
      const inline = unitMarkdown(el);
      const changed = converted || inline !== before.get(uid(el));
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
    // An empty new line has nothing to save.
    if (!inline || !current) continue;
    el.dataset.dirty = "";
    if (pending.length || (prev && !adjacent(prev, el))) {
      pending.push({ el, md: inline });
      continue;
    }
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
  // Nothing follows: append what is left after the last block.
  for (const { el, md } of pending) {
    if (!current) break;
    const prefix = blockPrefix(el, "");
    current.text += "\n\n" + prefix;
    current.pieces.push({ el, start: current.text.length, length: md.length });
    current.text += md;
    el.dataset.dirty = "";
  }
  flush();

  // Blocks removed by deleting a selection that spans several of them: drop
  // their source lines (marker included) and the blank line after.
  for (const [id, range] of original) {
    if (document.querySelector(`#article [data-uid="${id}"]`)) continue;
    // A table cell's source is part of a row; leave the table intact.
    if (range.cell) continue;
    const s = raw.lastIndexOf("\n", bodyStart + range.s - 1) + 1 - bodyStart;
    let e = range.e;
    while (body[e] === "\n") e++;
    edits.push({ s, e, text: "", pieces: [] });
  }
  return edits;
}

type Result = "saved" | "unchanged" | "cancelled";

// Unpublished edits, per post: the edited .md file, the edited article HTML
// (so a reload shows the draft), and the hash of the version it started from.
const DRAFTS_KEY = "edit-drafts";
type Draft = {
  raw: string;
  html: string;
  hash: string;
  baseHash: string;
  title: string;
};

function readDrafts(): Record<string, Draft> {
  try {
    return JSON.parse(localStorage.getItem(DRAFTS_KEY) ?? "{}");
  } catch {
    return {};
  }
}

function writeDrafts(drafts: Record<string, Draft>): void {
  if (Object.keys(drafts).length) {
    localStorage.setItem(DRAFTS_KEY, JSON.stringify(drafts));
  } else {
    localStorage.removeItem(DRAFTS_KEY);
  }
}

function splitBody(raw: string) {
  // Offsets are relative to the trimmed body after the frontmatter.
  const fm = raw.match(/^---\r?\n[\s\S]*?\r?\n---/)?.[0] ?? "";
  const rest = raw.slice(fm.length);
  const bodyStart = fm.length + (rest.length - rest.trimStart().length);
  return { fm, bodyStart, body: rest.trim() };
}

// Hash of the version this page was built from, before any draft is applied.
let deployedHash = "";

async function githubHeaders(): Promise<Record<string, string> | null> {
  const token = await getToken();
  if (!token) return null;
  return {
    Authorization: `Bearer ${token}`,
    Accept: "application/vnd.github+json",
  };
}

async function api(
  path: string,
  headers: Record<string, string>,
  init: RequestInit = {}
): Promise<unknown> {
  const res = await fetch(`https://api.github.com/repos/${REPO}/${path}`, {
    ...init,
    headers,
  });
  if (res.status === 401) forget();
  if (!res.ok) throw new Error(`GitHub request failed (${res.status})`);
  return res.json();
}

// The note's current source on GitHub. Public repo, so no token is needed;
// fall back to the token if the anonymous rate limit is hit.
async function fetchSource(path: string): Promise<string> {
  const url = `https://api.github.com/repos/${REPO}/contents/${path}?ref=${BRANCH}`;
  let res = await fetch(url, {
    headers: { Accept: "application/vnd.github+json" },
  });
  if (res.status === 403 || res.status === 429) {
    const headers = await githubHeaders();
    if (!headers)
      throw new Error("GitHub rate limit reached; unlock to continue.");
    res = await fetch(url, { headers });
  }
  if (!res.ok) throw new Error(`Reading the file failed (${res.status})`);
  return decodeBase64(((await res.json()) as { content: string }).content);
}

// Apply this page's edits to the note and keep the result as a local draft.
async function saveDraft(path: string): Promise<Result> {
  const dirty =
    removedBlocks() ||
    document.querySelector("#article [data-deleted]") !== null ||
    units().some(
      el =>
        el.dataset.new !== undefined ||
        el.dataset.was !== undefined ||
        unitMarkdown(el) !== before.get(uid(el))
    );
  if (!dirty) return "unchanged";

  const drafts = readDrafts();
  const hashEl = document.querySelector<HTMLElement>("#md-source")!;
  const raw = drafts[path]?.raw ?? (await fetchSource(path));
  const { fm, bodyStart, body } = splitBody(raw);
  if ((await sha1(body)) !== hashEl.dataset.hash) {
    throw new Error(
      "Note changed on GitHub. Reload after the deploy, then edit again."
    );
  }

  const edits = buildEdits(body, raw, bodyStart);
  let next = raw;
  for (const { s, e, text } of [...edits].sort(
    (a, b) => b.s - a.s || b.e - a.e
  )) {
    next = next.slice(0, bodyStart + s) + text + next.slice(bodyStart + e);
  }

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
    if (placed.has(el) || el.querySelector("[data-md-s]")) return;
    const shift = shiftBefore(Number(el.dataset.mdS));
    el.dataset.mdS = String(Number(el.dataset.mdS) + shift);
    el.dataset.mdE = String(Number(el.dataset.mdE) + shift);
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

  const article = document.querySelector<HTMLElement>("#article")!;
  const snapshot = article.cloneNode(true) as HTMLElement;
  snapshot
    .querySelectorAll("[contenteditable]")
    .forEach(el => el.removeAttribute("contenteditable"));
  drafts[path] = {
    raw: next,
    html: snapshot.innerHTML,
    hash: hashEl.dataset.hash,
    baseHash: drafts[path]?.baseHash ?? deployedHash,
    title:
      document
        .querySelector("main h1")
        ?.textContent?.replace(/\s+/g, " ")
        .trim() ?? path,
  };
  writeDrafts(drafts);
  return "saved";
}

// Show a saved draft after a reload, or drop it once it has been deployed.
function restoreDraft(path: string, article: HTMLElement): void {
  const drafts = readDrafts();
  const draft = drafts[path];
  if (!draft) return;
  if (draft.hash === deployedHash) {
    delete drafts[path];
    writeDrafts(drafts);
  } else if (draft.baseHash === deployedHash) {
    article.innerHTML = draft.html;
  } else {
    toast(
      "Draft is out of date",
      "error",
      "This note changed on GitHub after your draft."
    );
  }
}

// Commit every draft in one commit through the Git Data API.
async function publishDrafts(
  texts: Record<string, string> = {}
): Promise<Result> {
  const drafts = readDrafts();
  // Edits made in the review panel replace the stored draft text.
  for (const [path, text] of Object.entries(texts)) {
    if (drafts[path]) drafts[path].raw = text;
  }
  const paths = Object.keys(drafts);
  if (!paths.length) return "unchanged";
  const headers = await githubHeaders();
  if (!headers) return "cancelled";

  const ref = (await api(`git/ref/heads/${BRANCH}`, headers)) as {
    object: { sha: string };
  };
  const head = ref.object.sha;
  const commit = (await api(`git/commits/${head}`, headers)) as {
    tree: { sha: string };
  };

  // Refuse if a note changed on GitHub since its draft was started.
  for (const path of paths) {
    const file = (await api(`contents/${path}?ref=${head}`, headers)) as {
      content: string;
    };
    const { body } = splitBody(decodeBase64(file.content));
    if ((await sha1(body)) !== drafts[path].baseHash) {
      throw new Error(
        `"${drafts[path].title}" changed on GitHub. Nothing was published.`
      );
    }
  }

  const tree = (await api("git/trees", headers, {
    method: "POST",
    body: JSON.stringify({
      base_tree: commit.tree.sha,
      tree: paths.map(path => ({
        path,
        mode: "100644",
        type: "blob",
        content: drafts[path].raw,
      })),
    }),
  })) as { sha: string };
  const message =
    paths.length === 1
      ? `Edit: ${drafts[paths[0]].title}`
      : `Edit ${paths.length} notes\n\n${paths.map(p => `- ${drafts[p].title}`).join("\n")}`;
  const created = (await api("git/commits", headers, {
    method: "POST",
    body: JSON.stringify({ message, tree: tree.sha, parents: [head] }),
  })) as { sha: string };
  await api(`git/refs/heads/${BRANCH}`, headers, {
    method: "PATCH",
    body: JSON.stringify({ sha: created.sha, force: false }),
  });
  writeDrafts({});
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
  const unit = el?.closest<HTMLElement>("[data-md-s], [data-new]");
  return unit && units().includes(unit) ? unit : null;
}

// Focus the editing host that contains a block (the article, or a table cell).
function focusHost(el: HTMLElement): void {
  let host = el;
  while (host.parentElement?.isContentEditable) host = host.parentElement;
  host.focus({ preventScroll: true });
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
  // The article is one editing area, so the browser moves the caret itself.
  if (unit.parentElement?.isContentEditable) return false;
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
  focusHost(next);
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

// Undo/redo. Most edits here are made by script, which the browser's own
// undo can't track, so keep snapshots of the article and the caret.
type Snapshot = { html: string; caret: { uid: string; offset: number } | null };
let undoStack: Snapshot[] = [];
let redoStack: Snapshot[] = [];
let lastTyping = 0;

function resetHistory(): void {
  undoStack = [];
  redoStack = [];
  lastTyping = 0;
}

function snapshot(): Snapshot {
  const article = document.querySelector<HTMLElement>("#article")!;
  units().forEach(uid);
  const unit = caretUnit();
  let caret: Snapshot["caret"] = null;
  const sel = getSelection();
  if (unit && sel?.rangeCount) {
    const range = document.createRange();
    range.selectNodeContents(unit);
    range.setEnd(
      sel.getRangeAt(0).startContainer,
      sel.getRangeAt(0).startOffset
    );
    caret = { uid: uid(unit), offset: range.toString().length };
  }
  return { html: article.innerHTML, caret };
}

// Save the state before a change. Typing within a second counts as one step.
function record(typing = false): void {
  const now = Date.now();
  if (typing && now - lastTyping < 1000) {
    lastTyping = now;
    return;
  }
  lastTyping = typing ? now : 0;
  undoStack.push(snapshot());
  if (undoStack.length > 200) undoStack.shift();
  redoStack = [];
}

function restore(state: Snapshot): void {
  const article = document.querySelector<HTMLElement>("#article")!;
  article.innerHTML = state.html;
  lastTyping = 0;
  if (!state.caret) return;
  const unit = article.querySelector<HTMLElement>(
    `[data-uid="${state.caret.uid}"]`
  );
  if (!unit) return;
  focusHost(unit);
  // Walk the text to the saved character offset.
  let left = state.caret.offset;
  const walker = document.createTreeWalker(unit, NodeFilter.SHOW_TEXT);
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    const length = (node as Text).length;
    if (left <= length) {
      getSelection()?.collapse(node, left);
      return;
    }
    left -= length;
  }
  placeCaret(unit, true);
}

function undo(): void {
  const state = undoStack.pop();
  if (!state) return;
  redoStack.push(snapshot());
  restore(state);
}

function redo(): void {
  const state = redoStack.pop();
  if (!state) return;
  undoStack.push(snapshot());
  restore(state);
}

function onBeforeInput(e: InputEvent): void {
  if (e.inputType === "historyUndo" || e.inputType === "historyRedo") {
    e.preventDefault();
    if (e.inputType === "historyUndo") undo();
    else redo();
    return;
  }
  record(
    e.inputType.startsWith("insertText") ||
      e.inputType === "insertCompositionText" ||
      e.inputType.startsWith("delete")
  );
}

function onKeyDown(e: KeyboardEvent): void {
  const mod = e.metaKey || e.ctrlKey;
  // Snapshot before the structural edits handled below.
  if (
    e.key === "Enter" ||
    e.key === "Tab" ||
    (mod &&
      !["c", "a", "x", "v"].includes(e.key.toLowerCase()) &&
      !e.key.startsWith("Arrow"))
  ) {
    record();
  }
  if (e.key === "Backspace" || e.key === "Delete") record(true);
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
    // Enter at the very start of a block opens an empty line above it and
    // leaves the block (e.g. a heading) as it is.
    if (text && atBoundary(unit, false)) {
      const isItem =
        unit.tagName === "LI" || unit.parentElement?.tagName === "LI";
      const above = makeUnit(isItem ? "LI" : "P");
      above.append(ZWSP);
      (isItem ? unit.closest("li")! : unit).before(above);
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
    // Table cells keep their own content; never join them.
    if (unit.closest("td, th") || prev.closest("td, th")) return;
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
      if (unit.dataset.uid) tomb.dataset.uid = unit.dataset.uid;
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
function placeButton(btn: HTMLElement, offsetY = 0): void {
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
  btn.style.top = `calc(50% - ${btn.offsetHeight / 2 - offsetY}px)`;
}

// Whether the article has edits that are not in a saved draft yet.
function removedBlocks(): boolean {
  return [...original.keys()].some(
    id => !document.querySelector(`#article [data-uid="${id}"]`)
  );
}

function hasUnsaved(): boolean {
  return (
    removedBlocks() ||
    document.querySelector("#article [data-deleted]") !== null ||
    units().some(
      el =>
        el.dataset.new !== undefined ||
        el.dataset.was !== undefined ||
        unitMarkdown(el) !== before.get(uid(el))
    )
  );
}

// Formatting toolbar that floats above a text selection while editing.
const FORMATS: { cmd: string; label: string; icon: string }[] = [
  {
    cmd: "bold",
    label: "Bold (⌘B)",
    icon: '<path d="M7 5h6a3.5 3.5 0 0 1 0 7h-6z" /><path d="M13 12h1a3.5 3.5 0 0 1 0 7h-7v-7" />',
  },
  {
    cmd: "italic",
    label: "Italic (⌘I)",
    icon: '<path d="M11 5l6 0" /><path d="M7 19l6 0" /><path d="M14 5l-4 14" />',
  },
  {
    cmd: "strike",
    label: "Strikethrough",
    icon: '<path d="M5 12l14 0" /><path d="M16 6.5a4 2 0 0 0 -4 -1.5h-1a3.5 3.5 0 0 0 0 7h2a3.5 3.5 0 0 1 0 7h-1.5a4 2 0 0 1 -4 -1.5" />',
  },
  {
    cmd: "mark",
    label: "Highlight (⇧⌘H)",
    icon: '<path d="M3 19h4l10.5 -10.5a2.828 2.828 0 1 0 -4 -4l-10.5 10.5v4" /><path d="M12.5 5.5l4 4" />',
  },
  {
    cmd: "code",
    label: "Inline code (⇧⌘`)",
    icon: '<path d="M7 8l-4 4l4 4" /><path d="M17 8l4 4l-4 4" /><path d="M14 4l-4 16" />',
  },
  {
    cmd: "link",
    label: "Link",
    icon: '<path d="M9 15l6 -6" /><path d="M11 6l.463 -.536a5 5 0 0 1 7.071 7.072l-.534 .464" /><path d="M13 18l-.397 .534a5.068 5.068 0 0 1 -7.127 0a4.972 4.972 0 0 1 0 -7.071l.524 -.463" />',
  },
  {
    cmd: "ul",
    label: "Bullet list (⇧⌘])",
    icon: '<path d="M9 6l11 0" /><path d="M9 12l11 0" /><path d="M9 18l11 0" /><path d="M5 6l0 .01" /><path d="M5 12l0 .01" /><path d="M5 18l0 .01" />',
  },
  {
    cmd: "ol",
    label: "Numbered list (⇧⌘[)",
    icon: '<path d="M11 6h9" /><path d="M11 12h9" /><path d="M12 18h8" /><path d="M4 16a2 2 0 1 1 4 0c0 .591 -.5 1 -1 1.5l-3 2.5h4" /><path d="M6 10v-6l-2 2" />',
  },
  {
    cmd: "quote",
    label: "Quote (⇧⌘Q)",
    icon: '<path d="M10 11h-4a1 1 0 0 1 -1 -1v-3a1 1 0 0 1 1 -1h3a1 1 0 0 1 1 1v6c0 2.667 -1.333 4.333 -4 5" /><path d="M19 11h-4a1 1 0 0 1 -1 -1v-3a1 1 0 0 1 1 -1h3a1 1 0 0 1 1 1v6c0 2.667 -1.333 4.333 -4 5" />',
  },
];

function applyFormat(cmd: string): void {
  record();
  const sel = getSelection();
  if (cmd === "bold" || cmd === "italic") document.execCommand(cmd);
  else if (cmd === "strike") document.execCommand("strikeThrough");
  else if (cmd === "mark") toggleHighlight();
  else if (cmd === "link") {
    const url = prompt("Link URL")?.trim();
    if (url) document.execCommand("createLink", false, url);
  } else if (cmd === "code" && sel && !sel.isCollapsed) {
    const range = sel.getRangeAt(0);
    const code = document.createElement("code");
    code.textContent = range.toString();
    range.deleteContents();
    range.insertNode(code);
  } else {
    const unit = caretUnit();
    const kind = cmd === "ul" ? "UL" : cmd === "ol" ? "OL" : "QUOTE";
    if (unit && convertible(unit)) convert(unit, kind);
  }
}

function createFormatBar(article: HTMLElement): HTMLElement {
  const bar = document.createElement("div");
  bar.className = "format-bar";
  bar.hidden = true;
  bar.setAttribute("role", "toolbar");
  for (const { cmd, label, icon } of FORMATS) {
    const button = document.createElement("button");
    button.type = "button";
    button.title = label;
    button.setAttribute("aria-label", label);
    button.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${icon}</svg>`;
    // Keep the selection when clicking.
    button.addEventListener("mousedown", e => e.preventDefault());
    button.addEventListener("click", () => {
      applyFormat(cmd);
      bar.hidden = true;
    });
    if (cmd === "link" || cmd === "ul")
      bar.append(
        Object.assign(document.createElement("span"), {
          className: "format-sep",
        })
      );
    bar.append(button);
  }
  document.body.append(bar);

  document.addEventListener("selectionchange", () => {
    const sel = getSelection();
    const range = sel?.rangeCount ? sel.getRangeAt(0) : null;
    if (
      !article.classList.contains("editing") ||
      !range ||
      range.collapsed ||
      !article.contains(range.commonAncestorContainer) ||
      document.querySelector(".review-overlay")
    ) {
      bar.hidden = true;
      return;
    }
    const rect = range.getBoundingClientRect();
    bar.hidden = false;
    const width = bar.offsetWidth;
    const top = rect.top - bar.offsetHeight - 10;
    bar.style.top = `${top < 8 ? rect.bottom + 10 : top}px`;
    bar.style.left = `${Math.min(
      Math.max(8, rect.left + rect.width / 2 - width / 2),
      innerWidth - width - 8
    )}px`;
  });
  return bar;
}

const sleep = (ms: number) => new Promise(r => setTimeout(r, ms));

// Whether to show the editing buttons in this browser. Opening any page
// once with ?edit remembers the browser as the owner's; ?edit=off forgets
// it. Local previews always show them. Visitors never see them, and nothing
// can be published without the encrypted token anyway.
const OWNER_KEY = "voidindex-owner";

function isOwner(): boolean {
  const param = new URLSearchParams(location.search).get("edit");
  try {
    if (param === "off") localStorage.removeItem(OWNER_KEY);
    else if (param !== null) localStorage.setItem(OWNER_KEY, "1");
    if (localStorage.getItem(OWNER_KEY)) return true;
  } catch {
    // Storage unavailable: fall through to the other checks.
  }
  return (
    (param !== null && param !== "off") ||
    hasVault() ||
    ["localhost", "127.0.0.1"].includes(location.hostname)
  );
}

function setupEdit(): void {
  const btn = document.querySelector<HTMLButtonElement>("#edit-btn");
  const pub = document.querySelector<HTMLButtonElement>("#publish-btn");
  const article = document.querySelector<HTMLElement>("#article");
  const path = article?.dataset.path;
  if (!btn || !pub || !article || !path) return;

  // Visitors never see the buttons: they show only once this browser holds an
  // encrypted token, or when the page is opened with ?edit to set one up.
  if (!isOwner()) return;
  btn.hidden = false;

  deployedHash =
    document.querySelector<HTMLElement>("#md-source")?.dataset.hash ?? "";
  restoreDraft(path, article);

  const place = () => {
    placeButton(btn);
    placeButton(pub, 60);
  };
  place();
  window.addEventListener("resize", place);
  article.addEventListener("input", onInput);
  article.addEventListener("keydown", onKeyDown);
  article.addEventListener("beforeinput", onBeforeInput);
  // Paste as plain text, so no foreign HTML ends up in the note.
  article.addEventListener("paste", e => {
    if (!article.classList.contains("editing")) return;
    e.preventDefault();
    const text = e.clipboardData?.getData("text/plain") ?? "";
    if (!text) return;
    record();
    const inCode = !!getSelection()?.anchorNode?.parentElement?.closest("pre");
    insertText(inCode ? text : text.replace(/\s*\n\s*/g, " "));
  });

  const setMode = (el: HTMLElement, mode: string, label: string) => {
    el.dataset.mode = mode;
    el.setAttribute("aria-label", label);
    el.title = label;
  };

  // The publish button shows how many notes have unpublished drafts.
  const refreshPublish = () => {
    const count = Object.keys(readDrafts()).length;
    pub.hidden = count === 0 && pub.dataset.mode === "idle";
    pub.querySelector(".edit-badge")!.textContent = String(count);
    if (pub.dataset.mode === "idle") {
      setMode(pub, "idle", `Publish ${count} draft${count === 1 ? "" : "s"}`);
    }
    place();
  };
  refreshPublish();

  const formatBar = createFormatBar(article);
  const undoBtn = document.querySelector<HTMLButtonElement>("#undo-btn")!;
  const redoBtn = document.querySelector<HTMLButtonElement>("#redo-btn")!;
  // Keep focus (and the caret) in the text when clicking these.
  for (const [button, action] of [
    [undoBtn, undo],
    [redoBtn, redo],
  ] as const) {
    button.addEventListener("mousedown", e => e.preventDefault());
    button.addEventListener("click", action);
  }

  const setEditing = (editing: boolean) => {
    undoBtn.hidden = redoBtn.hidden = !editing;
    if (editing) {
      placeButton(undoBtn, -120);
      placeButton(redoBtn, -60);
    }
    article.classList.toggle("editing", editing);
    if (!editing) {
      article.removeAttribute("contenteditable");
      article
        .querySelectorAll("[contenteditable]")
        .forEach(el => el.removeAttribute("contenteditable"));
      formatBar.hidden = true;
      return;
    }
    // One editing area, so a selection can span blocks (tables included)
    // and ⌘A selects all. Code blocks, diagrams and the like stay locked.
    article.contentEditable = "true";
    const blocks = new Set(units());
    for (const child of article.children) {
      const el = child as HTMLElement;
      if (blocks.has(el)) continue;
      if (![...blocks].some(b => el.contains(b))) {
        el.contentEditable = "false";
      }
    }
    article
      .querySelectorAll<HTMLElement>(".heading-link, #md-source")
      .forEach(el => (el.contentEditable = "false"));
  };

  // Run an action with a spinning ring, then flash the result for two seconds.
  const run = async (
    el: HTMLElement,
    action: () => Promise<Result>,
    labels: {
      saved: string;
      after: string;
      afterLabel: string;
      onSaved?: () => void;
    }
  ): Promise<Result | "failed"> => {
    setMode(el, "saving", "Working…");
    const started = Date.now();
    try {
      const result = await action();
      await sleep(Math.max(0, 1000 - (Date.now() - started)));
      if (result === "cancelled") return result;
      setMode(
        el,
        result === "saved" ? "saved" : "unchanged",
        result === "saved" ? labels.saved : "No changes"
      );
      if (result === "saved") labels.onSaved?.();
      await sleep(2000);
      setMode(el, labels.after, labels.afterLabel);
      return result;
    } catch (error) {
      toast("Something went wrong", "error", (error as Error).message);
      setMode(el, "failed", "Failed");
      await sleep(2000);
      return "failed";
    }
  };

  // ⌘Z / ⇧⌘Z / ⌘Y anywhere on the page while editing, wherever the caret is.
  // Esc leaves edit mode.
  document.addEventListener(
    "keydown",
    e => {
      if (!article.classList.contains("editing")) return;
      if (document.querySelector(".review-overlay, dialog[open]")) return;
      if (e.isComposing) return;
      const key = e.key.toLowerCase();
      if ((e.metaKey || e.ctrlKey) && (key === "z" || key === "y")) {
        e.preventDefault();
        e.stopPropagation();
        if (key === "y" || e.shiftKey) redo();
        else undo();
      } else if (e.key === "Escape") {
        if (
          hasUnsaved() &&
          !confirm("Leave edit mode and drop unsaved changes?")
        )
          return;
        if (hasUnsaved()) {
          article.classList.remove("editing");
          location.reload();
          return;
        }
        setEditing(false);
        setMode(btn, "idle", "Edit page");
      }
    },
    true
  );

  // Warn before leaving the page with edits that aren't saved as a draft.
  window.addEventListener("beforeunload", e => {
    if (article.classList.contains("editing") && hasUnsaved())
      e.preventDefault();
  });

  btn.addEventListener("click", async () => {
    const mode = btn.dataset.mode;
    if (mode === "idle") {
      remember();
      resetHistory();
      setEditing(true);
      setMode(btn, "editing", "Save draft");
      return;
    }
    if (mode !== "editing") return;
    const result = await run(btn, () => saveDraft(path), {
      saved: "Draft saved",
      after: "editing",
      afterLabel: "Save draft",
      onSaved: () => toast("Draft saved", "success"),
    });
    setMode(btn, "editing", "Save draft");
    if (result !== "saved") return;
    // Stay in edit mode; the saved text is the new baseline.
    remember();
    resetHistory();
    refreshPublish();
  });

  // Publishing opens a review panel with the diff of every draft.
  pub.addEventListener("click", async () => {
    if (pub.dataset.mode !== "idle") return;
    const drafts = readDrafts();
    const paths = Object.keys(drafts);
    if (!paths.length) return;
    let bases: string[];
    try {
      bases = await Promise.all(paths.map(fetchSource));
    } catch (error) {
      toast("Something went wrong", "error", (error as Error).message);
      return;
    }
    openReview(
      paths.map((p, i) => ({
        path: p,
        title: drafts[p].title,
        base: bases[i],
        text: drafts[p].raw,
      })),
      {
        save: async texts => {
          const result = await run(pub, () => publishDrafts(texts), {
            saved: "Published",
            after: "idle",
            afterLabel: "Publish",
            onSaved: () =>
              toast("Published", "success", "Live in about a minute"),
          });
          setMode(pub, "idle", "Publish");
          refreshPublish();
          if (result !== "saved") throw new Error("not published");
        },
        discard: () => {
          writeDrafts({});
          location.reload();
        },
      }
    );
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
