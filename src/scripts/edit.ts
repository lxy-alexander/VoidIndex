/**
 * In-place editing for posts. The floating button switches the article into
 * edit mode (headings, paragraphs and table cells become editable); pressing
 * it again converts the changed blocks back to Markdown, splices them into
 * the .md file at the offsets from utils/remarkSourceMap.ts, and commits the
 * file to GitHub, which redeploys the site.
 */
const REPO = "lxy-alexander/VoidIndex";
const BRANCH = "main";
const TOKEN_KEY = "gh-token";
const UNITS = "#article [data-md-s], #article [data-new]";
const ZWSP = "\u200b";

// The innermost editable blocks, in page order. A loose list item renders as
// <li><p> with the same offsets on both; only the <p> is edited.
function units(): HTMLElement[] {
  return [...document.querySelectorAll<HTMLElement>(UNITS)].filter(
    el => !el.querySelector("[data-md-s], [data-new]")
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

// Block-level Markdown that goes in front of a new or converted block.
function blockPrefix(el: HTMLElement, listPrefix: string): string {
  const level = /^H([1-6])$/.exec(el.tagName)?.[1];
  if (level) return `${"#".repeat(Number(level))} `;
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

// The token lives in sessionStorage (gone when the tab closes) unless the
// user chooses to remember it on this device.
function getToken(): string | null {
  const stored =
    sessionStorage.getItem(TOKEN_KEY) ?? localStorage.getItem(TOKEN_KEY);
  if (stored) return stored;
  const token = prompt(
    `GitHub fine-grained token (Contents read/write on ${REPO} only):`
  )?.trim();
  if (!token) return null;
  const remember = confirm(
    "Remember this token on this device?\n\nOK: keep it until you clear it.\nCancel: forget it when this tab closes."
  );
  (remember ? localStorage : sessionStorage).setItem(TOKEN_KEY, token);
  return token;
}

function forgetToken(): void {
  sessionStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(TOKEN_KEY);
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

  for (const el of units()) {
    if (el.dataset.new === undefined) {
      flush();
      const s = Number(el.dataset.mdS);
      const e = Number(el.dataset.mdE);
      const inline = blockMarkdown(el);
      const converted = el.dataset.was !== undefined;
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
    const inline = blockMarkdown(el);
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
  const dirty = units().some(
    el =>
      el.dataset.new !== undefined ||
      el.dataset.was !== undefined ||
      blockMarkdown(el) !== before.get(el)
  );
  if (!dirty) return "unchanged";

  const token = getToken();
  if (!token) return "cancelled";
  const api = `https://api.github.com/repos/${REPO}/contents/${path}`;
  const headers = {
    Authorization: `Bearer ${token}`,
    Accept: "application/vnd.github+json",
  };

  const res = await fetch(`${api}?ref=${BRANCH}`, { headers });
  if (res.status === 401) forgetToken();
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
  if (put.status === 401) forgetToken();
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

function makeUnit(tag: string, from?: HTMLElement): HTMLElement {
  const el = document.createElement(tag);
  el.contentEditable = "true";
  if (from?.dataset.mdS !== undefined) {
    el.dataset.mdS = from.dataset.mdS;
    el.dataset.mdE = from.dataset.mdE;
    el.dataset.was = from.tagName;
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

// "# ", "## ", "- ", "1. " or "> " typed at the start of a paragraph turns it
// into a heading, list item or quote.
function applyBlockShortcut(): boolean {
  const unit = caretUnit();
  if (!unit || unit.tagName !== "P" || unit.closest("li, blockquote, td")) {
    return false;
  }
  const first = unit.firstChild;
  if (!(first instanceof Text)) return false;
  const m = first.data.match(/^(#{1,6}|[-*+]|\d+\.|>)[\s ]/);
  if (!m) return false;
  first.data = first.data.slice(m[0].length);
  let target: HTMLElement;
  let outer: HTMLElement;
  if (m[1].startsWith("#")) {
    target = outer = makeUnit(`H${m[1].length}`, unit);
  } else if (m[1] === ">") {
    outer = document.createElement("blockquote");
    target = makeUnit("P", unit);
    outer.append(target);
  } else {
    outer = document.createElement(/\d/.test(m[1]) ? "ol" : "ul");
    target = makeUnit("LI", unit);
    outer.append(target);
  }
  target.append(...unit.childNodes);
  if (!target.textContent) target.append(ZWSP);
  unit.replaceWith(outer);
  placeCaret(target);
  return true;
}

function onInput(): void {
  if (!applyBlockShortcut()) applyInlineShortcut();
}

function onKeyDown(e: KeyboardEvent): void {
  const unit = caretUnit();
  if (!unit) return;
  const sel = getSelection()!;
  const text = (unit.textContent ?? "").replace(/​/g, "");

  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    if (unit.tagName === "TD" || unit.tagName === "TH") return;
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
    (isItem ? unit.closest("li")! : unit).after(next);
    placeCaret(next);
    return;
  }

  // Backspace at the start of an empty new block removes it.
  if (
    e.key === "Backspace" &&
    unit.dataset.new !== undefined &&
    !text &&
    sel.isCollapsed
  ) {
    e.preventDefault();
    const all = units();
    const prev = all[all.indexOf(unit) - 1];
    const list = unit.tagName === "LI" ? unit.parentElement : null;
    unit.remove();
    if (list && !list.children.length) list.remove();
    if (prev) placeCaret(prev, true);
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
      before = new Map(units().map(el => [el, blockMarkdown(el)]));
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

export {};
