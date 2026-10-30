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
const EDITABLE = "#article [data-md-s]";

type Block = { el: HTMLElement; before: string };

let blocks: Block[] = [];

// Inline HTML back to Markdown: bold, italic, code and links survive.
function toMarkdown(node: Node): string {
  if (node.nodeType === Node.TEXT_NODE) {
    return (node.textContent ?? "").replace(/ /g, " ");
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

function getToken(): string | null {
  let token = localStorage.getItem(TOKEN_KEY);
  if (!token) {
    token = prompt(
      `GitHub token with Contents read/write on ${REPO} only.\nIt is stored in this browser.`
    );
    if (token) localStorage.setItem(TOKEN_KEY, token.trim());
  }
  return token?.trim() || null;
}

function toast(message: string): void {
  const el = document.createElement("div");
  el.className = "edit-toast";
  el.textContent = message;
  document.body.append(el);
  setTimeout(() => el.remove(), 5000);
}

async function save(path: string): Promise<boolean> {
  const changed = blocks
    .map(b => ({ ...b, after: blockMarkdown(b.el) }))
    .filter(b => b.after !== b.before);
  if (changed.length === 0) return true;

  const token = getToken();
  if (!token) return false;
  const api = `https://api.github.com/repos/${REPO}/contents/${path}`;
  const headers = {
    Authorization: `Bearer ${token}`,
    Accept: "application/vnd.github+json",
  };

  const res = await fetch(`${api}?ref=${BRANCH}`, { headers });
  if (res.status === 401) localStorage.removeItem(TOKEN_KEY);
  if (!res.ok) throw new Error(`Reading the file failed (${res.status})`);
  const file = (await res.json()) as { content: string; sha: string };
  const raw = decodeBase64(file.content);

  // Offsets are relative to the trimmed body after the frontmatter.
  const fm = raw.match(/^---\r?\n[\s\S]*?\r?\n---/)?.[0] ?? "";
  const rest = raw.slice(fm.length);
  const bodyStart = fm.length + (rest.length - rest.trimStart().length);
  const hashEl = document.querySelector<HTMLElement>("#md-source");
  if ((await sha1(rest.trim())) !== hashEl?.dataset.hash) {
    throw new Error(
      "The note changed since this page was built. Reload after the deploy finishes, then edit again."
    );
  }

  // Splice from the end so earlier offsets stay valid.
  let next = raw;
  const edits = changed
    .map(b => ({
      s: Number(b.el.dataset.mdS),
      e: Number(b.el.dataset.mdE),
      text: b.after,
    }))
    .sort((a, b) => b.s - a.s);
  for (const { s, e, text } of edits) {
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
  if (put.status === 401) localStorage.removeItem(TOKEN_KEY);
  if (!put.ok) throw new Error(`Saving failed (${put.status})`);

  // Shift the offsets of later blocks so a second edit on this page lands right.
  const shifts = edits.map(x => ({
    e: x.e,
    delta: x.text.length - (x.e - x.s),
  }));
  document.querySelectorAll<HTMLElement>(EDITABLE).forEach(el => {
    const s = Number(el.dataset.mdS);
    const e = Number(el.dataset.mdE);
    const edited = edits.find(x => x.s === s);
    const before = shifts
      .filter(x => x.e <= s)
      .reduce((n, x) => n + x.delta, 0);
    el.dataset.mdS = String(s + before);
    el.dataset.mdE = String(
      e + before + (edited ? edited.text.length - (e - s) : 0)
    );
  });
  if (hashEl) {
    const nextRest = next.slice(fm.length);
    hashEl.dataset.hash = await sha1(nextRest.trim());
  }
  return true;
}

function setupEdit(): void {
  const btn = document.querySelector<HTMLButtonElement>("#edit-btn");
  const article = document.querySelector<HTMLElement>("#article");
  const path = article?.dataset.path;
  if (!btn || !article || !path) return;

  const setMode = (editing: boolean) => {
    btn.dataset.mode = editing ? "editing" : "idle";
    btn.setAttribute("aria-label", editing ? "Save and publish" : "Edit page");
    btn.title = btn.getAttribute("aria-label")!;
    article.classList.toggle("editing", editing);
    document.querySelectorAll<HTMLElement>(EDITABLE).forEach(el => {
      if (editing) el.contentEditable = "true";
      else el.removeAttribute("contenteditable");
    });
  };

  btn.addEventListener("click", async () => {
    if (btn.dataset.mode === "saving") return;
    if (btn.dataset.mode !== "editing") {
      blocks = [...document.querySelectorAll<HTMLElement>(EDITABLE)].map(
        el => ({
          el,
          before: blockMarkdown(el),
        })
      );
      setMode(true);
      return;
    }
    btn.dataset.mode = "saving";
    try {
      const saved = await save(path);
      if (!saved) {
        btn.dataset.mode = "editing";
        return;
      }
      setMode(false);
      toast("Saved and pushed. The live site updates in about 1–2 minutes.");
    } catch (error) {
      btn.dataset.mode = "editing";
      toast((error as Error).message);
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
