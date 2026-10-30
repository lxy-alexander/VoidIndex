/**
 * Review panel for publishing drafts: a GitHub-style line diff (+ added,
 * − removed) of each note against the version on GitHub. Added and unchanged
 * lines can be edited in place; "Save Changes" commits what the panel shows.
 */
type Row = { type: "same" | "add" | "del"; text: string };

// Line diff via longest common subsequence, after trimming the shared head
// and tail so typical small edits stay cheap.
function diffLines(before: string[], after: string[]): Row[] {
  let head = 0;
  while (
    head < before.length &&
    head < after.length &&
    before[head] === after[head]
  )
    head++;
  let tail = 0;
  while (
    tail < before.length - head &&
    tail < after.length - head &&
    before[before.length - 1 - tail] === after[after.length - 1 - tail]
  ) {
    tail++;
  }
  const a = before.slice(head, before.length - tail);
  const b = after.slice(head, after.length - tail);
  const lcs = Array.from(
    { length: a.length + 1 },
    () => new Uint32Array(b.length + 1)
  );
  for (let i = a.length - 1; i >= 0; i--) {
    for (let j = b.length - 1; j >= 0; j--) {
      lcs[i][j] =
        a[i] === b[j]
          ? lcs[i + 1][j + 1] + 1
          : Math.max(lcs[i + 1][j], lcs[i][j + 1]);
    }
  }
  const middle: Row[] = [];
  let i = 0;
  let j = 0;
  while (i < a.length || j < b.length) {
    if (i < a.length && j < b.length && a[i] === b[j]) {
      middle.push({ type: "same", text: a[i] });
      i++;
      j++;
    } else if (
      i < a.length &&
      (j === b.length || lcs[i + 1][j] >= lcs[i][j + 1])
    ) {
      // Removed lines first, like GitHub.
      middle.push({ type: "del", text: a[i++] });
    } else {
      middle.push({ type: "add", text: b[j++] });
    }
  }
  return [
    ...before.slice(0, head).map(text => ({ type: "same" as const, text })),
    ...middle,
    ...before
      .slice(before.length - tail)
      .map(text => ({ type: "same" as const, text })),
  ];
}

const CONTEXT = 3;

function row(type: Row["type"], text: string): HTMLElement {
  const el = document.createElement("div");
  el.className = `diff-row diff-${type}`;
  const sign = document.createElement("span");
  sign.className = "diff-sign";
  sign.textContent = type === "add" ? "+" : type === "del" ? "−" : " ";
  const body = document.createElement("span");
  body.className = "diff-text";
  body.textContent = text;
  // Removed lines are shown for reference only.
  if (type !== "del") body.contentEditable = "plaintext-only";
  el.append(sign, body);
  return el;
}

// A run of unchanged lines collapsed behind "⋯ N unchanged lines".
function fold(lines: string[]): HTMLElement {
  const el = document.createElement("button");
  el.type = "button";
  el.className = "diff-fold";
  el.textContent = `⋯ ${lines.length} unchanged line${lines.length === 1 ? "" : "s"}`;
  el.dataset.lines = JSON.stringify(lines);
  el.addEventListener("click", () =>
    el.replaceWith(...lines.map(text => row("same", text)))
  );
  return el;
}

function renderDiff(box: HTMLElement, base: string, text: string): void {
  const rows = diffLines(base.split("\n"), text.split("\n"));
  const nodes: HTMLElement[] = [];
  let k = 0;
  while (k < rows.length) {
    if (rows[k].type !== "same") {
      nodes.push(row(rows[k].type, rows[k].text));
      k++;
      continue;
    }
    let end = k;
    while (end < rows.length && rows[end].type === "same") end++;
    const run = rows.slice(k, end).map(r => r.text);
    const keepHead = k === 0 ? 0 : CONTEXT;
    const keepTail = end === rows.length ? 0 : CONTEXT;
    if (run.length > keepHead + keepTail + 1) {
      run.slice(0, keepHead).forEach(t => nodes.push(row("same", t)));
      nodes.push(fold(run.slice(keepHead, run.length - keepTail)));
      run.slice(run.length - keepTail).forEach(t => nodes.push(row("same", t)));
    } else {
      run.forEach(t => nodes.push(row("same", t)));
    }
    k = end;
  }
  box.replaceChildren(...nodes);
}

// The note as currently shown: unchanged and added lines, folds included.
function collect(box: HTMLElement): string {
  const lines: string[] = [];
  for (const el of box.children) {
    if (el instanceof HTMLElement && el.classList.contains("diff-fold")) {
      lines.push(...(JSON.parse(el.dataset.lines ?? "[]") as string[]));
    } else if (!el.classList.contains("diff-del")) {
      lines.push(el.querySelector(".diff-text")?.textContent ?? "");
    }
  }
  return lines.join("\n");
}

export type ReviewFile = {
  path: string;
  title: string;
  base: string;
  text: string;
};

export function openReview(
  files: ReviewFile[],
  actions: {
    save: (texts: Record<string, string>) => Promise<void>;
    discard: () => void;
  }
): void {
  const overlay = document.createElement("div");
  overlay.className = "review-overlay";
  overlay.innerHTML = `
    <div class="review-panel" role="dialog" aria-modal="true" aria-label="Review changes">
      <header class="review-head">
        <span class="review-title"></span>
        <div class="review-actions">
          <button type="button" data-action="discard">Discard drafts</button>
          <button type="button" data-action="close">Close</button>
          <button type="button" data-action="save" class="review-save">Save Changes</button>
        </div>
      </header>
      <div class="review-body"></div>
    </div>`;
  overlay.querySelector(".review-title")!.textContent =
    `Review changes · ${files.length} note${files.length === 1 ? "" : "s"}`;
  const body = overlay.querySelector<HTMLElement>(".review-body")!;
  const boxes = new Map<string, HTMLElement>();
  for (const file of files) {
    const section = document.createElement("section");
    section.className = "review-file";
    const title = document.createElement("h3");
    title.textContent = file.title;
    const path = document.createElement("code");
    path.textContent = file.path;
    title.append(" ", path);
    const box = document.createElement("div");
    box.className = "diff";
    renderDiff(box, file.base, file.text);
    // Recolour the diff once editing moves out of this note.
    box.addEventListener("focusout", e => {
      if (box.contains(e.relatedTarget as Node)) return;
      renderDiff(box, file.base, collect(box));
    });
    section.append(title, box);
    body.append(section);
    boxes.set(file.path, box);
  }

  const close = () => {
    overlay.remove();
    document.documentElement.style.overflow = "";
    document.removeEventListener("keydown", onKey, true);
  };
  const onKey = (e: KeyboardEvent) => {
    if (e.key === "Escape" && !document.querySelector("dialog[open]")) close();
  };
  document.addEventListener("keydown", onKey, true);

  const saveBtn = overlay.querySelector<HTMLButtonElement>(
    '[data-action="save"]'
  )!;
  overlay
    .querySelector(".review-actions")!
    .addEventListener("click", async e => {
      const action = (e.target as HTMLElement).closest("button")?.dataset
        .action;
      if (action === "close") close();
      if (action === "discard" && confirm("Discard all unpublished drafts?")) {
        close();
        actions.discard();
      }
      if (action !== "save" || saveBtn.disabled) return;
      saveBtn.disabled = true;
      saveBtn.textContent = "Saving…";
      try {
        await actions.save(
          Object.fromEntries([...boxes].map(([p, box]) => [p, collect(box)]))
        );
        close();
      } catch {
        saveBtn.disabled = false;
        saveBtn.textContent = "Save Changes";
      }
    });
  // Clicking the dim backdrop closes the panel too.
  overlay.addEventListener("mousedown", e => {
    if (e.target === overlay) close();
  });

  document.body.append(overlay);
  document.documentElement.style.overflow = "hidden";
}
