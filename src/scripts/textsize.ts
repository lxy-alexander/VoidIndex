/**
 * Text size menu: five levels for the article text. Only the prose scales
 * (its sizes are in em), so the header and menus keep their layout. The
 * choice is stored in localStorage and applied before first paint by
 * Layout.astro through the --article-size variable.
 */
const SIZE_KEY = "page-text-size";

// Article font size per level. "l" is the default and comes from the CSS
// (typography.css), so it needs nothing stored.
const DEFAULT = "l";
export const SIZES = [
  { id: "xs", label: "Smaller", px: 14 },
  { id: "s", label: "Small", px: 15 },
  { id: "m", label: "Medium", px: 16 },
  { id: "l", label: "Large (default)", px: 17.5 },
  { id: "xl", label: "Larger", px: 19 },
] as const;

function current(): string {
  try {
    return localStorage.getItem(SIZE_KEY) ?? DEFAULT;
  } catch {
    return DEFAULT;
  }
}

function apply(id: string): void {
  const size = SIZES.find(s => s.id === id) ?? SIZES[3];
  const root = document.documentElement.style;
  if (size.id === DEFAULT) root.removeProperty("--article-size");
  else root.setProperty("--article-size", `${size.px}px`);
  try {
    if (size.id === DEFAULT) localStorage.removeItem(SIZE_KEY);
    else localStorage.setItem(SIZE_KEY, size.id);
  } catch {
    // Storage unavailable (private mode): the size still applies to this page.
  }
}

function setupTextSize(): void {
  const btn = document.querySelector<HTMLButtonElement>("#size-btn");
  const menu = document.querySelector<HTMLElement>("#size-menu");
  if (!btn || !menu) return;

  const close = () => {
    menu.hidden = true;
    btn.setAttribute("aria-expanded", "false");
  };

  const render = () => {
    const active = current();
    menu.replaceChildren(
      ...SIZES.map(size => {
        const item = document.createElement("button");
        item.type = "button";
        item.setAttribute("role", "menuitemradio");
        item.setAttribute("aria-checked", String(size.id === active));
        item.className = "size-item";
        // Preview each level at its own size.
        const sample = document.createElement("span");
        sample.className = "size-sample";
        sample.style.fontSize = `${size.px}px`;
        sample.textContent = "Aa";
        const label = document.createElement("span");
        label.textContent = size.label;
        item.append(sample, label);
        item.addEventListener("click", () => {
          apply(size.id);
          close();
        });
        return item;
      })
    );
  };

  btn.addEventListener("click", e => {
    e.stopPropagation();
    if (!menu.hidden) return close();
    render();
    menu.hidden = false;
    btn.setAttribute("aria-expanded", "true");
  });
  document.addEventListener("click", e => {
    if (!menu.contains(e.target as Node)) close();
  });
  document.addEventListener("keydown", e => {
    if (e.key === "Escape" && !menu.hidden) {
      close();
      btn.focus();
    }
  });
}

document.addEventListener("astro:page-load", setupTextSize);
