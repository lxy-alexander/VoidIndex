/**
 * Font picker: a menu of fonts suited to technical writing. The choice is
 * stored in localStorage and applied before first paint by Layout.astro.
 */
const FONT_KEY = "page-font";

type Font = { name: string; stack: string };

const MONO = "ui-monospace, monospace";
const SANS = "ui-sans-serif, system-ui, sans-serif";
const SERIF = "ui-serif, Georgia, serif";

// The first entry is the site default and needs no extra stylesheet.
const FONTS: Font[] = [
  { name: "Fira Code", stack: "" },
  { name: "Google Sans Code", stack: MONO },
  { name: "JetBrains Mono", stack: MONO },
  { name: "IBM Plex Mono", stack: MONO },
  { name: "Inter", stack: SANS },
  { name: "IBM Plex Sans", stack: SANS },
  { name: "Source Sans 3", stack: SANS },
  { name: "Noto Sans SC", stack: SANS },
  { name: "Source Serif 4", stack: SERIF },
  { name: "Literata", stack: SERIF },
];

function stylesheetUrl(name: string): string {
  return `https://fonts.googleapis.com/css2?family=${name.replace(/ /g, "+")}:wght@400;500;600;700&display=swap`;
}

function currentName(): string {
  try {
    return JSON.parse(localStorage.getItem(FONT_KEY) ?? "null")?.name ?? FONTS[0].name;
  } catch {
    return FONTS[0].name;
  }
}

function applyFont(font: Font): void {
  const root = document.documentElement;
  document.querySelector("#page-font-css")?.remove();
  if (!font.stack) {
    localStorage.removeItem(FONT_KEY);
    root.removeAttribute("data-font");
    root.style.removeProperty("--user-font");
    return;
  }
  const value = { name: font.name, family: `"${font.name}", ${font.stack}`, url: stylesheetUrl(font.name) };
  localStorage.setItem(FONT_KEY, JSON.stringify(value));
  const link = document.createElement("link");
  link.id = "page-font-css";
  link.rel = "stylesheet";
  link.href = value.url;
  document.head.append(link);
  root.setAttribute("data-font", "");
  root.style.setProperty("--user-font", value.family);
}

function setupFont(): void {
  const btn = document.querySelector<HTMLButtonElement>("#font-btn");
  const menu = document.querySelector<HTMLElement>("#font-menu");
  if (!btn || !menu) return;

  const render = () => {
    const active = currentName();
    menu.replaceChildren(
      ...FONTS.map(font => {
        const item = document.createElement("button");
        item.type = "button";
        item.role = "menuitemradio";
        item.ariaChecked = String(font.name === active);
        item.textContent = font.name;
        item.style.fontFamily = font.stack ? `"${font.name}", ${font.stack}` : "";
        item.addEventListener("click", () => {
          applyFont(font);
          close();
        });
        return item;
      })
    );
  };

  const close = () => {
    menu.hidden = true;
    btn.setAttribute("aria-expanded", "false");
  };

  btn.addEventListener("click", e => {
    e.stopPropagation();
    if (!menu.hidden) return close();
    render();
    // Load every font's stylesheet so each item previews in its own face.
    FONTS.slice(1).forEach(font => {
      if (document.querySelector(`link[data-preview="${font.name}"]`)) return;
      const link = document.createElement("link");
      link.rel = "stylesheet";
      link.href = stylesheetUrl(font.name);
      link.dataset.preview = font.name;
      document.head.append(link);
    });
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

setupFont();
document.addEventListener("astro:after-swap", setupFont);

export {};
