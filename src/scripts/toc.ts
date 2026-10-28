/**
 * Sticky "On this page" sidebar for posts: lists the article's h2/h3,
 * and highlights the section being read. CSS shows it only on wide screens
 * (see .toc-aside in global.css).
 */
const ACTIVE_OFFSET = 120;

let cleanup: (() => void) | undefined;

function setupToc(): void {
  cleanup?.();
  cleanup = undefined;

  const aside = document.querySelector<HTMLElement>("#toc-aside");
  const main = document.querySelector<HTMLElement>("#main-content");
  const list = aside?.querySelector("ul");
  if (!aside || !main || !list) return;

  const headings = [
    ...document.querySelectorAll<HTMLElement>(
      "#article h2[id], #article h3[id]"
    ),
  ].filter(h => h.id !== "table-of-contents");
  if (headings.length === 0) return;

  list.replaceChildren(
    ...headings.map(h => {
      const li = document.createElement("li");
      li.className = h.tagName === "H3" ? "toc-h3" : "toc-h2";
      const a = document.createElement("a");
      a.href = `#${h.id}`;
      // Drop the "#" anchor link that headings carry.
      a.textContent = (h.textContent ?? "").replace(/#\s*$/, "").trim();
      li.append(a);
      return li;
    })
  );
  const links = [...list.querySelectorAll("a")];

  const highlight = () => {
    let active = 0;
    headings.forEach((h, i) => {
      if (h.getBoundingClientRect().top <= ACTIVE_OFFSET) active = i;
    });
    links.forEach((a, i) => a.classList.toggle("active", i === active));
  };

  window.addEventListener("scroll", highlight, { passive: true });
  main.classList.add("has-toc");
  aside.hidden = false;
  highlight();

  cleanup = () => window.removeEventListener("scroll", highlight);
}

document.addEventListener("astro:page-load", setupToc);

export {};
