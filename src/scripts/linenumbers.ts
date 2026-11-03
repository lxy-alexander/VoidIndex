/**
 * Code line numbers: the button next to the text size menu shows or hides
 * line numbers on every code block. The choice is stored in localStorage and
 * applied before first paint by Layout.astro through html[data-line-numbers].
 */
const LINES_KEY = "code-line-numbers";

function reflect(on: boolean): void {
  document.documentElement.toggleAttribute("data-line-numbers", on);
  const btn = document.querySelector("#lines-btn");
  btn?.setAttribute("aria-pressed", String(on));
}

function setupLineNumbers(): void {
  const btn = document.querySelector<HTMLButtonElement>("#lines-btn");
  if (!btn) return;
  reflect(document.documentElement.hasAttribute("data-line-numbers"));
  btn.addEventListener("click", () => {
    const on = !document.documentElement.hasAttribute("data-line-numbers");
    reflect(on);
    try {
      if (on) localStorage.setItem(LINES_KEY, "on");
      else localStorage.removeItem(LINES_KEY);
    } catch {
      // Storage unavailable (private mode): the toggle still applies to this page.
    }
  });
}

document.addEventListener("astro:page-load", setupLineNumbers);
export {};
