const WIDTH_KEY = "page-width";
const WIDTHS = ["normal", "wide", "full"] as const;
type Width = (typeof WIDTHS)[number];

function getStoredWidth(): Width {
  const stored = localStorage.getItem(WIDTH_KEY) as Width | null;
  return stored && WIDTHS.includes(stored) ? stored : "normal";
}

let widthValue: Width = getStoredWidth();

function reflectWidth(): void {
  document.firstElementChild?.setAttribute("data-width", widthValue);
  document
    .querySelector("#width-btn")
    ?.setAttribute("aria-label", `Page width: ${widthValue}`);
}

function setupWidth(): void {
  reflectWidth();
  document.querySelector("#width-btn")?.addEventListener("click", () => {
    widthValue = WIDTHS[(WIDTHS.indexOf(widthValue) + 1) % WIDTHS.length];
    localStorage.setItem(WIDTH_KEY, widthValue);
    reflectWidth();
  });
}

setupWidth();

// Re-run after View Transitions navigation.
document.addEventListener("astro:after-swap", setupWidth);
export {};
