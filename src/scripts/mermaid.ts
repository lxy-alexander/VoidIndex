/**
 * Renders ```mermaid code blocks as diagrams.
 *
 * Inline diagrams are fitted to the content width and capped in height.
 * Clicking one opens a full-screen viewer with drag-to-pan, wheel and
 * pinch zoom, and double-click to reset.
 */
const SELECTOR = "pre:has(> code.language-mermaid)";
const MIN_SCALE = 0.2;
const MAX_SCALE = 8;

let diagramCount = 0;

function isDark(): boolean {
  return document.documentElement.classList.contains("dark");
}

async function renderAll(): Promise<void> {
  const figures = [
    ...document.querySelectorAll<HTMLElement>(".mermaid-figure"),
  ];
  const blocks = [...document.querySelectorAll<HTMLPreElement>(SELECTOR)];
  if (figures.length === 0 && blocks.length === 0) return;

  // Swap code blocks for empty figures that remember their source.
  for (const pre of blocks) {
    const figure = document.createElement("figure");
    figure.className = "mermaid-figure";
    figure.dataset.source = pre.querySelector("code")?.textContent ?? "";
    figure.tabIndex = 0;
    figure.setAttribute("role", "button");
    figure.setAttribute("aria-label", "Open diagram in full screen");
    pre.replaceWith(figure);
    figures.push(figure);
  }

  const { default: mermaid } = await import("mermaid");
  mermaid.initialize({
    startOnLoad: false,
    theme: isDark() ? "dark" : "default",
    securityLevel: "strict",
  });

  for (const figure of figures) {
    try {
      const { svg } = await mermaid.render(
        `mermaid-${++diagramCount}`,
        figure.dataset.source ?? ""
      );
      figure.innerHTML = `<div class="mermaid-frame">${svg}</div><span class="mermaid-hint">Click to expand</span>`;
      fitSvg(figure.querySelector<SVGSVGElement>(".mermaid-frame > svg"));
      figure.classList.remove("mermaid-error");
    } catch (error) {
      figure.textContent = `Mermaid error: ${(error as Error).message}`;
      figure.classList.add("mermaid-error");
    }
  }
}

// Give the SVG its natural size so CSS max-width/max-height can scale it down.
function fitSvg(svg: SVGSVGElement | null): void {
  if (!svg) return;
  const { width, height } = svg.viewBox.baseVal;
  svg.removeAttribute("style");
  svg.setAttribute("width", String(width));
  svg.setAttribute("height", String(height));
}

function openViewer(figure: HTMLElement): void {
  const source = figure.querySelector<SVGSVGElement>(".mermaid-frame > svg");
  if (!source) return;

  const overlay = document.createElement("div");
  overlay.className = "mermaid-viewer";
  overlay.innerHTML = `
    <div class="mermaid-viewer-stage"></div>
    <div class="mermaid-viewer-bar">
      <button type="button" data-action="out" aria-label="Zoom out">−</button>
      <button type="button" data-action="reset" aria-label="Reset zoom">⟲</button>
      <button type="button" data-action="in" aria-label="Zoom in">+</button>
      <button type="button" data-action="close" aria-label="Close">✕</button>
    </div>`;
  const stage = overlay.querySelector<HTMLElement>(".mermaid-viewer-stage")!;
  const svg = source.cloneNode(true) as SVGSVGElement;
  stage.append(svg);
  document.body.append(overlay);
  document.documentElement.style.overflow = "hidden";

  let scale = 1;
  let x = 0;
  let y = 0;

  const apply = () => {
    svg.style.transform = `translate(${x}px, ${y}px) scale(${scale})`;
  };

  // Start fitted to the screen and centred.
  const reset = () => {
    const { width, height } = svg.viewBox.baseVal;
    const box = stage.getBoundingClientRect();
    scale = Math.min(box.width / width, box.height / height) * 0.95;
    x = (box.width - width * scale) / 2;
    y = (box.height - height * scale) / 2;
    apply();
  };

  // Zoom by `factor`, keeping the point (cx, cy) on the stage fixed.
  const zoomAt = (factor: number, cx: number, cy: number) => {
    const next = Math.min(MAX_SCALE, Math.max(MIN_SCALE, scale * factor));
    x = cx - ((cx - x) * next) / scale;
    y = cy - ((cy - y) * next) / scale;
    scale = next;
    apply();
  };

  const zoomCenter = (factor: number) => {
    const box = stage.getBoundingClientRect();
    zoomAt(factor, box.width / 2, box.height / 2);
  };

  const pointers = new Map<number, { x: number; y: number }>();
  let pinchDistance = 0;

  const local = (e: PointerEvent | WheelEvent | MouseEvent) => {
    const box = stage.getBoundingClientRect();
    return { x: e.clientX - box.left, y: e.clientY - box.top };
  };

  stage.addEventListener("pointerdown", e => {
    stage.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, local(e));
    if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      pinchDistance = Math.hypot(a.x - b.x, a.y - b.y);
    }
  });

  stage.addEventListener("pointermove", e => {
    const prev = pointers.get(e.pointerId);
    if (!prev) return;
    const point = local(e);
    pointers.set(e.pointerId, point);

    if (pointers.size === 1) {
      x += point.x - prev.x;
      y += point.y - prev.y;
      apply();
    } else if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      const distance = Math.hypot(a.x - b.x, a.y - b.y);
      if (pinchDistance > 0) {
        zoomAt(distance / pinchDistance, (a.x + b.x) / 2, (a.y + b.y) / 2);
      }
      pinchDistance = distance;
    }
  });

  const release = (e: PointerEvent) => {
    pointers.delete(e.pointerId);
    pinchDistance = 0;
  };
  stage.addEventListener("pointerup", release);
  stage.addEventListener("pointercancel", release);

  stage.addEventListener(
    "wheel",
    e => {
      e.preventDefault();
      // Trackpad pinch arrives as ctrl+wheel with small deltas.
      const factor = Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.002));
      const point = local(e);
      zoomAt(factor, point.x, point.y);
    },
    { passive: false }
  );

  stage.addEventListener("dblclick", reset);

  const close = () => {
    overlay.remove();
    document.documentElement.style.overflow = "";
    document.removeEventListener("keydown", onKey);
    figure.focus();
  };

  const onKey = (e: KeyboardEvent) => {
    if (e.key === "Escape") close();
    else if (e.key === "+" || e.key === "=") zoomCenter(1.25);
    else if (e.key === "-") zoomCenter(0.8);
    else if (e.key === "0") reset();
  };
  document.addEventListener("keydown", onKey);

  overlay.querySelector(".mermaid-viewer-bar")?.addEventListener("click", e => {
    const action = (e.target as HTMLElement).closest("button")?.dataset.action;
    if (action === "in") zoomCenter(1.25);
    else if (action === "out") zoomCenter(0.8);
    else if (action === "reset") reset();
    else if (action === "close") close();
  });

  reset();
  overlay.querySelector<HTMLButtonElement>('[data-action="close"]')?.focus();
}

document.addEventListener("click", e => {
  const figure = (e.target as HTMLElement).closest<HTMLElement>(
    ".mermaid-figure"
  );
  if (figure) openViewer(figure);
});

document.addEventListener("keydown", e => {
  const figure = (e.target as HTMLElement).closest?.<HTMLElement>(
    ".mermaid-figure"
  );
  if (figure && (e.key === "Enter" || e.key === " ")) {
    e.preventDefault();
    openViewer(figure);
  }
});

// Re-render with matching colours when the light/dark theme changes.
new MutationObserver(() => void renderAll()).observe(document.documentElement, {
  attributes: true,
  attributeFilter: ["class"],
});

document.addEventListener("astro:page-load", () => void renderAll());

export {};
