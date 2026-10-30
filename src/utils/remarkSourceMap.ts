import { createHash } from "node:crypto";

/**
 * Tags headings, paragraphs and table cells with the source offsets of their
 * inline content (data-md-s / data-md-e), and appends a hash of the whole
 * Markdown body. Offsets count from the start of the body: the text after
 * the frontmatter, trimmed, which is what Astro hands to remark.
 * scripts/edit.ts uses these to write in-page edits back to the exact spot
 * in the .md file and to detect a stale page.
 */
type Node = {
  type: string;
  value?: string;
  children?: Node[];
  position?: { start: { offset?: number }; end: { offset?: number } };
  data?: { hProperties?: Record<string, unknown> };
};

const TAGGED = new Set(["heading", "paragraph", "tableCell"]);

function tag(node: Node): void {
  // Tight list items render without a <p>, so tag the <li> itself when it
  // holds a single paragraph.
  const only = node.children?.length === 1 ? node.children[0] : undefined;
  const source =
    node.type === "listItem" && only?.type === "paragraph"
      ? only
      : TAGGED.has(node.type)
        ? node
        : undefined;
  if (source?.children?.length) {
    const s = source.children[0].position?.start.offset;
    const e = source.children.at(-1)?.position?.end.offset;
    if (s !== undefined && e !== undefined) {
      node.data = {
        ...node.data,
        hProperties: {
          ...node.data?.hProperties,
          "data-md-s": s,
          "data-md-e": e,
        },
      };
    }
  }
  node.children?.forEach(tag);
}

export default function remarkSourceMap() {
  return (tree: Node, file: { value: unknown }) => {
    tag(tree);
    const hash = createHash("sha1").update(String(file.value)).digest("hex");
    tree.children?.push({
      type: "html",
      value: `<span id="md-source" data-hash="${hash}" hidden></span>`,
    });
  };
}
