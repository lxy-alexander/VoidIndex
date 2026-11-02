#!/usr/bin/env node
// Move a note's local images to R2 before it is published, so pictures never
// enter the git repo. Every image a note links to on disk (Markdown ![](...)
// or <img src="...">, relative to the note or absolute) is uploaded to the R2
// bucket "blog" under posts/<slug>/<content-hash>.<ext>, and the link in the
// note is rewritten to https://img.voidindex.net/... . Links that are already
// URLs, or that point at nothing on disk, are left alone.
//
//   node scripts/upload-images.mjs <note.md> [...]
//
// Auth is wrangler's own login, as for deploy-cf.sh.
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { basename, dirname, extname, isAbsolute, resolve } from "node:path";

const BUCKET = "blog";
const BASE_URL = "https://img.voidindex.net";
const ACCOUNT_ID = "1f1357c6bd74676420403146f10c3b75";
const TYPES = {
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".webp": "image/webp",
  ".avif": "image/avif",
  ".svg": "image/svg+xml",
};

// ![alt](<path> "title") / ![alt](path "title"), and <img ... src="path">.
const MD_IMAGE = /(!\[[^\]]*\]\()(<[^>]+>|[^)\s]+)((?:\s+"[^"]*")?\))/g;
const HTML_IMAGE = /(<img\b[^>]*?\bsrc=["'])([^"']+)(["'])/gi;

const repoRoot = resolve(dirname(new URL(import.meta.url).pathname), "..");

function localFile(link, noteDir, rootUrl) {
  let path = link.replace(/^<|>$/g, "").replace(/^file:\/\//, "");
  if (/^[a-z][a-z0-9+.-]*:/i.test(path) || path.startsWith("#")) return null;
  try {
    path = decodeURI(path);
  } catch {}
  const candidates = isAbsolute(path)
    ? [path, rootUrl && resolve(noteDir, rootUrl, "." + path)]
    : [resolve(noteDir, path)];
  return (
    candidates.find(p => p && existsSync(p) && statSync(p).isFile()) ?? null
  );
}

function upload(file, key, type) {
  execFileSync(
    "pnpm",
    [
      "exec",
      "wrangler",
      "r2",
      "object",
      "put",
      `${BUCKET}/${key}`,
      "--file",
      file,
      "--content-type",
      type,
      "--cache-control",
      "public, max-age=31536000, immutable",
      "--remote",
    ],
    {
      cwd: repoRoot,
      env: { ...process.env, CLOUDFLARE_ACCOUNT_ID: ACCOUNT_ID },
      stdio: ["ignore", "ignore", "inherit"],
    }
  );
}

let uploaded = 0;
for (const note of process.argv.slice(2)) {
  const notePath = resolve(note);
  const noteDir = dirname(notePath);
  const slug = basename(notePath).replace(/\.mdx?$/, "");
  const text = readFileSync(notePath, "utf8");
  const rootUrl = text.match(/^typora-root-url:\s*(.+)$/m)?.[1].trim();
  const done = new Map();

  const rewrite = (whole, head, link, tail) => {
    const file = localFile(link, noteDir, rootUrl);
    if (!file) return whole;
    const ext = extname(file).toLowerCase();
    const type = TYPES[ext];
    if (!type) {
      console.warn(`skipped (not an image type): ${file}`);
      return whole;
    }
    if (!done.has(file)) {
      const hash = createHash("sha256")
        .update(readFileSync(file))
        .digest("hex")
        .slice(0, 16);
      const key = `posts/${slug}/${hash}${ext}`;
      upload(file, key, type);
      uploaded++;
      console.log(`${basename(file)} -> ${BASE_URL}/${key}`);
      done.set(file, `${BASE_URL}/${key}`);
    }
    return head + done.get(file) + tail;
  };

  const next = text
    .replace(MD_IMAGE, rewrite)
    .replace(HTML_IMAGE, rewrite);
  if (next !== text) writeFileSync(notePath, next);
}
console.log(`${uploaded} image(s) uploaded`);
