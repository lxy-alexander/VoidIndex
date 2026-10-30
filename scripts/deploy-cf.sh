#!/bin/bash
# Manual fallback only. Normally Cloudflare Pages (project "voidindex-site")
# builds every push to main from GitHub by itself. This builds the local
# checkout and uploads it directly; the next push replaces it again.
#
# Auth: `pnpm exec wrangler login` once (stores an OAuth token in wrangler's
# own config, readable only by you).
set -euo pipefail
cd "$(dirname "$0")/.."

export CLOUDFLARE_ACCOUNT_ID="${CLOUDFLARE_ACCOUNT_ID:-1f1357c6bd74676420403146f10c3b75}"

# Publishing notes doesn't change code, so skip the ~15 s type check here;
# `pnpm run build` on its own still runs it.
SKIP_CHECK=1 pnpm run build
# Cache rules for Cloudflare (kept out of public/, which Vite tries to parse).
cp scripts/cf-headers dist/_headers
pnpm exec wrangler pages deploy dist --project-name voidindex-site --branch main --commit-dirty=true
