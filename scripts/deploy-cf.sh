#!/bin/bash
# Build the site and upload it straight to Cloudflare Pages; the new version
# is live as soon as the upload finishes.
#
# Auth: `pnpm exec wrangler login` once (stores an OAuth token in wrangler's
# own config, readable only by you). CI uses CLOUDFLARE_API_TOKEN instead.
set -euo pipefail
cd "$(dirname "$0")/.."

export CLOUDFLARE_ACCOUNT_ID="${CLOUDFLARE_ACCOUNT_ID:-1f1357c6bd74676420403146f10c3b75}"

pnpm run build
# Cache rules for Cloudflare (kept out of public/, which Vite tries to parse).
cp scripts/cf-headers dist/_headers
pnpm exec wrangler pages deploy dist --project-name voidindex --branch main --commit-dirty=true
