#!/bin/bash
# Build the site and upload it straight to Cloudflare Pages (live in seconds).
# Needs CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID, read from
# ~/.config/voidindex/cloudflare.env (keep that file private: chmod 600).
set -euo pipefail
cd "$(dirname "$0")/.."

ENV_FILE="$HOME/.config/voidindex/cloudflare.env"
if [ -f "$ENV_FILE" ]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi
: "${CLOUDFLARE_API_TOKEN:?Set CLOUDFLARE_API_TOKEN in $ENV_FILE}"
: "${CLOUDFLARE_ACCOUNT_ID:?Set CLOUDFLARE_ACCOUNT_ID in $ENV_FILE}"
export CLOUDFLARE_API_TOKEN CLOUDFLARE_ACCOUNT_ID

pnpm run build
# Cache rules for Cloudflare (kept out of public/, which Vite tries to parse).
cp scripts/cf-headers dist/_headers
pnpm exec wrangler pages deploy dist --project-name voidindex --branch main --commit-dirty=true
