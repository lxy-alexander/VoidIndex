#!/bin/bash
# The site now lives at https://voidindex.net (Cloudflare Pages). The old
# GitHub Pages workflow still runs on every push; when it does, publish only
# this redirect so lxy-alexander.github.io/VoidIndex/* goes to the new domain.
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf dist && mkdir -p dist
page='<!doctype html><meta charset="utf-8"><title>VoidIndex has moved</title>
<meta name="robots" content="noindex">
<script>location.replace("https://voidindex.net"+location.pathname.replace(/^\/VoidIndex/,"")+location.search+location.hash)</script>
<p>VoidIndex has moved to <a href="https://voidindex.net/">voidindex.net</a>.</p>'
printf '%s\n' "$page" > dist/index.html
printf '%s\n' "$page" > dist/404.html
