/**
 * Visiting /api/login on a new device signs in through Cloudflare Access,
 * then returns to the page given in ?next (same site only) or the home page.
 */
export const onRequestGet = ({ request }: { request: Request }) => {
  const url = new URL(request.url);
  const next = url.searchParams.get("next") ?? "/";
  const target = next.startsWith("/") && !next.startsWith("//") ? next : "/";
  return Response.redirect(new URL(target, url.origin).href, 302);
};
