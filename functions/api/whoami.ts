/**
 * Tells the page whether the visitor is signed in as the owner.
 *
 * /api/* sits behind Cloudflare Access (owner's address only, 1-month
 * session), which redirects anyone not signed in to its login page. For
 * those let through, Access attaches a signed token; checking its signature,
 * audience, expiry and email here keeps the answer honest even if a request
 * reaches the function some other way. The answer only decides whether the
 * edit buttons show; publishing still needs the encrypted GitHub token.
 */
const TEAM = "https://autumn-fog-f0c1.cloudflareaccess.com";
// The "VoidIndex owner" Access application's audience tag.
const AUD = "851fdb316d4a431043036e61250a415cee29d96efee152121d1b75dbe06e1b52";
const OWNER = "lxy_alexander@outlook.com";

type Jwk = JsonWebKey & { kid: string };

const b64url = (s: string) =>
  Uint8Array.from(atob(s.replace(/-/g, "+").replace(/_/g, "/")), c =>
    c.charCodeAt(0)
  );

function readToken(request: Request): string | null {
  const header = request.headers.get("cf-access-jwt-assertion");
  if (header) return header;
  const cookie = request.headers.get("cookie") ?? "";
  return cookie.match(/(?:^|;\s*)CF_Authorization=([^;]+)/)?.[1] ?? null;
}

async function verifiedEmail(token: string): Promise<string | null> {
  const [head, body, sig] = token.split(".");
  if (!head || !body || !sig) return null;
  const { kid } = JSON.parse(new TextDecoder().decode(b64url(head)));
  const { keys } = (await fetch(`${TEAM}/cdn-cgi/access/certs`).then(r =>
    r.json()
  )) as { keys: Jwk[] };
  const jwk = keys.find(k => k.kid === kid);
  if (!jwk) return null;
  const key = await crypto.subtle.importKey(
    "jwk",
    jwk,
    { name: "RSASSA-PKCS1-v1_5", hash: "SHA-256" },
    false,
    ["verify"]
  );
  const ok = await crypto.subtle.verify(
    "RSASSA-PKCS1-v1_5",
    key,
    b64url(sig),
    new TextEncoder().encode(`${head}.${body}`)
  );
  if (!ok) return null;
  const claims = JSON.parse(new TextDecoder().decode(b64url(body)));
  const aud: string[] = [claims.aud].flat();
  if (!aud.includes(AUD) || claims.iss !== TEAM) return null;
  if (typeof claims.exp !== "number" || claims.exp * 1000 < Date.now()) {
    return null;
  }
  return typeof claims.email === "string" ? claims.email : null;
}

export const onRequestGet = async ({ request }: { request: Request }) => {
  const token = readToken(request);
  const email = token ? await verifiedEmail(token).catch(() => null) : null;
  const owner = email?.toLowerCase() === OWNER;
  return Response.json(
    { owner },
    { status: owner ? 200 : 401, headers: { "cache-control": "no-store" } }
  );
};
