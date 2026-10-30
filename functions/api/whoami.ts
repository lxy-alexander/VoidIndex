/**
 * Tells the page whether the visitor is signed in as the owner.
 *
 * /api/* sits behind Cloudflare Access (email one-time code, owner's address
 * only), so a request only gets here after Access let it through and added
 * the signed-in email. Unauthenticated requests are redirected to the Access
 * login before they reach this function. The answer only decides whether the
 * edit buttons show; publishing still needs the encrypted GitHub token.
 */
export const onRequestGet = ({ request }: { request: Request }) => {
  const email = request.headers.get("cf-access-authenticated-user-email");
  return Response.json(
    { owner: Boolean(email), email },
    { status: email ? 200 : 401, headers: { "cache-control": "no-store" } }
  );
};
