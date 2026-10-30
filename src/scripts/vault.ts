/**
 * Keeps the GitHub token for in-page editing encrypted at rest.
 *
 * The token is encrypted with AES-GCM under a key derived from a passphrase
 * (PBKDF2-SHA256, 600k iterations) and only the ciphertext is stored in
 * localStorage. After unlocking, the plain token lives in this module's
 * memory only — never in storage — and is wiped after 30 minutes idle or when
 * the tab closes. A copy of localStorage alone is useless without the
 * passphrase.
 */
const VAULT_KEY = "gh-token-vault";
const LEGACY_KEY = "gh-token";
const ITERATIONS = 600_000;
const IDLE_MS = 30 * 60 * 1000;

type Vault = { salt: string; iv: string; data: string };

let token: string | null = null;
let idleTimer: ReturnType<typeof setTimeout> | undefined;

const toB64 = (bytes: Uint8Array) => btoa(String.fromCharCode(...bytes));
const fromB64 = (b64: string): Uint8Array<ArrayBuffer> =>
  Uint8Array.from(atob(b64), c => c.charCodeAt(0));

async function deriveKey(
  passphrase: string,
  salt: Uint8Array<ArrayBuffer>
): Promise<CryptoKey> {
  const base = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(passphrase),
    "PBKDF2",
    false,
    ["deriveKey"]
  );
  return crypto.subtle.deriveKey(
    { name: "PBKDF2", salt, iterations: ITERATIONS, hash: "SHA-256" },
    base,
    { name: "AES-GCM", length: 256 },
    false,
    ["encrypt", "decrypt"]
  );
}

async function seal(secret: string, passphrase: string): Promise<Vault> {
  const salt = crypto.getRandomValues(new Uint8Array(16));
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const key = await deriveKey(passphrase, salt);
  const data = await crypto.subtle.encrypt(
    { name: "AES-GCM", iv },
    key,
    new TextEncoder().encode(secret)
  );
  return {
    salt: toB64(salt),
    iv: toB64(iv),
    data: toB64(new Uint8Array(data)),
  };
}

async function open(vault: Vault, passphrase: string): Promise<string | null> {
  try {
    const key = await deriveKey(passphrase, fromB64(vault.salt));
    const plain = await crypto.subtle.decrypt(
      { name: "AES-GCM", iv: fromB64(vault.iv) },
      key,
      fromB64(vault.data)
    );
    return new TextDecoder().decode(plain);
  } catch {
    return null; // wrong passphrase or tampered data
  }
}

function readVault(): Vault | null {
  try {
    return JSON.parse(localStorage.getItem(VAULT_KEY) ?? "null");
  } catch {
    return null;
  }
}

// Clear the unlocked token after a period without saves.
function touch(): void {
  clearTimeout(idleTimer);
  idleTimer = setTimeout(lock, IDLE_MS);
}

export function lock(): void {
  token = null;
  clearTimeout(idleTimer);
}

export function hasVault(): boolean {
  return readVault() !== null || localStorage.getItem(LEGACY_KEY) !== null;
}

export function forget(): void {
  lock();
  localStorage.removeItem(VAULT_KEY);
  localStorage.removeItem(LEGACY_KEY);
  sessionStorage.removeItem(LEGACY_KEY);
}

// A small modal with a password field (prompt() would show the text).
function ask(
  title: string,
  hint: string,
  actions: { ok: string; extra?: string } = { ok: "OK" }
): Promise<{ value: string; extra: boolean } | null> {
  return new Promise(resolve => {
    const dialog = document.createElement("dialog");
    dialog.className = "vault-dialog";
    dialog.innerHTML = `
      <form method="dialog">
        <p class="vault-title"></p>
        <p class="vault-hint"></p>
        <input type="password" autocomplete="off" spellcheck="false" />
        <div class="vault-actions">
          ${actions.extra ? `<button type="button" value="extra" class="vault-extra"></button>` : ""}
          <button type="button" value="cancel">Cancel</button>
          <button type="submit" value="ok" class="vault-ok"></button>
        </div>
      </form>`;
    dialog.querySelector(".vault-title")!.textContent = title;
    dialog.querySelector(".vault-hint")!.textContent = hint;
    dialog.querySelector(".vault-ok")!.textContent = actions.ok;
    const extra = dialog.querySelector<HTMLButtonElement>(".vault-extra");
    if (extra) extra.textContent = actions.extra!;
    const input = dialog.querySelector("input")!;
    const done = (result: { value: string; extra: boolean } | null) => {
      dialog.close();
      dialog.remove();
      resolve(result);
    };
    dialog.querySelector("form")!.addEventListener("submit", e => {
      e.preventDefault();
      done({ value: input.value, extra: false });
    });
    dialog
      .querySelector('[value="cancel"]')!
      .addEventListener("click", () => done(null));
    extra?.addEventListener("click", () => done({ value: "", extra: true }));
    dialog.addEventListener("cancel", () => done(null));
    document.body.append(dialog);
    dialog.showModal();
    input.focus();
  });
}

async function setUp(existing?: string): Promise<string | null> {
  const secret =
    existing ??
    (
      await ask(
        "GitHub token",
        "Fine-grained token with Contents read/write on lxy-alexander/VoidIndex only."
      )
    )?.value.trim();
  if (!secret) return null;
  const first = await ask(
    "Choose a passphrase",
    "Encrypts the token on this device. At least 8 characters; you'll enter it once per session."
  );
  if (!first || first.value.length < 8) return null;
  const again = await ask(
    "Repeat the passphrase",
    "Type the same passphrase again."
  );
  if (!again || again.value !== first.value) {
    alert("The passphrases didn't match. Nothing was saved.");
    return null;
  }
  localStorage.setItem(
    VAULT_KEY,
    JSON.stringify(await seal(secret, first.value))
  );
  // Drop any plain-text token left by older versions.
  localStorage.removeItem(LEGACY_KEY);
  sessionStorage.removeItem(LEGACY_KEY);
  return secret;
}

// Returns the token, asking to unlock (or set up) the vault when needed.
export async function getToken(): Promise<string | null> {
  if (token) {
    touch();
    return token;
  }
  const legacy =
    localStorage.getItem(LEGACY_KEY) ?? sessionStorage.getItem(LEGACY_KEY);
  const vault = readVault();
  if (!vault) {
    token = await setUp(legacy ?? undefined);
  } else {
    for (let attempt = 0; attempt < 3 && !token; attempt++) {
      const answer = await ask(
        attempt ? "Wrong passphrase, try again" : "Unlock editing",
        "Enter your passphrase to decrypt the GitHub token.",
        { ok: "Unlock", extra: "Reset token" }
      );
      if (!answer) return null;
      if (answer.extra) {
        forget();
        token = await setUp();
        break;
      }
      token = await open(vault, answer.value);
    }
  }
  if (token) touch();
  return token;
}
