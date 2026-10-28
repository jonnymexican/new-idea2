// Verify a content credential in the browser, the way a client does.
//
// The lab's Python arm answers "what is in these bytes" with c2pa-python; this
// page answers what a person on the other end sees, using the same CAI SDK a
// viewer application would embed. Two consequences worth knowing:
//
//   * it is the browser's trust list, not the lab's, that decides `Trusted`, so
//     a self-issued test certificate reads Valid with signingCredential.untrusted;
//   * the verifier itself (about 9 MB of WebAssembly) is fetched from a pinned
//     CDN URL on first use, because this repo has no JavaScript build step.

const SDK = "https://cdn.jsdelivr.net/npm/@contentauth/c2pa-web@0.15.4";

const TRUSTED_TYPES = /trainedAlgorithmicMedia|compositeWithTrainedAlgorithmicMedia|algorithmicMedia/i;

const els = {
  drop: document.getElementById("drop"),
  file: document.getElementById("file"),
  pick: document.getElementById("pick"),
  loading: document.getElementById("loading"),
  samples: document.getElementById("samples"),
  sampleList: document.getElementById("sample-list"),
  notice: document.getElementById("notice"),
  home: document.getElementById("home"),
  results: document.getElementById("results"),
};

let sdkPromise = null;

function loadSdk() {
  if (!sdkPromise) {
    els.loading.hidden = false;
    sdkPromise = (async () => {
      const { createC2pa, Reader } = await import(`${SDK}/+esm`);
      const c2pa = await createC2pa({ wasmSrc: `${SDK}/dist/resources/c2pa_bg.wasm` });
      return { c2pa, Reader };
    })().finally(() => { els.loading.hidden = true; });
  }
  return sdkPromise;
}

const bytes = (n) => `${new Intl.NumberFormat("en").format(n)} bytes`;

function actionsOf(manifest) {
  const out = [];
  for (const assertion of manifest?.assertions ?? []) {
    if (!String(assertion.label || "").startsWith("c2pa.actions")) continue;
    const data = assertion.data;
    const list = Array.isArray(data) ? data : data?.actions ?? [];
    for (const action of list) out.push(action);
  }
  return out;
}

async function inspect(blob, type) {
  const { c2pa, Reader } = await loadSdk();
  const reader = await Reader.fromBlob(c2pa, type || undefined, blob);
  if (!reader) return { present: false };
  try {
    const store = await reader.manifestStore();
    const manifest = store.active_manifest ? store.manifests?.[store.active_manifest] : undefined;
    const results = store.validation_results?.activeManifest;
    return {
      present: true,
      state: store.validation_state ?? null,
      manifestId: store.active_manifest ?? null,
      manifest,
      failures: results?.failure ?? [],
      successes: results?.success ?? [],
      informational: results?.informational ?? [],
      actions: actionsOf(manifest ?? {}),
      ingredients: (manifest?.ingredients ?? []).length,
    };
  } finally {
    await reader.free();
  }
}

function badgeFor(state) {
  const text = state || "no state";
  const cls = state === "Trusted" ? "trusted"
    : state === "Valid" ? "valid"
    : state === "Invalid" ? "invalid" : "";
  return `<span class="badge ${cls}">${escapeHtml(text)}</span>`;
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[ch]));
}

// A card names the rendition it came from, so following the same link twice
// replaces the card instead of stacking a second copy of it.
function keyed(key) {
  return key ? ` data-sample="${escapeHtml(key)}"` : "";
}

function card(name, blob, outcome, key) {
  const head = `
    <header>
      <h2>${escapeHtml(name)}</h2>
      ${outcome.present ? badgeFor(outcome.state) : '<span class="badge">no credential</span>'}
    </header>
    <p class="meta">${bytes(blob.size)} · ${escapeHtml(blob.type || "unknown type")}</p>`;

  if (!outcome.present) {
    return `<article class="card"${keyed(key)}>${head}
      <p>This file carries no content credential. Either it never had one, or
      the service that served it re-encoded the image and left the manifest
      behind — which is the failure this lab measures.</p>
    </article>`;
  }

  const signer = outcome.manifest?.signature_info ?? {};
  const generator = (outcome.manifest?.claim_generator_info ?? [])
    .map((g) => (g.version ? `${g.name} ${g.version}` : g.name))
    .join(", ");
  const ai = outcome.actions.filter((a) => TRUSTED_TYPES.test(String(a.digitalSourceType || "")));

  const rows = [
    ["signer", signer.issuer || signer.common_name || "—"],
    ["certificate", signer.common_name && signer.issuer ? signer.common_name : ""],
    ["algorithm", signer.alg || ""],
    ["signed at", signer.time || ""],
    ["generator", generator],
    ["manifest", outcome.manifestId || ""],
    ["assertions", (outcome.manifest?.assertions ?? []).map((a) => a.label).join(", ")],
    ["ingredients", String(outcome.ingredients)],
  ].filter(([, value]) => value);

  const findings = outcome.failures.length
    ? outcome.failures.map((f) => `<li>${escapeHtml(f.code || "failure")}${
        f.explanation ? ` — ${escapeHtml(f.explanation)}` : ""}</li>`).join("")
    : '<li class="good">no validation failures reported</li>';

  const aiNote = ai.length
    ? `<p class="ai"><strong>Claims AI-generated media.</strong>
        ${ai.map((a) => escapeHtml(String(a.digitalSourceType).split("/").pop()))
          .filter((v, i, all) => all.indexOf(v) === i)
          .join(", ")} on a stated action.</p>`
    : "";

  return `<article class="card"${keyed(key)}>${head}
    ${aiNote}
    <dl>${rows.map(([k, v]) => `<dt>${escapeHtml(k)}</dt><dd>${escapeHtml(v)}</dd>`).join("")}</dl>
    <ul class="findings">${findings}</ul>
  </article>`;
}

async function show(name, blob, key) {
  const previous = key && document.querySelector(`[data-sample="${CSS.escape(key)}"]`);
  if (previous) previous.remove();

  const placeholder = document.createElement("article");
  placeholder.className = "card";
  if (key) placeholder.dataset.sample = key;
  placeholder.innerHTML = `<header><h2>${escapeHtml(name)}</h2></header>
    <p class="meta">reading ${bytes(blob.size)}…</p>`;
  els.results.prepend(placeholder);

  try {
    const outcome = await inspect(blob, blob.type);
    placeholder.outerHTML = card(name, blob, outcome, key);
  } catch (error) {
    placeholder.outerHTML = `<article class="card"${keyed(key)}>
      <header><h2>${escapeHtml(name)}</h2><span class="badge invalid">error</span></header>
      <ul class="findings"><li>${escapeHtml(error?.message || String(error))}</li></ul>
    </article>`;
  }
}

async function showAll(files) {
  // Sequential on purpose: each card is a WASM call, and a queue keeps memory
  // flat when a whole run directory is dropped at once.
  for (const file of files) await show(file.name, file);
}

function wireDrop() {
  els.pick.addEventListener("click", () => els.file.click());
  els.file.addEventListener("change", () => {
    showAll([...els.file.files]);
    els.file.value = "";
  });
  for (const type of ["dragenter", "dragover"]) {
    els.drop.addEventListener(type, (event) => {
      event.preventDefault();
      els.drop.classList.add("over");
    });
  }
  for (const type of ["dragleave", "drop"]) {
    els.drop.addEventListener(type, (event) => {
      event.preventDefault();
      els.drop.classList.remove("over");
    });
  }
  els.drop.addEventListener("drop", (event) => {
    const items = [...(event.dataTransfer?.files ?? [])];
    if (items.length) showAll(items);
  });
}

// --- the catalogue, and the address that names one of its files -------------

let samples = [];

function backTo(home) {
  // Only the published page has a trend page above it. The local server serves
  // this page alone, so it sends no `home` and the link stays hidden rather
  // than pointing at the page you are already on.
  if (!home?.href) return;
  els.home.innerHTML =
    `← <a href="${escapeHtml(home.href)}">${escapeHtml(home.label || "the trend page")}</a>`;
  els.home.hidden = false;
}

function say(text) {
  els.notice.hidden = !text;
  els.notice.textContent = text || "";
}

function decodeHash() {
  // The fragment is a rendition's catalogue URL. A browser percent-encodes
  // what goes into a hash -- a duplicate run is `live (2)` -- so decode before
  // comparing, or the link would not match its own chip.
  try {
    return decodeURIComponent(location.hash.slice(1));
  } catch {
    return location.hash.slice(1);
  }
}

function linkedSample() {
  const wanted = decodeHash();
  return wanted ? samples.find((file) => file.url === wanted) || null : null;
}

function open(file) {
  // Setting the fragment is what opens it: `hashchange` does the work, so the
  // address bar always names what is on screen and the URL is worth sending.
  if (decodeHash() === file.url) openSample(file);
  else location.hash = file.url;
}

async function openSample(file) {
  say("");
  try {
    const reply = await fetch(file.url);
    if (!reply.ok) throw new Error(`HTTP ${reply.status}`);
    await show(`${file.run} · ${file.label}`, await reply.blob(), file.url);
  } catch (error) {
    say(`could not read ${file.label}: ${error?.message || error}`);
  }
}

async function followHash() {
  const file = linkedSample();
  if (file) return openSample(file);
  const wanted = decodeHash();
  say(wanted ? `this page has no rendition at “${wanted}”` : "");
}

async function loadSamples() {
  let payload;
  try {
    const reply = await fetch("samples.json", { cache: "no-store" });
    if (!reply.ok) return;
    payload = await reply.json();
  } catch {
    return; // Served by a plain static server: there is nothing to list.
  }

  backTo(payload.home);

  samples = (payload.runs ?? []).flatMap((run) =>
    (run.files ?? []).map((file) => ({ ...file, run: run.name })));
  if (!samples.length) return;

  els.samples.hidden = false;
  for (const file of samples) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.innerHTML = `${escapeHtml(file.label)} <small>${escapeHtml(file.run)}</small>`;
    chip.addEventListener("click", () => open(file));
    els.sampleList.append(chip);
  }
}

wireDrop();
loadSamples().then(followHash);
// A shared link arrives as a fragment, and so does Back: both land here.
window.addEventListener("hashchange", followHash);
