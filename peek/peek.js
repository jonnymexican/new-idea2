// Read a checkpoint from the inside.
//
// The server does the model work and sends back one prompt's analysis; this
// file only draws it. Three views of the same window:
//
//   * the strip — every token, shaded by how many bits the model was wrong by
//     before it saw that token, so the bright ones are where it guessed;
//   * the matrix — one square per (reader, read) pair for the chosen layer and
//     head, so the causal triangle and the head's habits are both visible;
//   * the bars — the distribution the model currently holds for the next token.

const els = {
  text: document.getElementById("text"),
  go: document.getElementById("go"),
  examples: document.getElementById("examples"),
  status: document.getElementById("status"),
  view: document.getElementById("view"),
  strip: document.getElementById("strip"),
  matrix: document.getElementById("matrix"),
  layer: document.getElementById("layer"),
  gallery: document.getElementById("gallery"),
  focusNote: document.getElementById("focus-note"),
  entropy: document.getElementById("entropy"),
  bars: document.getElementById("bars"),
  eyebrow: document.getElementById("eyebrow"),
  modelMeta: document.getElementById("model-meta"),
};

const DEFAULT_TEXT = "ROMEO:\nBut soft, what light through yonder window breaks?";
const MAX_SURPRISE = 8;      // bits, where the ramp saturates
const CELL = 7;              // pixels per matrix square

const state = { data: null, focus: null, layer: null, head: "all", info: null, seq: 0 };

// --- drawing helpers -------------------------------------------------------

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[ch]));
}

// A token that is nothing but whitespace would be an invisible click target, so
// those — and only those — get visible glyphs.
function visible(text) {
  if (text.trim() !== "") return text;
  return text.replace(/ /g, "␣").replace(/\n/g, "⏎").replace(/\t/g, "⇥");
}

// Cool and dim when the model was sure, warm and bright when it was guessing.
function surpriseColour(bits) {
  if (bits === null || bits === undefined) return "transparent";
  const t = Math.min(bits / MAX_SURPRISE, 1);
  const hue = 205 - 200 * t;
  const sat = 45 + 45 * t;
  const light = 15 + 40 * t;
  return `hsl(${hue} ${sat}% ${light}%)`;
}

const pct = (p) => `${(p * 100).toFixed(p < 0.01 ? 1 : 0)}%`;

// --- the strip -------------------------------------------------------------

function renderStrip() {
  const { tokens } = state.data;
  els.strip.innerHTML = tokens
    .map((token, index) => {
      const bits = token.surprise;
      const title = bits === null
        ? "the first token: nothing came before it to predict it"
        : `${bits.toFixed(2)} bits of surprise`;
      const classes = ["tok"];
      if (index === state.focus) classes.push("focus");
      return `<span class="${classes.join(" ")}" data-i="${index}"` +
        ` style="background:${surpriseColour(bits)}" title="${escapeHtml(title)}"` +
        `>${escapeHtml(visible(token.text))}</span>`;
    })
    .join("");
  for (const span of els.strip.querySelectorAll(".tok")) {
    span.addEventListener("click", () => setFocus(Number(span.dataset.i)));
  }
  markRead();
}

// The tokens the focused one reads most: the matrix says how much, this says
// where to look in the text.
function markRead() {
  for (const span of els.strip.querySelectorAll(".tok")) span.classList.remove("read");
  const weights = weightsFor(state.focus);
  if (!weights) return;
  const spans = [...els.strip.querySelectorAll(".tok")];
  // Relative rather than absolute: one head can put 90% on a single token and
  // the mean of six rarely exceeds 15%, and both are "the places it read".
  const strongest = Math.max(...weights);
  const cut = Math.max(0.05, strongest * 0.4);
  weights.forEach((weight, index) => {
    if (weight >= cut && spans[index]) spans[index].classList.add("read");
  });
}

// --- the attention matrix --------------------------------------------------

// One cell of one head's matrix: a mean when `head` is null, since "any head at
// all" is what the gallery's first tile shows.
function attentionAt(grid, head, q, k, heads) {
  if (head === null) {
    let weight = 0;
    for (const row of grid) weight += row[q][k];
    return weight / (heads * 1000);
  }
  return grid[head][q][k] / 1000;
}

function weightsFor(focus) {
  const { attention, layers, heads } = state.data;
  if (!attention.length || focus === null) return null;
  const layer = Math.max(0, Math.min(state.layer ?? layers - 1, layers - 1));
  const row = attention[layer];
  if (state.head === "all") {
    // The mean of the heads: what "any head at all" looks like.
    const out = new Array(row[0][focus].length).fill(0);
    for (const head of row) for (let j = 0; j < out.length; j += 1) out[j] += head[focus][j];
    return out.map((w) => w / (heads * 1000));
  }
  const head = Math.min(Number(state.head), heads - 1);
  return row[head][focus].map((w) => w / 1000);
}

function renderMatrix() {
  const { attention, layers, heads, tokens } = state.data;
  const canvas = els.matrix;
  if (!attention.length) {
    canvas.hidden = true;
    return;
  }
  canvas.hidden = false;

  const n = tokens.length;
  const size = n * CELL;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = size * dpr;
  canvas.height = size * dpr;
  canvas.style.width = `${size}px`;
  canvas.style.height = `${size}px`;

  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, size, size);

  const layer = Math.max(0, Math.min(state.layer ?? layers - 1, layers - 1));
  const grid = attention[layer];
  const head = state.head === "all" ? null : Math.min(Number(state.head), heads - 1);

  for (let q = 0; q < n; q += 1) {
    for (let k = 0; k <= q; k += 1) {
      const weight = attentionAt(grid, head, q, k, heads);
      if (weight < 0.002) continue;
      // Square-root so the small attentions — which are most of them in the
      // mean of six heads — stay visible without washing the strong ones out.
      const alpha = Math.min(1, Math.pow(weight, 0.5));
      ctx.fillStyle = `rgba(224, 160, 74, ${alpha.toFixed(3)})`;
      ctx.fillRect(k * CELL, q * CELL, CELL - 0.5, CELL - 0.5);
    }
  }

  if (state.focus !== null) {
    // The row being read: everything above the diagonal in it is the future,
    // which is empty by construction and is the shape worth seeing.
    ctx.strokeStyle = "#e7e9ec";
    ctx.lineWidth = 1;
    ctx.strokeRect(0.5, state.focus * CELL + 0.5, n * CELL - 1, CELL - 1);
  }
}

function wireMatrix() {
  const focusAt = (event) => {
    if (!state.data || !state.data.attention.length) return null;
    const rect = els.matrix.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    if (x < 0 || y < 0) return null;
    const row = Math.floor(y / CELL);
    const column = Math.floor(x / CELL);
    if (row >= state.data.tokens.length || column > row) return null;
    return row;
  };
  els.matrix.addEventListener("mousemove", (event) => {
    const row = focusAt(event);
    if (row !== null) setFocus(row);
  });
  els.matrix.addEventListener("click", (event) => {
    const row = focusAt(event);
    if (row !== null) setFocus(row);
  });
}

function setFocus(index) {
  state.focus = index;
  for (const span of els.strip.querySelectorAll(".tok")) {
    span.classList.toggle("focus", Number(span.dataset.i) === index);
  }
  markRead();
  renderMatrix();
  renderFocusNote();
  queueGallery();
}

// --- the gallery -----------------------------------------------------------

// Every head of the layer at once, because a head's habit is a shape you compare
// by eye rather than a number you read, and showing them one at a time hides
// exactly that. Each tile backs at four pixels per token and is scaled down by
// CSS; the pictures are cached per head and layer, so moving the focus only
// re-blits an image and draws one line.

const TILE_PX = 4;
const TILE_COLOUR = [224, 160, 74];   // the same amber as the large matrix

const tiles = new Map();       // "all" | head index -> { button, canvas }
const tileImages = new Map();  // `${head}@${layer}` -> ImageData
let tileKey = "";              // the token and head counts the tiles were built for
let tileFrame = 0;

function ensureTiles(n, heads) {
  const key = `${n}|${heads}`;
  if (key === tileKey) return;
  tileKey = key;
  tiles.clear();
  tileImages.clear();
  els.gallery.innerHTML = "";
  // A model with one head has no use for a mean of one.
  const specs = heads > 1 ? ["all", ...Array.from({ length: heads }, (_, i) => i)] : [0];
  for (const head of specs) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "head-tile";
    button.title = head === "all"
      ? "the mean of every head"
      : `head ${head + 1} on its own`;
    const canvas = document.createElement("canvas");
    canvas.setAttribute("aria-hidden", "true");   // the label names the tile
    const label = document.createElement("span");
    label.textContent = head === "all" ? "mean" : `head ${head + 1}`;
    // What the picture's brightest spot is, in words: the habit named, not just
    // implied. It depends on which token is being read from, so it is set by
    // every paint rather than here.
    const read = document.createElement("small");
    read.className = "read";
    button.append(canvas, label, read);
    button.addEventListener("click", () => chooseHead(head));
    els.gallery.append(button);
    tiles.set(head, { button, canvas, read });
  }
}

function chooseHead(head) {
  state.head = head;
  paintGallery();
  markRead();
  renderMatrix();
  renderFocusNote();
}

function queueGallery() {
  // Hovering the large matrix moves the focus on every mousemove; one repaint
  // per frame is what the eye can see anyway.
  if (tileFrame) return;
  tileFrame = requestAnimationFrame(() => {
    tileFrame = 0;
    paintGallery();
  });
}

function paintGallery() {
  if (!state.data) return;
  const { tokens, heads, layers, attention } = state.data;
  if (!attention.length) return;
  const n = tokens.length;
  ensureTiles(n, heads);
  const layer = Math.max(0, Math.min(state.layer ?? layers - 1, layers - 1));
  for (const [head, tile] of tiles) {
    const index = head === "all" ? null : head;
    tile.button.setAttribute("aria-pressed",
      String(String(head) === String(state.head)));
    drawTile(tile.canvas, attention[layer], index, n, heads, layer);
    tile.read.textContent = readNote(attention[layer], index, state.focus, tokens,
      heads);
  }
}

// The one place a head puts most of the focused token's attention. The picture
// shows the whole row; this names the spot to look at, so "head 2" becomes
// "head 2 reads the M" without opening it.
function readNote(grid, head, focus, tokens, heads) {
  if (focus === null) return "";
  let best = 0;
  let at = 0;
  for (let k = 0; k <= focus; k += 1) {
    const weight = attentionAt(grid, head, focus, k, heads);
    if (weight > best) {
      best = weight;
      at = k;
    }
  }
  if (best < 0.01) return "nothing before it";
  const own = at === focus ? "itself" : `“${visible(tokens[at].text)}”`;
  return `${own} ${pct(best)}`;
}

function drawTile(canvas, grid, head, n, heads, layer) {
  const size = n * TILE_PX;
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  let image = tileImages.get(`${head}@${layer}`);
  if (!image || image.width !== size) {
    image = ctx.createImageData(size, size);
    const data = image.data;
    for (let q = 0; q < n; q += 1) {
      for (let k = 0; k <= q; k += 1) {
        const weight = attentionAt(grid, head, q, k, heads);
        if (weight < 0.002) continue;
        // The same square root the large matrix uses, so a faint attention is
        // still visible at a quarter of the size.
        const alpha = Math.round(Math.min(1, Math.pow(weight, 0.5)) * 255);
        for (let y = q * TILE_PX; y < (q + 1) * TILE_PX; y += 1) {
          for (let x = k * TILE_PX; x < (k + 1) * TILE_PX; x += 1) {
            const at = (y * size + x) * 4;
            data[at] = TILE_COLOUR[0];
            data[at + 1] = TILE_COLOUR[1];
            data[at + 2] = TILE_COLOUR[2];
            data[at + 3] = alpha;
          }
        }
      }
    }
    tileImages.set(`${head}@${layer}`, image);
  }
  ctx.putImageData(image, 0, 0);
  if (state.focus !== null && state.focus < n) {
    // The row being read, so the same token can be found in every head at once.
    ctx.fillStyle = "rgba(231, 233, 236, 0.75)";
    ctx.fillRect(0, state.focus * TILE_PX, size, Math.max(1, Math.round(TILE_PX / 2)));
  }
}

function renderFocusNote() {
  const weights = weightsFor(state.focus);
  if (!weights) {
    els.focusNote.textContent = "";
    return;
  }
  const { tokens } = state.data;
  const top = weights
    .map((weight, index) => ({ weight, index }))
    .sort((a, b) => b.weight - a.weight)
    .slice(0, 3)
    .filter((entry) => entry.weight > 0.01)
    .map(({ weight, index }) => {
      const own = index === state.focus ? "itself" : `“${visible(tokens[index].text)}”`;
      return `${own} ${pct(weight)}`;
    });
  const layer = Math.max(0, Math.min(state.layer ?? state.data.layers - 1,
    state.data.layers - 1)) + 1;
  const head = state.head === "all" ? "all heads" : `head ${Number(state.head) + 1}`;
  els.focusNote.textContent = top.length
    ? `layer ${layer}, ${head} — reads ${top.join(", ")}`
    : `layer ${layer}, ${head} — attends to nothing before it`;
}

// --- next-token bars -------------------------------------------------------

function renderNext() {
  const { next, entropy_bits: entropy } = state.data;
  const widest = next.length ? next[0].p : 1;
  els.bars.innerHTML = next
    .map((entry) => `
      <div class="bar">
        <span class="tok-text">${escapeHtml(visible(entry.text) || "∅")}</span>
        <span class="track"><span class="fill" style="width:${(entry.p / widest) * 100}%"></span></span>
        <span class="p">${pct(entry.p)}</span>
      </div>`)
    .join("");
  els.entropy.textContent = next.length
    ? `${entropy.toFixed(2)} bits undecided — as if ` +
      `${Math.pow(2, entropy).toFixed(1)} characters were equally likely.`
    : "";
}

// --- controls --------------------------------------------------------------

function renderControls() {
  const { layers } = state.data;
  const layer = Math.max(0, Math.min(state.layer ?? layers - 1, layers - 1));
  els.layer.innerHTML = Array.from({ length: layers }, (_, i) =>
    `<option value="${i}"${i === layer ? " selected" : ""}>${i + 1}</option>`).join("");
}

// --- the round trip --------------------------------------------------------

async function read(text) {
  const seq = ++state.seq;
  els.status.textContent = "reading…";
  els.status.hidden = false;
  try {
    const reply = await fetch("api/look", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!reply.ok) throw new Error(`HTTP ${reply.status}`);
    const data = await reply.json();
    if (seq !== state.seq) return;      // a newer edit already answered
    show(data, text);
  } catch (error) {
    if (seq === state.seq) {
      els.status.textContent = `could not read that: ${error?.message || error}`;
    }
  }
}

function show(data, text) {
  state.data = data;
  if (!data.tokens.length) {
    els.view.hidden = true;
    els.status.textContent = "nothing to read — type something first.";
    els.status.hidden = false;
    return;
  }
  // The first reading opens on the last layer and the last token, since that is
  // the position whose prediction is on screen; later edits keep the reader's
  // own choice of layer.
  if (state.layer === null) state.layer = data.layers - 1;
  state.focus = data.tokens.length - 1;
  tileImages.clear();          // the pictures belonged to the last reading
  els.view.hidden = false;
  renderStrip();
  renderControls();
  paintGallery();
  renderMatrix();
  renderFocusNote();
  renderNext();

  if (data.dropped) {
    els.status.textContent = `the window is the last ${data.window} tokens; ` +
      `${data.dropped} earlier one${data.dropped === 1 ? "" : "s"} are not read`;
    els.status.hidden = false;
  } else {
    els.status.hidden = true;
    els.status.textContent = "";
  }
  // The passage is the address, so a reading can be sent to someone.
  history.replaceState(null, "", `#${encodeURIComponent(text)}`);
}

function buildExamples(info) {
  const examples = [
    ["Shakespeare", DEFAULT_TEXT],
    info.sample ? ["What it wrote", info.sample] : null,
    ["Code", "def train(model, steps):\n    for step in range(steps):\n        logits, loss = model(x, y)"],
    ["A list", "- alpha\n- beta\n- gamma\n"],
  ].filter(Boolean);
  els.examples.innerHTML = "";
  for (const [label, text] of examples) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.textContent = label;
    chip.addEventListener("click", () => {
      els.text.value = text;
      read(text);
    });
    els.examples.append(chip);
  }
}

function renderMeta(info) {
  const millions = (info.params / 1e6).toFixed(2);
  els.eyebrow.textContent = `${info.checkpoint} · ${millions}M parameters`;
  els.modelMeta.textContent =
    `${info.layers} layers × ${info.heads} heads, ${info.block_size}-token window, ` +
    `${info.vocab}-token vocabulary. Everything runs in this process on the CPU; ` +
    `nothing is sent anywhere.`;
}

async function main() {
  wireInput();
  wireMatrix();
  let info;
  try {
    info = await fetch("api/info").then((reply) => reply.json());
  } catch (error) {
    els.status.textContent = `the server is not answering: ${error?.message || error}`;
    return;
  }
  state.info = info;
  renderMeta(info);
  buildExamples(info);
  let text = DEFAULT_TEXT;
  if (location.hash.length > 1) {
    try {
      text = decodeURIComponent(location.hash.slice(1));
    } catch {
      text = DEFAULT_TEXT;      // a fragment nobody encoded: not a passage
    }
  }
  els.text.value = text;
  read(text);
}

function wireInput() {
  els.go.addEventListener("click", () => read(els.text.value));
  let timer = null;
  els.text.addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(() => read(els.text.value), 600);
  });
  els.text.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      read(els.text.value);
    }
  });
  els.layer.addEventListener("change", () => {
    state.layer = Number(els.layer.value);
    markRead();
    paintGallery();
    renderMatrix();
    renderFocusNote();
  });
}

main();
