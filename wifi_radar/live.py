#!/usr/bin/env python3
"""
A live motion view in a browser, served by the standard library.

    .venv/bin/python -m wifi_radar.live                    # the demo, on a loop
    .venv/bin/python -m wifi_radar.live --source sim:walk
    .venv/bin/python -m wifi_radar.live --source file:session.jsonl --fast
    .venv/bin/python -m wifi_radar.live --source serial:COM3 --rate 20

`--source serial:COM3` reads a board on a USB port. A CSI receiver gives the
same three panels as the simulator; a board with no radio (an Arduino Mega, say)
sends a profile of sensor channels instead, and the view labels the panels and
leaves the speed out, because a profile of light and distance has no Doppler
axis to report.

The demo scenario repeats every forty seconds, so the walk comes back if you
missed it. A session runs until stopped unless `--seconds` is given, and the
page says when a finite one has ended rather than freezing on its last frame.

Open http://127.0.0.1:8770/ and walk between the two radios. The page shows
three things, all of which answer a different question:

- the *trace*: the motion metric over the last few seconds, with the calibrated
  noise floor and the detection threshold drawn on it. This is the "did
  something move" view, and the two lines are what make a number readable;
- the *spectrum*: where the movement sits in frequency. A person crossing the
  room and a person shifting in a chair both raise the trace, and this is what
  tells them apart (see detect.py);
- the *waterfall*: the same motion per subcarrier, newest at the right. It shows
  whether the whole channel moved or one subcarrier glitched, which is the
  difference between a body and a bad packet.

A board with no radio gets one more panel: one row per analog pin — all sixteen
of a Mega's A0–A15, growing the canvas rather than shrinking the rows — with the
row's range (`swing`) and its current reading (`level`) printed beside it. Each
row is scaled to its own range so a pin idling at 512±2 is as legible as an LDR
at 800, and the swing column is what keeps that honest: the height of the line
is not the measurement, the number is. The pin currently standing out is
highlighted and named. That is the wiring view — plug a sensor into A3, wave at
it, and A3 is the row that answers.

The session runs in a background thread at the frame rate of the source, and the
page polls a JSON snapshot ten times a second. Only the newest waterfall column
is sent per poll rather than the whole grid, so a slow page cannot make the
server do more work than a fast one.
"""

import argparse
import json
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from wifi_radar import csi, detect

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8770

#: Frames of trace kept for the page to draw.
TRACE_FRAMES = 400


def doctor_check(port, timeout=30.0):
    """
    (verdict, message) from `python -m wifi_radar.doctor --json`, or unknown.

    A subprocess rather than an import: the doctor opens the port for its own
    bootloader probe and listen window, and has its own timeouts. If it cannot
    answer, the page says that instead of claiming the board is broken.
    """
    command = [sys.executable, "-m", "wifi_radar.doctor", "--port", port,
               "--json", "--scan-seconds", "0"]
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                encoding="utf-8", errors="replace",
                                timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as error:
        return "unknown", f"the doctor could not run: {error}"
    try:
        document = json.loads(result.stdout)
    except ValueError:
        text = (result.stdout or result.stderr or "").strip()
        return "unknown", (text[-300:] or "the doctor printed nothing")
    return document.get("verdict", "unknown"), document.get("message", "")


def busiest_channel(changes, ratio=2.0):
    """
    Index of the channel that stands out, or None when none does.

    Every channel changes a little all the time, so the largest of a set of
    near-equal numbers is noise and pointing at it would make a still room look
    wired. The one worth naming is the one that moved at least `ratio` times the
    median of the others — which is exactly the question a person with a
    breadboard is asking: which pin is doing something.
    """
    values = [float(v) for v in (changes or [])]
    if not values:
        return None
    top = max(range(len(values)), key=values.__getitem__)
    if values[top] <= 0:
        return None
    rest = sorted(v for i, v in enumerate(values) if i != top)
    if not rest:
        return None
    mid = len(rest) // 2
    baseline = (rest[mid] if len(rest) % 2
                else 0.5 * (rest[mid - 1] + rest[mid]))
    return top if values[top] >= ratio * baseline else None


class RadarSession:
    """
    A source and a detector, advanced in the background.

    The lock is held only while a snapshot is copied out, so a slow page
    request can never stall the detector, and the detector never waits on the
    network.
    """

    def __init__(self, source, realtime=True, calibrate_seconds=None,
                 highpass_seconds=detect.DEFAULT_HIGHPASS_SECONDS,
                 on_sigma=detect.DEFAULT_ON_SIGMA):
        self.source = csi.PacedSource(source, realtime=realtime)
        # Doppler is turned on per frame from the source: a serial one only
        # learns from its first line whether it is talking to a CSI receiver.
        self.detector = detect.MotionDetector(
            rate=self.source.rate, calibrate_seconds=calibrate_seconds,
            highpass_seconds=highpass_seconds, on_sigma=on_sigma)
        self.lock = threading.Lock()
        self.trace = deque(maxlen=TRACE_FRAMES)
        self.latest = None
        self.changes = []
        # The frame's own values, which the detector does not keep: the page
        # needs the level of each pin to draw one row per pin, not just how much
        # each one changed.
        self.levels = []
        self.error = None
        # Filled only when a serial source has failed: what the doctor makes of
        # the board, so the page can say why it is silent.
        self.diagnosis = None
        self.finished = False
        self._stop = threading.Event()
        self._thread = None

    @property
    def description(self):
        """
        Asked every time rather than captured once: a live source learns things
        as it runs — how many tones, which header revision, how fast frames are
        actually arriving — and a page that still says "waiting for packets"
        while drawing a trace is the kind of contradiction that makes a reader
        distrust the trace.
        """
        return self.source.describe()

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self):
        try:
            for frame in self.source:
                if self._stop.is_set():
                    break
                self.detector.doppler = self.source.doppler
                detection = self.detector.update(frame)
                with self.lock:
                    self.latest = detection
                    self.trace.append(detection.metric)
                    self.changes = list(detection.changes or [])
                    self.levels = list(frame.amplitudes())
        except Exception as error:                      # reported to the page
            self.error = f"{type(error).__name__}: {error}"
            self.diagnosis = self._diagnose()
        finally:
            # A view whose source has run out must say so: freezing on the last
            # frame looks exactly like a quiet room, and that is the one reading
            # of the page that would be wrong.
            self.finished = True

    def _serial_port(self):
        inner = getattr(self.source, "inner", self.source)
        return getattr(inner, "port", None)

    def _diagnose(self):
        """
        The doctor's verdict, for a board that is attached and not printing.

        An empty trace is the least useful reading of a silent serial source:
        the doctor already separates "no sketch", "wrong rate" and "no board",
        so the page repeats its answer rather than making the user run a second
        command to find out. The port is released first — only one process can
        have it open — and the doctor runs as a subprocess so a wedged adapter
        cannot stall the page.
        """
        port = self._serial_port()
        if not port:
            return None
        try:
            self.source.close()
        except Exception:
            pass
        verdict, message = doctor_check(port)
        return {"verdict": verdict, "message": message}

    def snapshot(self):
        with self.lock:
            latest = self.latest
            trace = list(self.trace)
            changes = list(self.changes)
            levels = list(self.levels)
            floor = self.detector.floor
            calibrating = self.detector.calibrating
            progress = self.detector.calibration_progress
            frames = self.detector.frames
        if latest is None:
            return {"source": self.description, "rate": self.source.rate,
                    "frames": 0, "calibrating": True, "progress": 0.0,
                    "metrics": [], "changes": [], "levels": [], "busiest": None,
                    "doppler": self.detector.doppler, "error": self.error,
                    "diagnosis": self.diagnosis}
        # No Doppler, no spectrum: for a channel profile the FFT of the metric
        # is the rate the profile changed at, and drawing it on a frequency axis
        # labelled in hertz invites reading it as a speed.
        freqs, powers = (detect.power_spectrum(self.trace, self.source.rate)
                         if self.detector.doppler and len(trace) >= self.detector.window
                         else ([], []))
        return {
            "source": self.description,
            "rate": self.source.rate,
            "frames": frames,
            "t": round(latest.t, 3),
            "calibrating": calibrating,
            "progress": round(progress, 3),
            "calibration": {"auto": self.detector.calibrate_auto,
                            "seconds": round(self.detector.calibration_seconds, 1),
                            "reason": self.detector.calibration_reason},
            "floor": floor,
            "threshold": (floor + self.detector.on_sigma * self.detector.scale
                          if floor and self.detector.scale else None),
            "presence": latest.presence,
            "motion": latest.motion,
            "z": latest.z,
            "peak_hz": latest.peak_hz,
            "band_power_db": latest.band_power,
            "agreement": latest.agreement,
            "amplitude": latest.amplitude,
            "metrics": [round(m, 6) for m in trace],
            "spectrum": {"freqs": [round(f, 3) for f in freqs],
                         "powers": [round(p, 9) for p in powers]},
            "changes": [round(c, 5) for c in changes],
            # Raw channel values, and the index of the one that stands out:
            # computed here rather than in the page so it can be tested.
            "levels": [round(v, 3) for v in levels],
            "busiest": (busiest_channel(changes)
                        if not self.detector.doppler else None),
            "subcarriers": len(changes),
            "doppler": self.detector.doppler,
            "finished": self.finished,
            "error": self.error,
            "diagnosis": self.diagnosis,
        }


class Handler(BaseHTTPRequestHandler):
    session = None
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        """Quiet by default: one line per poll would bury everything else."""

    def _send(self, body, content_type):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/state":
            body = json.dumps(self.session.snapshot()).encode("utf-8")
            self._send(body, "application/json")
        else:
            self.send_error(404, "no route")


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>WiFi radar</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; padding: 18px 22px 28px; background: #0f1115; color: #d8dde6;
         font: 14px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; }
  h1 { font-size: 15px; font-weight: 600; letter-spacing: .04em; margin: 0 0 2px; }
  .sub { color: #7e8798; font-size: 12px; margin-bottom: 16px; }
  .row { display: flex; gap: 14px; flex-wrap: wrap; }
  .card { background: #161a21; border: 1px solid #232833; border-radius: 10px;
          padding: 12px 14px; flex: 1 1 320px; min-width: 300px; }
  .card h2 { font-size: 12px; font-weight: 600; color: #8b94a6; margin: 0 0 8px;
             text-transform: uppercase; letter-spacing: .08em; }
  canvas { width: 100%; display: block; }
  #state { font-size: 22px; font-weight: 700; letter-spacing: .06em; }
  .quiet { color: #55606f; }
  .moving { color: #ffb454; }
  .present { color: #6ee7a0; }
  .numbers { display: grid; grid-template-columns: repeat(2, 1fr); gap: 2px 16px;
             color: #9aa4b4; font-size: 12px; margin-top: 8px; }
  .numbers b { color: #d8dde6; font-weight: 600; }
  .cal { color: #8ab4ff; }
  .err { color: #ff8080; }
</style>
</head>
<body>
  <h1 id="title">WiFi radar &mdash; channel state information</h1>
  <div class="sub" id="source">connecting…</div>
  <div class="row">
    <div class="card" style="flex: 2 1 520px">
      <h2>Motion</h2>
      <div id="state" class="quiet">IDLE</div>
      <canvas id="trace" height="150"></canvas>
      <div class="numbers">
        <div>metric <b id="metric">–</b></div>
        <div>z <b id="z">–</b></div>
        <div>floor <b id="floor">–</b></div>
        <div>agreement <b id="agree">–</b></div>
      </div>
    </div>
    <div class="card">
      <h2 id="specTitle">Doppler</h2>
      <canvas id="spectrum" height="120"></canvas>
      <div class="numbers">
        <div>peak <b id="peak">–</b></div>
        <div>in band <b id="band">–</b></div>
      </div>
    </div>
  </div>
  <div class="row" style="margin-top:14px">
    <div class="card" style="flex: 2 1 520px">
      <h2 id="wfTitle">Subcarriers &mdash; newest on the right</h2>
      <canvas id="waterfall" height="170"></canvas>
    </div>
    <div class="card">
      <h2>Session</h2>
      <div class="numbers" style="grid-template-columns: 1fr">
        <div>frames <b id="frames">0</b></div>
        <div>rate <b id="rate">–</b></div>
        <div><span id="subsLabel">subcarriers</span> <b id="subs">–</b></div>
        <div>time <b id="time">–</b></div>
      </div>
      <div class="sub" id="error" style="margin-top:10px"></div>
      <div class="sub" id="diagnosis" style="margin-top:6px"></div>
    </div>
  </div>
  <div class="row" style="margin-top:14px" id="pinsCard">
    <div class="card">
      <h2>Pins &mdash; level and swing per row</h2>
      <canvas id="pins" height="206"></canvas>
      <div class="sub" id="pinNote">waiting for samples</div>
    </div>
  </div>

<script>
const TRACE_POINTS = 400, WATERFALL_COLUMNS = 220, PIN_HISTORY = 160;
let trace = [], waterfall = [], waterfallRows = 0, pins = [];
// Which frame each history last took a column from. A poll faster than the
// source would otherwise append the same frame again, drawing a flat tail that
// looks like a sensor that went quiet rather than a page that stopped reading.
let waterfallAt = -1, pinsAt = -1;

// Give a canvas a bitmap that matches the box it is actually in. Called before
// every draw, because the first measurement can happen before the flex row has
// settled: a canvas keeps its intrinsic ratio when CSS width changes, so a width
// measured too early leaves the panel stretched to the wrong height forever.
function fit(target, cssHeight) {
  const ratio = window.devicePixelRatio || 1;
  const canvas = target.ctx.canvas;
  const cssWidth = Math.max(1, canvas.clientWidth || target.width || 600);
  const w = Math.round(cssWidth * ratio), h = Math.round(cssHeight * ratio);
  if (canvas.width === w && canvas.height === h) return;
  canvas.width = w; canvas.height = h;
  target.width = cssWidth; target.height = cssHeight;
  target.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
}
function setup(canvas, cssHeight) {
  const target = { ctx: canvas.getContext('2d'), width: 0,
                   height: cssHeight || canvas.height };
  fit(target, target.height);
  return target;
}
function clear(ctx, width, height) {
  ctx.fillStyle = '#11151b'; ctx.fillRect(0, 0, width, height);
}
function line(ctx, y, width, colour, dash) {
  ctx.save(); ctx.strokeStyle = colour; ctx.setLineDash(dash || []);
  ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke(); ctx.restore();
}

const traceCanvas = setup(document.getElementById('trace'), 150);
const spectrumCanvas = setup(document.getElementById('spectrum'), 120);
const waterfallCanvas = setup(document.getElementById('waterfall'), 170);
const pinsCanvas = setup(document.getElementById('pins'), 206);

function drawTrace(state) {
  const { ctx, width, height } = traceCanvas;
  clear(ctx, width, height);
  const threshold = state.threshold, floor = state.floor;
  // A generous ceiling: the point of the view is to see a walk dwarf the floor,
  // so the scale is the detection threshold rather than the maximum sample,
  // which would rescale the whole chart every time anything moved.
  const ceiling = Math.max((threshold || 1e-6) * 4, 4 * (floor || 1e-6), 1e-6);
  const y = value => height - Math.min(1, value / ceiling) * (height - 8) - 4;
  if (floor) line(ctx, y(floor), width, '#3a4351', [4, 4]);
  if (threshold) line(ctx, y(threshold), width, '#7a4a1e', [6, 4]);
  ctx.strokeStyle = '#6ee7a0'; ctx.lineWidth = 1.5; ctx.beginPath();
  trace.forEach((value, i) => {
    const x = i / (TRACE_POINTS - 1) * width;
    i ? ctx.lineTo(x, y(value)) : ctx.moveTo(x, y(value));
  });
  ctx.stroke();
}

function drawSpectrum(state) {
  const { ctx, width, height } = spectrumCanvas;
  clear(ctx, width, height);
  const spec = state.spectrum;
  if (!spec || !spec.powers || !spec.powers.length) return;
  const freqs = spec.freqs, powers = spec.powers;
  let top = 0, lo = 0, hi = powers.length - 1;
  for (let i = 0; i < freqs.length; i++) {         // draw 0.2–8 Hz only
    if (freqs[i] >= 0.2 && lo === 0) lo = i;
    if (freqs[i] <= 8.0) hi = i;
    top = Math.max(top, powers[i]);
  }
  const span = Math.max(1, hi - lo), bars = Math.min(64, span);
  for (let b = 0; b < bars; b++) {
    const from = lo + Math.floor(b * span / bars);
    const to = lo + Math.max(from + 1, Math.floor((b + 1) * span / bars));
    let peak = 0;
    for (let i = from; i < to && i < powers.length; i++) peak = Math.max(peak, powers[i]);
    const h = top > 0 ? peak / top * (height - 18) : 0;
    const x = b / bars * width;
    ctx.fillStyle = '#8ab4ff';
    ctx.fillRect(x + 1, height - 14 - h, width / bars - 2, h);
  }
  // Mark where the peak actually is, so the bars can be read as a frequency.
  const peak = state.peak_hz;
  if (peak && peak >= 0.2 && peak <= 8.0) {
    const x = (peak - 0.2) / (8.0 - 0.2) * width;
    ctx.strokeStyle = '#ffb454'; ctx.beginPath();
    ctx.moveTo(x, 0); ctx.lineTo(x, height - 14); ctx.stroke();
  }
}

function drawWaterfall(state) {
  const { ctx, width, height } = waterfallCanvas;
  const column = state.changes || [];
  if (column.length && state.frames !== waterfallAt) {
    waterfallAt = state.frames;
    if (waterfallRows !== column.length) { waterfall = []; waterfallRows = column.length; }
    waterfall.push(column);
    while (waterfall.length > WATERFALL_COLUMNS) waterfall.shift();
  }
  clear(ctx, width, height);
  if (!waterfall.length || !waterfallRows) return;
  let top = 0;
  waterfall.forEach(col => col.forEach(v => { top = Math.max(top, v); }));
  if (top <= 0) return;
  const columns = waterfall.length;
  const cellW = width / WATERFALL_COLUMNS, cellH = height / waterfallRows;
  for (let c = 0; c < columns; c++) {
    const offset = columns - c;                    // newest at the right edge
    for (let r = 0; r < waterfallRows; r++) {
      const v = Math.min(1, waterfall[c][r] / top);
      if (v < 0.06) continue;
      // Dark blue to warm white: brightness is how much this subcarrier moved.
      const level = Math.floor(v * 255);
      ctx.fillStyle = `rgb(${Math.floor(level * 0.9)},${Math.floor(level * 0.75)},${180 + Math.floor(v * 75)})`;
      ctx.fillRect(width - offset * cellW, r * cellH, Math.max(1, cellW), Math.max(1, cellH));
    }
  }
}

// Pin panel geometry. The row height is what gives way as channels are added:
// 16 pins (a Mega's A0–A15) must stay as legible as 4, so the canvas grows
// rather than squeezing the font, and every row keeps its own printed scale.
const PIN_PAD_TOP = 14, PIN_LABEL_W = 36, PIN_SWING_W = 40, PIN_VALUE_W = 44;

function pinLayout(n) {
  const rows = Math.max(1, n);
  const rowH = Math.max(14, Math.min(26, Math.floor(320 / rows)));
  return { rowH, height: PIN_PAD_TOP + rows * rowH };
}

function fitPins(height) {
  fit(pinsCanvas, height);
}

function drawPins(state) {
  const card = document.getElementById('pinsCard');
  const levels = state.levels || [];
  // Only a channel node has pins to point at; a CSI frame's rows are tones,
  // and the waterfall already shows which of those moved.
  if (state.doppler || !levels.length) {
    card.style.display = 'none';
    pins = []; pinsAt = -1;
    return;
  }
  card.style.display = '';
  if (pins.length && pins[0].length !== levels.length) { pins = []; pinsAt = -1; }
  if (state.frames !== pinsAt) {
    pinsAt = state.frames;
    pins.push(levels);
    while (pins.length > PIN_HISTORY) pins.shift();
  }

  const n = levels.length, layout = pinLayout(n), busiest = state.busiest;
  fitPins(layout.height);
  const { ctx, width } = pinsCanvas;
  const height = layout.height, rowH = layout.rowH;
  clear(ctx, width, height);

  const x0 = PIN_LABEL_W, x1 = width - PIN_SWING_W - PIN_VALUE_W;
  const swingX = width - PIN_VALUE_W - 6, valueX = width - 3;
  ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
  ctx.textBaseline = 'middle';
  // The column names are what make the numbers a scale: "swing" is the range
  // this row has covered, in the same units as "level".
  ctx.fillStyle = '#55606f';
  ctx.textAlign = 'right';
  ctx.fillText('pin', PIN_LABEL_W - 8, PIN_PAD_TOP / 2 + 1);
  ctx.fillText('swing', swingX, PIN_PAD_TOP / 2 + 1);
  ctx.fillText('level', valueX, PIN_PAD_TOP / 2 + 1);

  const spans = [];
  for (let r = 0; r < n; r++) {
    const top = PIN_PAD_TOP + r * rowH;
    const series = pins.map(column => column[r]);
    let lo = Math.min(...series), hi = Math.max(...series);
    const swing = Math.round(hi - lo);
    spans.push(swing);
    if (hi - lo < 1e-9) { lo -= 0.5; hi += 0.5; }   // a dead-flat pin still draws
    const y = v => top + rowH - 3 - (v - lo) / (hi - lo) * (rowH - 6);
    if (r === busiest) {
      ctx.fillStyle = 'rgba(255,180,84,0.10)';
      ctx.fillRect(0, top, width, rowH);
    }
    if (rowH >= 16) {
      ctx.fillStyle = 'rgba(255,255,255,0.045)';
      ctx.fillRect(PIN_LABEL_W, top + rowH - 1, width - PIN_LABEL_W, 1);
    }
    ctx.strokeStyle = r === busiest ? '#ffb454' : '#5d7fa8';
    ctx.lineWidth = 1;
    ctx.beginPath();
    // Stretched across the full plot while the history fills, then scrolling
    // once it is full: a line that grows from the left for eight seconds reads
    // as a truncated chart, which is the one thing a wiring view must not do.
    const columns = Math.max(1, series.length - 1);
    series.forEach((v, i) => {
      const x = x0 + i / columns * (x1 - x0);
      i ? ctx.lineTo(x, y(v)) : ctx.moveTo(x, y(v));
    });
    ctx.stroke();
    // Right-aligned so A0..A15 form a column instead of a ragged edge, and none
    // of them runs into the trace.
    ctx.textAlign = 'right';
    ctx.fillStyle = r === busiest ? '#ffb454' : '#7e8798';
    ctx.fillText('A' + r, PIN_LABEL_W - 8, top + rowH / 2);
    ctx.textAlign = 'right';
    ctx.fillStyle = r === busiest ? '#ffb454' : '#6f7a8a';
    ctx.fillText(String(swing), swingX, top + rowH / 2);
    ctx.fillStyle = '#9aa4b4';
    ctx.fillText(String(Math.round(levels[r])), valueX, top + rowH / 2);
  }
  ctx.textAlign = 'left';
  const changes = state.changes || [];
  let note;
  if (busiest === null || busiest === undefined || !changes.length) {
    // "Nothing is wired" and "several things are wired at once" are different
    // answers, and a single sentence for both would mislead one of them.
    note = state.motion
      ? 'several pins are moving — none stands out'
      : 'no pin stands out — wire one and disturb it';
  } else {
    // Adc counts rather than the detector's normalised metric: this panel is
    // read by someone holding a jumper wire, not by someone tuning a threshold.
    note = `responding: A${busiest} — level ${Math.round(levels[busiest])}, `
         + `swing ${spans[busiest]} (change ${Number(changes[busiest]).toFixed(3)})`;
  }
  document.getElementById('pinNote').textContent = note;
}

function number(id, value, digits) {
  document.getElementById(id).textContent =
    (value === null || value === undefined) ? '–' : Number(value).toFixed(digits);
}

async function poll() {
  try {
    const response = await fetch('/api/state');
    const state = await response.json();
    trace = state.metrics || [];

    // Two kinds of source share this page, and the panels say which one is
    // behind them rather than labelling a light sensor's output "Doppler".
    document.getElementById('specTitle').textContent =
      state.doppler ? 'Doppler' : 'Doppler — not on this source';
    document.getElementById('wfTitle').textContent =
      state.doppler ? 'Subcarriers — newest on the right'
                    : 'Channels — newest on the right';
    document.getElementById('subsLabel').textContent =
      state.doppler ? 'subcarriers' : 'channels';
    document.getElementById('title').textContent =
      state.doppler ? 'WiFi radar — channel state information'
                    : 'Radar — sensor channel profile';

    const label = document.getElementById('state');
    if (state.error) { label.textContent = 'ERROR'; label.className = 'err'; }
    else if (state.finished) { label.textContent = 'SESSION ENDED'; label.className = 'quiet'; }
    else if (state.calibrating) {
      const cal = state.calibration || {};
      // A percentage of an unknown-length window is a guess; when the length is
      // automatic, show how long it has been measuring instead.
      label.textContent = cal.auto
        ? `CALIBRATING ${Math.round(cal.seconds || 0)}s`
        : `CALIBRATING ${Math.round((state.progress || 0) * 100)}%`;
      label.className = 'cal';
    } else if (state.presence) { label.textContent = 'PRESENT'; label.className = 'present'; }
    else if (state.motion) { label.textContent = 'MOVING'; label.className = 'moving'; }
    else { label.textContent = 'IDLE'; label.className = 'quiet'; }

    document.getElementById('source').textContent =
      `${state.source} — ${state.frames} frames`;
    number('metric', state.metrics && state.metrics.length
      ? state.metrics[state.metrics.length - 1] : null, 5);
    number('z', state.z, 1);
    number('floor', state.floor, 5);
    number('agree', state.agreement, 2);
    number('peak', state.peak_hz, 2);
    number('band', state.band_power_db, 1);
    document.getElementById('frames').textContent = state.frames;
    number('rate', state.rate, 0);
    document.getElementById('subs').textContent = state.subcarriers || '–';
    number('time', state.t, 1);
    document.getElementById('error').textContent = state.error || '';
    const diagnosis = state.diagnosis;
    document.getElementById('diagnosis').textContent = diagnosis
      ? `doctor says: ${diagnosis.message}` : '';

    fit(traceCanvas, 150);
    if (state.doppler) fit(spectrumCanvas, 120);
    fit(waterfallCanvas, 170);
    drawTrace(state); if (state.doppler) drawSpectrum(state);
    drawWaterfall(state); drawPins(state);
  } catch (error) {
    document.getElementById('error').textContent = String(error);
  }
  setTimeout(poll, 100);
}
addEventListener('resize', () => { location.reload(); });
poll();
</script>
</body>
</html>
"""


def build_server(session, host=DEFAULT_HOST, port=DEFAULT_PORT):
    handler = type("RadarHandler", (Handler,), {"session": session})
    return ThreadingHTTPServer((host, port), handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="sim:demo",
                        help="sim:<scenario>, file:<path>, udp:[host:]port "
                             "(nexmon CSI) or serial:<port>")
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="session length in seconds; 0 (the default) runs "
                             "until stopped")
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--fast", action="store_true",
                        help="run the source as fast as the CPU allows")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--calibrate", type=detect.parse_calibrate,
                        default=None, metavar="SECONDS|auto",
                        help="how long to measure the still floor before "
                             "deciding anything; auto (the default) keeps "
                             "measuring while the input drifts and stops once "
                             "the floor has settled, a number fixes the window "
                             "at that many seconds")
    parser.add_argument("--highpass", type=float,
                        default=detect.DEFAULT_HIGHPASS_SECONDS,
                        metavar="SECONDS",
                        help="subtract a baseline that follows the input with "
                             "this time constant, so drift slower than it is "
                             "ignored (default %(default)s); 0 turns it off")
    parser.add_argument("--on-sigma", type=float,
                        default=detect.DEFAULT_ON_SIGMA, metavar="SIGMA",
                        help="how many calibrated sigmas of change count as "
                             "motion (default %(default)s); thresholds.py "
                             "measures the right number from a recording")
    args = parser.parse_args(argv)

    if args.highpass < 0:
        print("error: --highpass cannot be negative; 0 turns the filter off.")
        return 2

    if args.calibrate is not None and args.calibrate <= 0:
        print("error: --calibrate must be more than 0 seconds (or auto). The "
              "detector needs a still window to measure the noise floor from; "
              "without one every frame would be judged against nothing.")
        return 2

    if args.on_sigma <= 0:
        print("error: --on-sigma must be more than 0; it multiplies the "
              "calibrated spread, and a threshold of zero would call every "
              "frame movement.")
        return 2

    try:
        source = csi.open_source(args.source,
                                 seconds=args.seconds or None, rate=args.rate)
    except (ValueError, OSError) as error:
        # `serial:auto` with nothing plugged in is an OSError, and the message
        # says what to do; the traceback would only say where.
        print(f"error: {error}")
        return 2

    session = RadarSession(source, realtime=not args.fast,
                           calibrate_seconds=args.calibrate,
                           highpass_seconds=args.highpass,
                           on_sigma=args.on_sigma)
    server = build_server(session, args.host, args.port)
    session.start()
    # Flushed so the URL is visible immediately when output is redirected to a
    # file, which is how this is usually started.
    print(f"source  : {session.description}", flush=True)
    print(f"view    : http://{args.host}:{args.port}/", flush=True)
    print("ctrl-c  : stop", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        session.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
