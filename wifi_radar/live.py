#!/usr/bin/env python3
"""
A live motion view in a browser, served by the standard library.

    .venv/bin/python -m wifi_radar.live                    # the demo, on a loop
    .venv/bin/python -m wifi_radar.live --source sim:walk
    .venv/bin/python -m wifi_radar.live --source file:session.jsonl --fast

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

The session runs in a background thread at the frame rate of the source, and the
page polls a JSON snapshot ten times a second. Only the newest waterfall column
is sent per poll rather than the whole grid, so a slow page cannot make the
server do more work than a fast one.
"""

import argparse
import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from wifi_radar import csi, detect

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8770

#: Frames of trace kept for the page to draw.
TRACE_FRAMES = 400


class RadarSession:
    """
    A source and a detector, advanced in the background.

    The lock is held only while a snapshot is copied out, so a slow page
    request can never stall the detector, and the detector never waits on the
    network.
    """

    def __init__(self, source, realtime=True):
        self.source = csi.PacedSource(source, realtime=realtime)
        self.detector = detect.MotionDetector(rate=self.source.rate)
        self.description = self.source.describe()
        self.lock = threading.Lock()
        self.trace = deque(maxlen=TRACE_FRAMES)
        self.latest = None
        self.changes = []
        self.error = None
        self.finished = False
        self._stop = threading.Event()
        self._thread = None

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
                detection = self.detector.update(frame)
                with self.lock:
                    self.latest = detection
                    self.trace.append(detection.metric)
                    self.changes = list(detection.changes or [])
        except Exception as error:                      # reported to the page
            self.error = f"{type(error).__name__}: {error}"
        finally:
            # A view whose source has run out must say so: freezing on the last
            # frame looks exactly like a quiet room, and that is the one reading
            # of the page that would be wrong.
            self.finished = True

    def snapshot(self):
        with self.lock:
            latest = self.latest
            trace = list(self.trace)
            changes = list(self.changes)
            floor = self.detector.floor
            calibrating = self.detector.calibrating
            progress = self.detector.calibration_progress
            frames = self.detector.frames
        if latest is None:
            return {"source": self.description, "rate": self.source.rate,
                    "frames": 0, "calibrating": True, "progress": 0.0,
                    "metrics": [], "changes": [], "error": self.error}
        freqs, powers = detect.power_spectrum(self.trace, self.source.rate) \
            if len(trace) >= self.detector.window else ([], [])
        return {
            "source": self.description,
            "rate": self.source.rate,
            "frames": frames,
            "t": round(latest.t, 3),
            "calibrating": calibrating,
            "progress": round(progress, 3),
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
            "subcarriers": len(changes),
            "finished": self.finished,
            "error": self.error,
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
  <h1>WiFi radar &mdash; channel state information</h1>
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
      <h2>Doppler</h2>
      <canvas id="spectrum" height="120"></canvas>
      <div class="numbers">
        <div>peak <b id="peak">–</b></div>
        <div>in band <b id="band">–</b></div>
      </div>
    </div>
  </div>
  <div class="row" style="margin-top:14px">
    <div class="card" style="flex: 2 1 520px">
      <h2>Subcarriers &mdash; newest on the right</h2>
      <canvas id="waterfall" height="170"></canvas>
    </div>
    <div class="card">
      <h2>Session</h2>
      <div class="numbers" style="grid-template-columns: 1fr">
        <div>frames <b id="frames">0</b></div>
        <div>rate <b id="rate">–</b></div>
        <div>subcarriers <b id="subs">–</b></div>
        <div>time <b id="time">–</b></div>
      </div>
      <div class="sub" id="error" style="margin-top:10px"></div>
    </div>
  </div>

<script>
const TRACE_POINTS = 400, WATERFALL_COLUMNS = 220;
let trace = [], waterfall = [], waterfallRows = 0;

function setup(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 600, height = canvas.height;
  canvas.width = width * ratio; canvas.height = height * ratio;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width, height };
}
function clear(ctx, width, height) {
  ctx.fillStyle = '#11151b'; ctx.fillRect(0, 0, width, height);
}
function line(ctx, y, width, colour, dash) {
  ctx.save(); ctx.strokeStyle = colour; ctx.setLineDash(dash || []);
  ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke(); ctx.restore();
}

const traceCanvas = setup(document.getElementById('trace'));
const spectrumCanvas = setup(document.getElementById('spectrum'));
const waterfallCanvas = setup(document.getElementById('waterfall'));

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
  if (column.length) {
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

function number(id, value, digits) {
  document.getElementById(id).textContent =
    (value === null || value === undefined) ? '–' : Number(value).toFixed(digits);
}

async function poll() {
  try {
    const response = await fetch('/api/state');
    const state = await response.json();
    trace = state.metrics || [];

    const label = document.getElementById('state');
    if (state.error) { label.textContent = 'ERROR'; label.className = 'err'; }
    else if (state.finished) { label.textContent = 'SESSION ENDED'; label.className = 'quiet'; }
    else if (state.calibrating) {
      label.textContent = `CALIBRATING ${Math.round((state.progress || 0) * 100)}%`;
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

    drawTrace(state); drawSpectrum(state); drawWaterfall(state);
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
                        help="sim:<scenario>, file:<path> or serial:<port>")
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="session length in seconds; 0 (the default) runs "
                             "until stopped")
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--fast", action="store_true",
                        help="run the source as fast as the CPU allows")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)

    try:
        source = csi.open_source(args.source,
                                 seconds=args.seconds or None, rate=args.rate)
    except ValueError as error:
        print(f"error: {error}")
        return 2

    session = RadarSession(source, realtime=not args.fast)
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
