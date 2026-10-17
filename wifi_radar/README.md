# WiFi radar — movement and presence from channel state information

A motion sensor built on the channel state information (CSI) a WiFi receiver
reports. It answers one question continuously: **is anything moving in this
room, and how fast?** It runs today with no hardware, on simulated frames, and
it takes real frames without a single change when a receiver is attached — that
seam is the point of the design, not a side effect.

```bash
.venv/bin/python -m wifi_radar.watch                    # live trace in the terminal
.venv/bin/python -m wifi_radar.live                     # live view in a browser
.venv/bin/python -m unittest wifi_radar.test_radar      # 36 tests, no dependencies
```

Nothing here is measured from a real radio yet. Every number below comes from
the simulator or from arithmetic, and is labelled as such.

## What you are looking at

`--source sim:demo` is an empty room for five seconds, someone walking through
it for ten, someone sitting and using their hands for ten, then empty again. The
terminal trace looks like this — this is real output, one line per half second:

```text
   5.00s         ▂▂▂▂▁▁▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▃ m=0.03187 z=  3.8 peak=  n/aHz
   5.50s PRESENT ▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▃▃▃▂▂▂▂▂▂▂▂ m=0.01813 z=  0.4 peak=  n/aHz
   6.50s PRESENT ▂▂▂▃▃▃▂▂▂▂▂▂▂▂▃▅▇▇▅▃▂▂▂▂▂▃▆██▆▃▂▂▂ m=0.02148 z=  1.2 peak= 0.31Hz
   7.50s PRESENT ▂▂▂▂▂▃▆██▆▃▂▂▂▄▆█▇▇▇█▇▅▃▃▄▇█▇███▆▄ m=0.04292 z=  6.5 peak= 1.72Hz
   9.00s PRESENT ██▆▄▅▆█▇████▇█▇▅▄▆████████▇▅▇█████ m=0.11741 z= 24.7 peak= 1.88Hz
```

The columns are the last 34 frames of the motion metric — the sparkline is a
shape, not a number, because a person walking is a rise and fall over a second
or two. `z` is how far above the calibrated noise floor the metric is. `peak` is
where the movement sits in frequency.

In the browser (`wifi_radar.live`, port 8770) the same session draws three
panels: the **trace** with the calibrated floor and the detection threshold
drawn on it, the **Doppler spectrum**, and a **waterfall** of the same movement
per subcarrier — which shows whether the whole channel moved or one subcarrier
glitched, the difference between a body and a bad packet.

| Number | Means |
| --- | --- |
| `metric` | frame-to-frame change in the channel's shape, gain-normalised. The detector's only input |
| `floor` | that metric measured during calibration, in an empty room |
| `threshold` | `floor + 4 x` the floor's own spread — a multiple of the measured noise, never a fixed number |
| `z` | how many spreads above the floor the current frame is |
| `PRESENT` | something moved within the last two seconds |
| `peak` | Doppler peak in the 0.3–8 Hz band: the speed of whatever moved |
| `agreement` | fraction of subcarriers that moved together |

## What it can and cannot do

This is the honest list, and the tests hold each line to a number rather than a
sentence.

**It works on movement.** A person walking through the link raises the metric
from `0.016` to `0.12–0.17` — roughly ten times the noise floor — for
essentially every frame of the walk, and the room returns to its floor within
two seconds of them leaving. `z` reaches 25–37 while they walk.

**It reads speed.** The Doppler peak is the path speed over the wavelength:
12.4 cm at 2.4 GHz, so 15 cm/s reads as 1.2 Hz and 35 cm/s as 2.8 Hz. The suite
checks both against that arithmetic, and they come out at 2.81 Hz measured
against 2.81 Hz predicted. A seated person moving their hands reads 0.78 Hz; a
walk reads 1.88 Hz.

**It sees someone sitting and working, not someone holding still.** A person
using a laptop is detected at about twice the noise floor and holds presence. A
person sitting *completely* still — 8 mm of chest movement, the "breathe"
scenario — is **invisible** to this pipeline: the test asserts that fewer than
20% of frames flag them, and they come in at zero. That is a consequence of a
deliberate choice: the detector uses magnitudes and throws phase away, because
every frame arrives with an unknown phase rotation that makes phase differences
meaningless without synchronisation. Reading a motionless person's breathing
from phase is a real, published technique — it is simply not what this pipeline
does, and the flag does not pretend otherwise.

**It cannot tell you where anything is.** Range resolution is `c / (2B)`: at the
20 MHz channel modelled here that is **7.5 m**, wider than most rooms. No amount
of signal processing fixes that at this bandwidth. It also cannot tell which way
someone is moving: that needs phase, or two receivers.

**It is calibrated, not absolute.** The first two seconds of any session measure
an empty room; every threshold afterwards is a multiple of that measurement.
Calibrate with a person sitting in the room and their movement becomes part of
the baseline — the view says "keep still" while it calibrates for exactly this
reason.

**It is simulated.** No part of this has touched a real radio. The physics in
`csi.py` is real physics (2.4 GHz carrier, 312.5 kHz subcarrier spacing,
per-frame phase and gain wander, complex noise), but a simulator cannot tell you
what a real room's multipath looks like or how a real receiver's AGC behaves
second to second.

**Aliasing is a 20 fps limit.** At 20 frames per second the Nyquist limit is
10 Hz, so movement faster than about 1.2 m/s of path change would wrap around.
Walking is comfortably inside it; a thrown object is not.

## How it works

One quantity, and everything else is about making it mean something over time:

```text
a[t]  = |H[t]|                        magnitude per subcarrier
m[t]  = mean | a[t]/mean(a[t]) - a[t-1]/mean(a[t-1]) |
```

A person moves one path in the room, which rotates that path's phase against the
static ones, so the interference between them — and therefore the amplitude on
every subcarrier — moves. Magnitudes rather than phase differences, and divided
by each frame's own mean, together make the metric blind to the two things real
receivers do constantly: rotate every frame's phase (carrier and sampling
frequency offset) and re-scale every frame's gain (automatic gain control). Both
invariance claims are tested directly, not statistically.

On top of that scalar: a calibration window measures the floor and its spread
(median and median absolute deviation, so one noisy frame cannot inflate the
thresholds); a short moving average stops the flag following single frames;
hysteresis stops it chattering; presence is "moved within the last two seconds";
and a windowed FFT of the metric gives the Doppler band and its peak.

`detect.py`'s module docstring has the full reasoning, including what each
choice costs.

| File | What it holds |
| --- | --- |
| [`csi.py`](csi.py) | The channel model, the scenario menu, and the sources: simulation, recording replay, hardware seam |
| [`detect.py`](detect.py) | The metric, calibration, presence, the FFT and the Doppler band |
| [`watch.py`](watch.py) | The terminal trace |
| [`live.py`](live.py) | The browser view (standard library server, no dependencies) |
| [`test_radar.py`](test_radar.py) | 36 tests against simulated ground truth |

## Attaching a real radio

The pipeline does not care where frames come from. A source is an iterator of
`Frame(t, [complex per subcarrier])`, and everything downstream is written
against that.

**What to know before buying anything.** A laptop's own WiFi card cannot do
this. On Windows the API exposes RSSI and nothing finer; on Linux, CSI needs a
supported chip and a patched driver, and WSL cannot reach the radio to provide
it. So the practical options are:

| Route | Cost | Bandwidth | What it buys |
| --- | --- | --- | --- |
| **ESP32-S3, e.g. [esp-csi](https://github.com/espressif/esp-csi)** | $10–15 per board | 2.4 GHz, 20 MHz | Per-subcarrier CSI over USB; motion and presence exactly as modelled here. The right first purchase |
| **Raspberry Pi 5 + [nexmon_csi](https://github.com/seemoo-lab/nexmon_csi)** | ~$70–120 | up to 80 MHz | 4x finer range bins (1.9 m), a separate capture appliance |
| **Intel AX210 + [PicoScenes](https://ps.zpj.io/)** | $30–60 | up to 160 MHz | A research rig: ~1 kHz CSI, 0.94 m range bins. Needs native Ubuntu, not WSL |

Range resolution is the reason that column matters: at 20 MHz, two people 3 m
apart are in the same bin, and no amount of cleverness changes that. If the goal
is presence and movement — which is where this project is — a $10 ESP32 is
enough. If the goal is range, no ESP32 will do and the answer is 80–160 MHz and
a Linux capture host.

**The interface to implement.** A logger should write one JSON object per frame:

```json
{"t": 0.05, "csi": [[12, -3], [10, 4], ...]}
```

Signed integer pairs are what Espressif's CSI API reports, so firmware can dump
them without converting anything; `read_recording` scales them back (see
`RECORD_SCALE`) and the rest of the pipeline never learns they were integers.
Record a session, then:

```bash
.venv/bin/python -m wifi_radar.watch --source file:session.jsonl
```

`open_source("serial:...")` deliberately raises rather than inventing frames —
there is no CSI on a USB port until something on the other end produces it.
When there is, the adapter belongs in `csi.py` next to the other sources, and it
should emit exactly that JSON shape. The round-trip through that format is
already tested at the integer precision a real logger uses, including that
quantisation does not break detection.

## The tests

`python -m unittest wifi_radar.test_radar` — 36 tests, standard library only,
under a minute. They can be strict because the simulator knows the answer:
it knows when the person is walking, at what cadence, and when the room is
empty.

They pin the things that would otherwise be quietly wrong: the hand-written FFT
against a naive DFT; the spectrum of a known tone; an empty room producing no
detections; a walk detected for essentially all of it and none of the rest;
presence holding for a seated person and releasing after they leave; the Doppler
peak against `v / wavelength`; breathing staying invisible; phase and gain
wander not registering as movement; a frame-wide gain change measuring *exactly*
zero; a session surviving the recording round trip; and the source spec refusing
the serial case with a message that points somewhere.

Four bugs were found by these tests before any of it was wired to a view, and
worth naming because each would have been comfortable to believe: the simulator
originally omitted the carrier phase term (leaving a channel almost inert to
motion, which would have validated the detector on a signal with no Doppler in
it); `power_spectrum` windowed its own zero-padding, which put a large
low-frequency spike under a perfectly constant input; the metric's claimed
gain-invariance was false as first written, and is exact now; and the simulated
walker's torso sway initially outweighed their stride, so their path speed
crossed zero mid-walk and a correctly-working detector reported nothing.
