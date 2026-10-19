# WiFi radar — movement and presence from channel state information

A motion sensor built on the channel state information (CSI) a WiFi receiver
reports. It answers one question continuously: **is anything moving in this
room, and how fast?** It runs today with no hardware, on simulated frames, and
it takes real frames without a single change when a receiver is attached — that
seam is the point of the design, not a side effect.

```bash
.venv/bin/python -m wifi_radar.watch                    # live trace in the terminal
.venv/bin/python -m wifi_radar.live                     # live view in a browser
.venv/bin/python -m unittest wifi_radar.test_radar      # 89 tests, no dependencies
```

Nothing here is measured from a real radio yet. Every number below comes from
the simulator or from arithmetic, and is labelled as such.

If the board on your desk is an Arduino rather than a radio, it cannot see the
channel at all — see [If you already have an Arduino](#if-you-already-have-an-arduino)
for why, and for the sensor node it can be instead.

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

A source with no radio gets a fourth panel, the **pins** one: a row per analog
pin — all sixteen of a Mega's A0–A15, with the canvas growing to keep every row
legible rather than squeezing the font — and three columns per row. `pin` is the
label, `swing` is the range that row has covered over the visible window
(max − min, in ADC counts), `level` is what it reads now. The pin currently
standing out is highlighted, and named underneath with its level and swing. It
is the wiring view: put a sensor on A3, disturb it, and A3 is the row that
answers.

`swing` exists because every row is scaled to its own range, which is both the
point and the trap: a pin carrying six counts of ADC noise fills its row as
convincingly as a sensor with a 283-count swing does. Read the swing and the
level, not the height of the line — a noise row says `6`, a pin with a hand
waving past it says `283`. When several pins move together and none
leads, the panel says so rather than picking one.

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
| [`nexmon.py`](nexmon.py) | Real frames off the wire: the nexmon packet layout and the UDP source |
| [`serial_node.py`](serial_node.py) | Frames off a USB serial port: CSI lines, and sensor-channel lines from a board with no radio |
| [`arduino/mega_channel_node`](arduino/mega_channel_node/mega_channel_node.ino) | The Mega 2560 sketch that prints those channel lines |
| [`test_radar.py`](test_radar.py) | 89 tests against simulated ground truth |

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
| **Raspberry Pi 3B+, 4B or 5 + [nexmon_csi](https://github.com/seemoo-lab/nexmon_csi)** | ~$35–120 | up to 80 MHz | 4x finer range bins (1.9 m) at 80 MHz, a separate capture appliance. Broadcasts CSI as UDP, which `udp:` already reads |
| **Intel AX210 + [PicoScenes](https://ps.zpj.io/)** | $30–60 | up to 160 MHz | A research rig: ~1 kHz CSI, 0.94 m range bins. Needs native Ubuntu, not WSL |

Range resolution is the reason that column matters: at 20 MHz, two people 3 m
apart are in the same bin, and no amount of cleverness changes that. If the goal
is presence and movement — which is where this project is — a $10 ESP32 is
enough. If the goal is range, no ESP32 will do and the answer is 80–160 MHz and
a Linux capture host.

### If you already have a Raspberry Pi

The Pi is the one option here that may already be on your desk, and it needs no
adapter: it broadcasts CSI as UDP and `udp:` reads it directly. Two things decide
whether it will work.

**The radio.** nexmon_csi patches the **BCM43455c0**, which is the WiFi chip in
the **Pi 3B+, 4B and 5**. The Pi Zero 2 W (BCM43436) and Pi 400 (BCM43456) have
different radios with no CSI patch — Nexmon supports them for other features,
which is not the same thing. Identify the board from the board itself rather
than from the silkscreen or the box it came in:

```bash
cat /sys/firmware/devicetree/base/model      # "Raspberry Pi 4 Model B Rev 1.4", etc.
dmesg | grep -i brcmfmac | head -3           # the chip firmware version it loaded
```

**The firmware.** The CSI patch is written for firmware **7_45_189**; the newer
7_45_234 is not supported by it, and picking the wrong one is the standard way
this fails. The Pi 5 additionally needs the `kernel8.img` (64-bit, 4K pages)
boot image. Both are in the project's own instructions; the current
`Makefile.rpi` route also avoids the old modified-driver step that a kernel
update used to break.

Then, with the Pi patched, on the machine that should process the frames:

```bash
.venv/bin/python -m wifi_radar.nexmon --port 5500          # is anything arriving?
.venv/bin/python -m wifi_radar.watch --source udp:5500
.venv/bin/python -m wifi_radar.watch --source udp:5500 --record real.jsonl
```

The first command is the one to run first: it prints the first packets' tone
count, header length, sequence and core, so a working capture is visible before
the detector is involved. If nothing arrives it says which three things to
check, because **nexmon extracts one frame per received WiFi frame** — an idle
channel produces no data at all, and a Pi with stock firmware produces none ever.

Two consequences of the Pi being the sensor: its own `wlan0` is busy capturing,
so connect to it over Ethernet, and something else on the channel (your access
point) has to be transmitting for there to be frames to measure. Reported rates
are around 50 frames/s, comfortably above the 20 the views are tuned for;
`--rate` sets the number the Doppler axis is scaled by, and `describe()` reports
what is actually arriving.

### If you already have an Arduino

It cannot do this, and it is worth being exact about why. An ATmega2560 — the
chip on an Arduino Mega 2560, including Elegoo's clone — has no radio at all: no
WiFi, no Bluetooth, and no way to observe the channel state of anything. The
2.4 GHz module that comes with most Arduino kits does not rescue it. An
nRF24L01 is a packet radio whose entire received-power output is one bit meaning
"something above −64 dBm is on the air"; that is one bit per packet, not 64
complex numbers per frame, and no firmware turns it into CSI.

What the board does have is a 10-bit analog-to-digital converter on sixteen pins,
and that is enough to be a *sensor node* — a different instrument with the same
output shape. Upload
[`arduino/mega_channel_node/mega_channel_node.ino`](arduino/mega_channel_node/mega_channel_node.ino)
(select **Arduino Mega or Mega 2560**; the Elegoo uses a CH340 USB chip, so
Windows needs that driver before a COM port appears at all), close the Arduino
IDE's serial monitor — it holds the port open — then:

```bash
.venv/bin/python -m pip install pyserial                 # the one optional dependency
.venv/bin/python -m wifi_radar.serial_node --list        # which port is the board?
.venv/bin/python -m wifi_radar.serial_node               # is it talking?
.venv/bin/python -m wifi_radar.watch --source serial:auto      # terminal trace
.venv/bin/python -m wifi_radar.live  --source serial:auto      # browser view
```

In the browser, `serial:auto` also adds the pins panel described above — the
same command for wiring that the CSI path uses for walking.

`--port` and `serial:` both accept `auto`, which is what they default to, and
which is the point: `COM3` or `COM4` is the question a first bring-up stops on,
before anyone suspects the sketch. The choice is made from the USB vendor ID —
`1A86` is the CH340 on Elegoo and most clone boards, and it is emphatically not
an Arduino ID, so matching on the word "Arduino" alone would find nothing on the
board in the picture above. Ties break on the port name, so two boards are chosen
between the same way twice rather than at random, and the port it picked is
printed with the reason. `--list` shows every port and marks the one `auto` would
open; when nothing is plugged in it names the CH340 driver, which is the actual
cause on Windows more often than not.

The sketch samples A0–A7 twenty times a second and prints one line per sample:

```text
# mega_channel_node 8 channels 20 Hz
S,1204,512,498,733,501,455,610,521,487
S,1254,513,499,731,503,452,612,519,488
```

Each line becomes a frame of eight channel amplitudes, and everything downstream
runs unchanged: the same calibration, the same measured noise floor, the same
hysteresis, the same presence flag, and the same waterfall with one row per
channel instead of one per subcarrier. What a channel *is* depends on what you
wired to it: light on an LDR, sound on a piezo, distance from an HC-SR04, or the
mains hum that a bare 10 cm wire in A0 couples out of the air — which shifts
measurably when a hand comes near it, and needs no parts at all. That last one is
the one to try first.

**What it cannot report is speed.** For CSI the metric's spectrum peaks at
`speed / wavelength`, so 1.88 Hz means 23 cm/s. For a channel profile the same
spectrum measures how fast the *profile* changed, which depends on where each
sensor points and how much of the room it sees, not on how fast anyone walked.
So the source reports `doppler = False`, the terminal prints `peak = n/a`, and
the browser labels that panel "Doppler — not on this source" and draws nothing
in it. The numbers that survive are real ones: with a hand waving in front of
four of the eight channels, the metric goes from `0.0019` to `0.024` — thirteen
times the calibrated floor, `z` around 50 — while a still room stays under the
threshold. That is a test, not a claim: a simulated Mega's output goes through
the real detector, and it asserts no detections before the hand appears and over
90% of frames after.

Two things it does that the CSI path does not. A hand that appears and then
holds still changes the profile *once*, so the moment it arrived is marked even
though nothing moves afterwards. And an unplugged channel reads a constant, which
the detector now survives: a floor and a spread of exactly zero used to make `z`
undefined and the event invisible, and calibration can no longer divide by
nothing.

**What it also cannot do:** tell you where anything is (one amplitude per sensor
is not a range profile), see through a wall, or separate a person from a cat, a
curtain, or a lamp switched on. It is a set of light, sound and capacitance
channels; it answers questions about those channels.

The sketch compiles for `arduino:avr:mega` under `--warnings all` with nothing
from its own file: 2474 bytes of flash and 192 bytes of RAM, on a board with
253952 and 8192. CI compiles it on every push, so it cannot rot into something
that only compiles in a screenshot. What CI cannot do is upload it; no board here
has had it run on real pins, and the output above is what the protocol says, not
a captured log.

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

`nexmon.py` is that interface for the Pi: it decodes the packet layout the
firmware documents — int16 real/imaginary pairs behind an 18- or 20-byte header,
64/128/256 tones for 20/40/80 MHz — and hands the detector frames with only the
timestamp added. Payload length fingerprints the header revision, so a packet is
either decoded exactly or refused with the accepted sizes; misreading 256 bytes
silently would report confident nonsense for the rest of the session.

`open_source("serial:COM3")` reads a board on a USB port live, and accepts
either kind of line, because the port does not say what is on the other end:

```text
S,<millis>,<v0>,<v1>,...        a sensor node: channel values
C,<millis>,<re0>,<im0>,...      a CSI receiver: int16 real/imaginary pairs
# anything                        ignored, so firmware can print a banner
```

An ESP32 running esp-csi printing `C,` lines therefore works through the same
gate as the Mega, which is the useful property: the cheap board that *can* do CSI
and the board that cannot are one flag apart, and the flag is set from what
arrives rather than from what the user claims.

`millis` is the device's own clock, because it is the only one that knows when a
sample was taken. It is an unsigned 32-bit count that wraps after 49.7 days, so
the wrap is undone rather than allowed to jump the session backwards. Junk lines
are counted and skipped; fifty of them before the first sample is a configuration
error and says so, and ten seconds of silence names the three things that are
actually wrong on a desk — the sketch is not running, the port is the wrong one,
or a serial monitor already holds it. A saved capture replayed from a file ends
at EOF instead, so a log can be read back without pretending it is a live board.

The round-trip through the JSON Lines recording format is tested at the integer
precision a real logger uses, including that quantisation does not break
detection.

## The tests

`python -m unittest wifi_radar.test_radar` — 89 tests, standard library only
(pyserial is needed to open a board, never to test one), well under a minute.
They can be strict because the simulator knows the answer:
it knows when the person is walking, at what cadence, and when the room is
empty.

They pin the things that would otherwise be quietly wrong: the hand-written FFT
against a naive DFT; the spectrum of a known tone; an empty room producing no
detections; a walk detected for essentially all of it and none of the rest;
presence holding for a seated person and releasing after they leave; the Doppler
peak against `v / wavelength`; breathing staying invisible; phase and gain
wander not registering as movement; a frame-wide gain change measuring *exactly*
zero; a session surviving the recording round trip; and a serial spec naming a port
that cannot be opened failing with that port's name and the reason, instead of
returning a source that would never produce a frame.

The hardware path is tested without the hardware. A payload builder writes the
byte layout the firmware documents, so the parser is checked against the format
rather than against itself: all three channel widths, both header revisions,
int16 values passed through unchanged, the RSSI fork's extra byte not shifting
the CSI, and unknown lengths and missing magic refused instead of half-parsed.
Then frames are pushed through a **real loopback UDP socket** into the detector —
97.6% of the walking frames flagged, 0% of the quiet ones, the same 1.72 Hz peak
as the simulated source — plus the things that actually go wrong on a desk: junk
on the port skipped and counted, a stream of nothing but junk failing loudly
rather than looking like a quiet room, a second core ignored before it can
interleave two channels into fake motion, and silence producing a message that
names the three things to check.

The serial path is tested the same way, on a string instead of a wire. Both
dialects of the line protocol are checked, along with the ways it goes wrong in
practice — a wrong baud rate, a stray `println`, a line that arrived without its
newline — and the 32-bit wrap by feeding a session that crosses it and asserting
the timestamps keep going forward. Then eight channels of ADC data, with a hand
waving in front of half of them, go through the detector: no detections while the
room is still, over 90% while it moves, and `peak_hz` `None` throughout, because
that is a number this sensor cannot produce.

Five bugs were found by these tests before any of it was wired to a view, and
worth naming because each would have been comfortable to believe: the simulator
originally omitted the carrier phase term (leaving a channel almost inert to
motion, which would have validated the detector on a signal with no Doppler in
it); `power_spectrum` windowed its own zero-padding, which put a large
low-frequency spike under a perfectly constant input; the metric's claimed
gain-invariance was false as first written, and is exact now; and the simulated
walker's torso sway initially outweighed their stride, so their path speed
crossed zero mid-walk and a correctly-working detector reported nothing. The
nexmon parser added one of its own: the first version of the walk test fed
frames over a real socket but timestamped them from the real clock, so 400
frames arrived in a few milliseconds, every frame sat at `t≈0`, and the test
reported zero detections — the frames were fine, the clock was wrong.
