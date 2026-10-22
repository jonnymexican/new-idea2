# WiFi radar — movement and presence from channel state information

A motion sensor built on the channel state information (CSI) a WiFi receiver
reports. It answers one question continuously: **is anything moving in this
room, and how fast?** It runs today with no hardware, on simulated frames, and
it takes real frames without a single change when a receiver is attached — that
seam is the point of the design, not a side effect.

```bash
.venv/bin/python -m wifi_radar.watch                    # live trace in the terminal
.venv/bin/python -m wifi_radar.live                     # live view in a browser
.venv/bin/python -m unittest wifi_radar.test_radar      # 212 tests, no dependencies
.venv/bin/python -m wifi_radar.selftest                 # needs a Mega on a cable
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

**It is calibrated, not absolute.** Every session starts by measuring an empty
room, and every threshold afterwards is a multiple of that measurement.
Calibrate with a person sitting in the room and their movement becomes part of
the baseline — the view says "keep still" while it calibrates for exactly this
reason. The length is automatic: at least two seconds, which is right for radio
noise because it is fast and flat, and then as long as the input keeps drifting.
A wire on an ADC pin is the case that needs the longer window — its reading
swings on a slow mains envelope, so a two-second sample can catch one part of a
cycle and the rest of every cycle then reads as movement. (Measured on a Mega
with a wire in A0: an 8 s baseline left presence on for 94% of an idle session;
25 s left it at 0%.)

How the automatic length decides, in [detect.py](detect.py#L324):

- **Is it drifting at all?** The high-pass is compared with the same frames
  unfiltered. A walk survives that filter almost intact — ratio 0.94 on the
  simulated walk and on breathing — while a 0.1 Hz envelope is most of what it
  removes — ratio 0.2-0.4. Near 1 means nothing slow is happening and the two-
  second minimum is enough.
- **If it is, how long is a cycle?** The spectrum of the metric so far says how
  slow the drift is, and the window keeps growing until it holds three cycles of
  it. One or two is not enough: a floor measured over part of an envelope is a
  floor for that part only.
- **What if it never settles?** Forty seconds is the limit, after which the
  window closes anyway and the run says so. A hand resting on the sensor, or a
  battery slowly dying, is not something to wait out.

`--calibrate SECONDS` still fixes the window at exactly that length, and
`--calibrate auto` is the default. The terminal summary prints which happened —
`calibration : 16.0s (auto, covered whole cycles of the drift)` — and the browser
counts the seconds rather than showing a percentage of a length it does not
know yet.

**The threshold is a multiple of the measured spread, and the multiple can be
measured.** `thresholds.py` runs an input at several `on_sigma` values and
reports, for each, how often a still room trips it and what fraction of the
frames where anything rose at all it still catches:

```bash
.venv/bin/python -m wifi_radar.thresholds --source serial:COM5 \
    --hand serial:COM5 --seconds 60
```

`--source` is the idle run — leave the room, or keep still — and `--hand` is the
same port while you wave at the antenna; the seconds are spent on each in turn.
It names the lowest threshold whose idle rate stays under `--max-idle` (1%), or
says that none of them does.

A recording does the same without the hardware, and answers a question a live run
cannot: how long the calibration window has to be. `--record` takes what
`handtrial` wrote — or a plain recording of an empty room, which needs a minute of
standing still rather than a hand moved on cue — tries each window length against
the idle frames that follow it, then sweeps the threshold with the winner fixed:

```bash
.venv/bin/python -m wifi_radar.thresholds --record out/wire-new.jsonl --sigmas 2,3,4,5,6
```

On the wire, a 2 s window leaves the idle stretch tripping 16.7% of the time at
4σ and a 5 s window 26.4%, so the floor the detector measures in two seconds is
not the floor the room has; 15 s leaves 1.0%. With 15 s fixed, 4σ holds the idle
at 1.0% and 5σ at 0.0%, and the recording names `--calibrate 15 --on-sigma 5`.
Both are flags on `watch` and `live`, so the number the measurement produced is
the number the session runs with.

On the simulator the idle rate is 0.0% at every threshold from 2 to 6 while the
walk is caught 98.4% at 2σ and 89.5% at 6σ; the default stays at 4σ because a
simulated noise floor is not a wire's, and the wire's tail is the thing worth
measuring.

The metric is also high-passed by default. Each channel is compared with a
baseline that follows it with a 0.5 s time constant, so a change slower than that
— the ten-second envelope a wire picks up, a supply sagging, a sensor warming up
— is not reported as motion. It costs nothing measurable on the simulated walk:
97.1% of the walk is detected at any time constant from 0 to 1 s. `--highpass 0`
turns it off; a longer value rejects more drift and starts to cost the slowest
movement. What it cannot do is invent a signal that never arrived: on this board
a hand brought to within 2 cm of the wire moved A0 by less than the noise
(5.45 counts of spread during the approaches against 5.36 while still), so
touching the wire remains the way to make this input respond. That was the first
wire; the longer one now in A0 swings its level by ±40% on a five-second cycle,
which is why the calibration window matters as much as the threshold.

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
| [`thresholds.py`](thresholds.py) | Measures false alarms against missed movement for each detection threshold, live or from a marked recording, and derives the calibration length from it |
| [`cues.py`](cues.py) | Tones and speech for a trial's segment changes, so the person can keep their eyes on the wire |
| [`handtrial.py`](handtrial.py) | A guided antenna trial: records empty room then a hand at each distance, and reports z with and without the high-pass |
| [`fakeboard.py`](fakeboard.py) | A board on a pseudo-terminal, so CI can run the doctor against a real port |
| [`nexmon.py`](nexmon.py) | Real frames off the wire: the nexmon packet layout and the UDP source |
| [`serial_node.py`](serial_node.py) | Frames off a USB serial port: CSI lines, and sensor-channel lines from a board with no radio |
| [`doctor.py`](doctor.py) | One-step bring-up: which board is this, is its chip alive, and is the sketch printing? |
| [`arduino/mega_channel_node`](arduino/mega_channel_node/mega_channel_node.ino) | The Mega 2560 sketch that prints those channel lines |
| [`selftest.py`](selftest.py) | The hardware-in-the-loop check: flash a silent sketch, prove `doctor` diagnoses it, put `mega_channel_node` back |
| [`test_radar.py`](test_radar.py) | 212 tests against simulated ground truth |

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
.venv/bin/python -m wifi_radar.doctor                    # which board, is it alive, is it printing?
.venv/bin/python -m wifi_radar.serial_node --list        # which port is the board?
.venv/bin/python -m wifi_radar.serial_node               # is it talking?
.venv/bin/python -m wifi_radar.watch --source serial:auto       # terminal trace
.venv/bin/python -m wifi_radar.live  --source serial:auto       # browser view
```

`doctor` is the command to start with, because it answers all three questions a
silent board asks at once — is there a port, is the chip alive, is anything being
printed — and names the fix for whichever one fails:

```text
board      : COM5  Arduino Mega 2560 (COM5)  VID:PID=2341:0042  SER=85935333537351609150
recognised : Arduino USB-serial

bootloader : answered 5 syncs in 2.1s (status 0xc0 = OK)
             the MCU is powered, clocked and running its bootloader
sketch     : 63 lines, 62 samples in 4.1s, 20.0/s on the board's own clock
             # mega_channel_node 8 channels 20 Hz
             S,50,392,389,362,379,337,345,339,315

ok — the board is identified, alive, and printing samples.
   next: python -m wifi_radar.watch --source serial:COM5
```

The middle line is the one `--list` cannot give you, and the one that separates a
broken board from a missing sketch. Every Arduino resets when its port opens,
because DTR is wired to the reset pin, and for about a second the bootloader
waits for stk500v2 commands and answers `GET_SYNC`. A reply proves the MCU, its
clock and the USB-serial wiring are all fine; it says nothing about the sketch,
so `doctor` then resets the board again and listens. Exit status is 0 when samples
arrive, 1 when a board was found but nothing usable is printing, and 2 when
nothing could be opened — so a script can tell "no board" from "wrong firmware".

When `--baud` produces no samples, the doctor tries the common rates — 9600,
57600, 38400, 19200, 230400, 250000, 500000 — giving each two seconds, and stops
at the first one that decodes. Then it names the rate instead of leaving you to
guess at the sketch's `Serial.begin()`:

```text
sketch     : 71 lines, 70 samples in 2.1s, 20.0/s on the board's own clock
             (arrived at 9600 baud; 115200 produced no samples)

ok — the board is identified, alive, and printing samples.
   baud : the firmware is printing at 9600, not 115200 — pass --baud 9600
```

`--scan-seconds` sets the time given to each rate (2 s by default, which is about
a second of samples after the board's reset); `0` disables the scan entirely.

`--json` prints the same answers as one object and nothing else, so a CI job or
a wiring check can read them without parsing the table above:

```bash
.venv/bin/python -m wifi_radar.doctor --json
```

```json
{
  "port": "COM5",
  "verdict": "printing",
  "exit_status": 0,
  "message": "the board is identified, alive, and printing samples",
  "board": { "device": "COM5", "description": "Arduino Mega 2560 (COM5)",
             "manufacturer": "Arduino LLC (www.arduino.cc)",
             "vid": 9025, "pid": 66,
             "serial_number": "85935333537351609150",
             "recognised": "Arduino USB-serial", "confidence": 3 },
  "bootloader": { "answered": true, "replies": 18, "status": 192,
                  "status_ok": true, "seconds": 2.1, "error": null },
  "sketch": { "printing": true, "bytes": 2458, "lines": 62, "samples": 61,
              "rejected": 0, "rate_hz": 20.0, "baud": 115200,
              "baud_requested": 115200,
              "banner": "# mega_channel_node 8 channels 20 Hz",
              "first_sample": "S,50,431,422,392,405,364,369,362,334",
              "seconds": 4.1, "error": null },
  "bauds_tried": [115200]
}
```

`verdict` is one of `printing`, `silent_firmware` (bootloader answers, nothing
prints), `silent_no_bootloader` (nothing at all), `wrong_protocol` (bytes that
are not `S,` or `C,` lines), `unopenable`, or `no_port`. `sketch.baud` is the
rate the samples actually arrived at, `sketch.baud_requested` what was asked for,
and `bauds_tried` every rate the scan opened — so a caller can read the answer
out of the document rather than out of the prose. Numbers stay numbers —
`rate_hz` and `vid` are not the formatted strings the text report draws — and
`exit_status` repeats the process exit code so a stored log still records what
the run concluded.

For a hands-on session, `handtrial` does the counting for you: it records an
empty room, then prints when to hold your hand at 20, 30 and 40 cm, marks where
each stretch is, and replays the recording through the detector twice — once
with the high-pass and once without — reporting each segment's z.

```bash
.venv/bin/python -m wifi_radar.handtrial --source serial:auto \
    --record out/wire.jsonl
```

The report is one table per high-pass setting — frames, median and maximum `z`,
and the fraction above the 4σ threshold for each segment. That pairing is the
point: a hand that shows up with the filter on and vanishes with it off means the
filter is eating the signal; the reverse means it is doing its job. The marks are
written beside the recording (`out/wire.jsonl.marks.json`), so a trial can be
analysed again later without repeating the hand movements.

Every trial now opens with a placement control: `--control 10` (the default)
records ten seconds of empty room and ten with a hand 2 cm from the wire, and
abandons the run with exit 4 if no channel level moves by 3σ. A hand that is not
where the wire is looks exactly like an antenna that sees nothing, and the
difference between the two is twenty seconds of the person's time instead of
eighty. When it passes it says which channel moved, which is also the answer to
"is the wire in the pin I think it is": the sketch reads A0..A7 in order, so a
wire in A0 is the channel that moves most, and one in A3 is the fourth.

The report carries a second table, because the motion metric answers a narrower
question than "was the hand there". `levels` gives, for each stretch, the channel
that moved most and by how much, in units of the idle stretch's own spread,
taken over one-second windows rather than the whole stretch: a hand that arrives
slowly and stays for four seconds of a fifteen-second stretch is a presence, and
a median over the whole stretch calls it nothing. A run where no stretch clears
either test is named a baseline and exits 3, instead of printing a row of zeros
to be read as reach.

With the cues working, the same trial produced a result. 1601 frames over 80 s:
20 s of empty room, then 15 s each with the hand at 2, 20, 30 and 40 cm, placed
on the spoken prompt.

| segment | frames | median z | max z | z > 4σ |
| --- | --- | --- | --- | --- |
| idle | 359 | −0.4 | 5.2 | 2.8% |
| 2 cm | 300 | −0.1 | 71.6 | 25.3% |
| 20 cm | 300 | −0.6 | 5.0 | 0.7% |
| 30 cm | 300 | −0.9 | 4.8 | 1.3% |
| 40 cm | 300 | 0.3 | 5.1 | 3.7% |

**The hand is visible close in and nowhere else.** At what the schedule called
2 cm the hand drove the first two channels into the floor — medians of 0 against
301 and 306 — and the detector flagged 20-25% of the stretch from 4σ to 8σ. At
20, 30 and 40 cm nothing exceeded the idle's own maximum (4.8-5.1 against 5.2),
and the raw levels agree: +22, +27 and −25 counts against a 40-count idle spread.
The response sits in t=25-35 s of the 2 cm stretch, which began at 20 s — the
cue, then a hand walked into place. `thresholds --record` on that recording picks
3 s and 4σ (idle 0.0% at 3 s against 2.8% at 2 s).

It did not reproduce, and a sweep from contact outwards explains why.

| distance | median z | max z | z > 4σ |
| --- | --- | --- | --- |
| idle | −0.3 | 5.5 | 5.7% |
| 0 cm (touch) | 1.4 | 71.2 | 32.5% |
| 0.5 cm | 0.2 | 32.6 | 7.5% |
| 1 cm | −0.7 | 2.8 | 0.0% |
| 1.5 cm | −0.4 | 4.1 | 1.2% |
| 2 cm | −0.5 | 4.2 | 0.6% |

Touching the wire takes A0 down to zero — the control's row was `A0..A7 −100%
−100% −92% −77% −68% −59% −53% −51%` — and by 1 cm there is nothing left. So the
earlier stretches recorded as "2 cm" that showed a large effect were closer than
2 cm; the coupling here falls off a cliff between half a centimetre and one,
not gently over centimetres. Reach for a bare wire on this board, with this
detector, is contact to about half a centimetre — not the 2-20 cm the first
trial suggested, and the 20-40 cm null stands.

The pin is confirmed the same way: the touch response peaks on A0 and falls
monotonically to A7, which is the order the sketch reads pins in, so the wire is
the one in A0 and the board is coupled to it.

### Measuring a new antenna

The sweep above belongs to one piece of wire, so it is a reference point rather
than a property of the board. A different antenna changes both numbers at once —
how much of a hand it couples, and how much hum it collects as an idle floor —
and the tables are comparable only if everything else is held still: same board,
same sketch, same pin, same `--idle` and `--seconds`. The distance at which the
hand stops registering is then the number that moved.

```bash
.venv/bin/python -m wifi_radar.handtrial --source serial:auto \
    --record out/sweep-antenna2.jsonl --control 8 --control-distance 0 \
    --idle 10 --seconds 8 --distances 0,0.5,1,1.5,2 --no-reconnect
```

That is the wire's sweep with the antenna swapped, and the control at contact is
the reason it is cheap: `--control-distance 0` asks for a finger on the new
antenna before the schedule starts, and an antenna that hears nothing costs
twenty seconds rather than a minute and a half of waving at an empty room. The
control's `A0..A7 level` line is the other half of it — a peak on A0 falling to
A7 means the new antenna is on the pin the old one was on, while a peak anywhere
else compares two pins rather than two antennas. `--no-reconnect` keeps a
dropped board from rebasing the clock under a schedule nobody is being guided
through.

Re-derive the threshold from the new recording before trusting any of it:

```bash
.venv/bin/python -m wifi_radar.thresholds --record out/sweep-antenna2.jsonl
```

3 s and 4σ were measured against the wire's own drift and are not transferable;
the antenna that collects more hum moves the floor, and the calibration window
is chosen from the idle stretch of whatever was actually recorded.

Read the comparison off the cliff, not off the maximum. A touch drives A0 into
the floor, so contact saturates and cannot rank two antennas; what ranks them is
the first distance whose `z > 4σ` share falls back to the idle's. That number
came from one sweep for the wire, so run the new one twice — 8 s per stretch is
one sample of a hand, and the 0.5 → 1 cm boundary here is one measurement wide.

An earlier attempt, run without cues, found nothing at 2 cm either.
The prompts were a segment late and the hand followed the prompts, so it was
never where the marks said it was. `--no-reconnect` is the other guard here: a
reconnect keeps the schedule aligned by rebasing the clock, which is right for a
long session and wrong for a trial, where nobody is being guided during the gap.

Two bugs surfaced only on the wire. The first: `open_serial` left a stale line in
the driver's buffer, and its older `millis()` became the session's zero — every
live frame then looked 30-120 s in the future, so the schedule raced through every
segment in seconds while the person was still standing there. It reads as a dead
board and was not one; `open_serial` now flushes the input buffer after the DTR
pulse. The second: a reconnect installed the raw port as the source, so the next
frame failed — the opener returns a `SerialSource` now, and `ReconnectingSource`
wraps a bare stream rather than trust its caller.

That is a process failure, and the fix belongs in the tool rather than in the
procedure: every change of stretch is now announced out loud as well as printed.
`--cue both` (the default) sounds two tones and says the distance — "hand at 20
centimetres" — `--cue bell` keeps the tones only, and `--cue off` goes back to
watching the screen. A trial then works with its output going anywhere, or with
the person facing the wire instead of the terminal.

The speech engine is started once when the trial opens and kept open, because
PowerShell plus SAPI take about three seconds to load on Windows — a cue that
arrives three seconds into a fifteen-second stretch is one the hand follows
late, which corrupts the marks it is supposed to describe. Against an already
running engine a prompt is a write and a flush. `--cue voice` with no engine
installed degrades to the tones rather than to silence, and every failure in
`cues.py` is swallowed: a machine with no audio device must not end a recording.

Fixing the announcement exposed a real bug in the schedule. `record()` advanced
to the next stretch when that stretch had *ended* rather than when it began, so
every prompt was one whole segment late: 15 s on a 15-second stretch. A person
watching the terminal therefore moved when the previous stretch was already
over, which puts the hand in the wrong segment of the recording — which is
exactly what the uncued run shows. It now compares against the next mark's
`start`, and a test pins the 20 cm cue to frame 40 of a two-second-per-stretch
run instead of frame 80.

A trial is still run from a terminal, and still writes the marks that
`thresholds --record` reads afterwards:

```bash
.venv/bin/python -m wifi_radar.handtrial --source serial:auto \
    --record out/wire.jsonl --idle 20 --seconds 15 --distances 2,20,30,40 --cue both
```

For a plain trace, give it time and a longer idle timeout:

```bash
.venv/bin/python -m wifi_radar.watch --source serial:auto --seconds 300 \
    --timeout 60 --record antenna.jsonl
```

`--timeout` is how long a serial source waits in silence before giving up; the
default of 10 s is right for a bring-up and too short while a board is being
reset by hand. Five minutes of trace with a recording of the same frames is what
makes an antenna experiment repeatable: the JSONL can be replayed through a
different `--highpass` or threshold afterwards instead of being remeasured.

A pulled cable also no longer ends the session. A serial source reconnects on
its own — one attempt a second, for two minutes — and the session clock picks up
where the last frame left off plus the real gap, because the board's `millis()`
restarts at zero after a reset and a detector fed a clock going backwards would
throw the session away. A reset that leaves the port open — the usual case on a
Mega, whose USB bridge never disappears — is handled the same way: a backwards
step in `millis()` that is not the 49.7-day wrap is counted as a board reset, and
the session clock continues instead of jumping back to zero. The reconnects are counted in the source line
(`serial COM5 at 115200 baud, …, reconnected 2x`) and announced in the terminal:

```text
note  : COM5 stopped: SerialException: device disconnected. Reconnecting for up to 120s
note  : reconnected to COM5 after 3.2s
```

`--no-reconnect` ends the run instead, and `--reconnect-seconds` changes the
wait. With `serial:auto` the port is chosen again on every attempt, since a
replugged adapter does not always come back under the same name. The first open
is never retried: a board that was not there at all is a bring-up failure, and
pretending otherwise would make the doctor's job harder to see.

There is nothing to set in those two commands. A bare wire in A0 acts as an
antenna for the slow envelope of mains hum, and the floor is a measure of
whatever the input did while calibrating, so a two-second sample of a ten-second
swing is a sample of one part of it — which is why the window grows while the
input drifts instead of being a number the user has to guess. On a simulated
wire — A0 swinging 10 counts on a 0.1 Hz envelope — the automatic window closes
at 16 s with a floor within 2% of a 40 s reference, and the run prints
`calibration : 16.0s (auto, covered whole cycles of the drift)`.
`--calibrate 25` still works if you would rather fix it. The real board has not
run this: its adapter is not accepting writes, so the numbers above are the
simulator's.

In the browser, `serial:auto` also adds the pins panel described above — the
same command for wiring that the CSI path uses for walking. And when a board is
attached but not printing, the page stops guessing: once the source has been
silent long enough to give up, the view runs the doctor once and puts its verdict
under the numbers — `doctor says: not printing — the cable, the port and the
chip are all fine, so the sketch is the missing piece` — instead of leaving an
empty trace for the reader to interpret. `RadarSession` releases the port first, since
only one process can hold it, and the doctor runs as a subprocess so a wedged
adapter cannot stall the page.

The **rate** is part of the spec too. `serial:COM5:9600` says it, and
`serial:COM5:auto` finds it the way the doctor does — the requested rate
first, then 9600, 57600, 38400, 19200, 230400, 250000 and 500000, stopping at the
first that decodes. That scan costs a few seconds and resets the board once per rate,
which is why it happens once at open rather than continuously; the source line
then says which rate it settled on:

```text
serial COM5 at 9600 baud (found by scanning), 8 sensor channels, no Doppler, 20/s (paced to real time)
```

A board that comes back from a reset at a different rate is visible in that line
rather than looking like a source that quietly stopped, and
`python -m wifi_radar.serial_node --baud auto` finds it too. `parse_spec` returns
`None` for auto, so the discovery happens before the stream is opened and the
session that follows has a fixed rate.

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

`python -m unittest wifi_radar.test_radar` — 212 tests, standard library only
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

### The board self-test

One claim above is asserted by proxy: that `doctor` tells "the chip is alive and
the sketch is missing" from "nothing is there" is tested against a fake port, not
a real bootloader. `selftest.py` closes that gap, and it is the only command here
that needs a board:

```bash
.venv/bin/python -m wifi_radar.selftest                    # auto-detects the port
.venv/bin/python -m wifi_radar.selftest --port /dev/ttyACM0
```

It works in four steps, each one an assertion:

1. **Preflight.** `doctor` must already be printing samples. If it is not, the
   run stops before flashing anything — overwriting a broken board, and then
   promising to restore firmware that was never running, turns a diagnosis into
   a second fault. `--force` overrides this for the one case where the reasoning
   inverts: a board that is silent *because* its sketch is missing, which is what
   the restore puts back. The preflight is then recorded under its own name —
   "the board is silent, so it is being overwritten on purpose" — rather than as
   a board that prints: a pass, but one that says what it actually did. It does
   not override a port that will not open (exit 2) — no upload can fix that.
2. **Break it on purpose.** It writes a sketch whose `setup` and `loop` are empty
   and flashes it with `arduino-cli compile --upload` (found on `PATH`, from
   `ARDUINO_CLI`, or in the copy the Arduino IDE bundles). The compile is not
   incidental: `arduino-cli upload` alone does not compile, and fails with
   "Compiled sketch not found" for any sketch that has not been built before —
   which is every sketch this module writes.
3. **Assert the diagnosis.** `doctor` must exit 1, say "not printing", name
   `mega_channel_node` as the sketch to upload, and still show
   `bootloader : answered` — that last one is the whole claim: the chip was
   alive the entire time.
4. **Put it back.** `mega_channel_node` is uploaded again and `doctor` must exit
   0 with samples arriving.

`--json` prints one object and nothing else — every check with its detail, the
pass count, the failures by name, and the exit status — so a scheduled job can
record a result without parsing the running commentary:

```bash
.venv/bin/python -m wifi_radar.selftest --json > selftest.json
```

The restore is in a `finally`, so a failed assertion in step 3 still puts the
firmware back. The run ends with `9/9 checks passed`, or with what failed and
what state the board was left in: a refused run says "nothing was flashed", and
a failed restore says `WARNING` instead of reassuring you — and says whether the
silent sketch actually got on, so you know what to re-flash. Exit status is 0
only when every check passed.

Uploads run under a timeout that kills the **whole process tree**. That detail
is not housekeeping: `arduino-cli` uploads through `avrdude`, which it starts as
its own child, and killing only the parent on a timeout leaves `avrdude` alive
and holding the serial port. Every later upload then fails with "Access is
denied" — indistinguishable from a dead USB adapter, and it cost this bench an
afternoon of blaming the cable before the orphan was found in `tasklist`.

CI does run one check against a real port, without a board: [fakeboard.py](fakeboard.py)
opens a pseudo-terminal that prints the sketch's lines and runs the actual
doctor against it, failing when the verdict is not `printing`:

```bash
python -m wifi_radar.fakeboard --verbose      # Linux and macOS; needs pyserial
```

That job exists because the unit tests pin the pieces but never open a port, and
the first time the real path ran against a pty it crashed on the DTR ioctl — a
driver detail no fake stream would have raised. `--silent` makes the simulated
board print nothing, which is how the failure branch is checked.

It is deliberately not in the unittest suite: CI has no Mega on the end of a
cable, and a test that skips is a test that rots. The script itself is tested
without a board — a scripted `subprocess.run` answers the uploads and the doctor
calls, so the ordering, the refusal, the restore and the wording are pinned.
What that cannot prove is that a real bootloader answers `GET_SYNC` the way the
fake does; that is what running it on a desk is for. On this bench the CH340
still enumerates and returns no bytes, so step 1 currently stops the run with
`0/1 checks passed` and the board untouched.
