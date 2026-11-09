# Kali cheat-sheet

Everything here is run from this checkout inside **Kali** (`wsl -d kali-linux`).
The repo sits on the Windows drive, so the directory is
`/mnt/c/Users/jonat/new-idea2`.

Two rules:

- call `.venv/bin/python` explicitly — Kali's `python3` is 3.14, the venvs are 3.12;
- never run `pip`, or create a venv, from Windows into `.venv`: those are Linux
  environments, and `venv_check.py` reports a Windows-written one as broken.

## Preflight

```bash
cd /mnt/c/Users/jonat/new-idea2
python3 venv_check.py          # both .venv dirs: ok, or BROKEN plus the two commands that rebuild it
```

## Tests

`ci_tests.py` is the runner CI uses. It fails on any skip and on a suite that ran
nothing, so a green line means tests actually ran.

```bash
.venv/bin/python ci_tests.py test_all               # training, tokeniser, server (136 tests)
.venv/bin/python ci_tests.py test_peek              # attention viewer (20)
.venv/bin/python ci_tests.py test_archil_store      # disk module, SDK faked
.venv/bin/python ci_tests.py test_venv_check        # the environment check itself
.venv/bin/python ci_tests.py test_bench_map         # map page vs README + workflows
.venv/bin/python ci_tests.py mediatop.test_mediatop # the board (56)
.venv/bin/python ci_tests.py othello.test_othello   # rules, search, ladder
.venv/bin/python ci_tests.py wifi_radar.test_radar  # radar (212)
.venv/bin/python ci_tests.py test_all test_peek     # several suites in one run
```

The provenance lab keeps its own environment and its own suite:

```bash
cd provenance_lab
.venv/bin/python ../ci_tests.py --discover .        # 169 tests
cd ..
```

## The model

```bash
.venv/bin/python train.py --max-iters 2000
.venv/bin/python chat.py                       # default checkpoint; --prompt, --checkpoint, --stats
.venv/bin/python serve.py                      # replies over HTTP
.venv/bin/python peek.py                       # attention viewer, http://127.0.0.1:8766/
.venv/bin/python render_curve.py               # the loss curves from runs/metrics.json
```

## ditherlab

Its own git repo, standard library only, so the system `python3` is enough.

```bash
cd ditherlab
python3 -m unittest discover -s ditherlab -t .      # 87 tests, about half a second
python3 -m ditherlab.lab --scene radial --palettes 4,8,16
python3 -m ditherlab.lab --methods bayer,floyd --kinds uniform,median --out out   # writes out/sheet.png
```

## Provenance lab

```bash
cd provenance_lab
.venv/bin/python lab.py run --out out/full     # sign, transform, verify, report
.venv/bin/python lab.py run --transforms jpeg_q75,crop_10pct
.venv/bin/python lab.py run --modes as_is,rebind
.venv/bin/python lab.py verify out/full/assets/signed.jpg --ca out/full/certs/ca.cert.pem
.venv/bin/python demo.py                       # http://127.0.0.1:8788/
```

## Radar, board, othello

```bash
.venv/bin/python -m wifi_radar.watch           # live trace; --source serial:auto, udp:5500
.venv/bin/python -m wifi_radar.live            # browser view
.venv/bin/python -m wifi_radar.doctor          # needs pyserial and a board on a cable
.venv/bin/python -m wifi_radar.fakeboard       # the doctor against a pty, no hardware
.venv/bin/python -m mediatop.server --print    # the board as text; drop --print for the page
.venv/bin/python -m othello.cli play --depth 6
.venv/bin/python -m othello.cli ladder
```

## Notes

- `git` works from Kali: the private `ditherlab` remote authenticates through the
  Windows Git Credential Manager, and both repos are set to `core.autocrlf=input`
  so `git status` reads the same from either shell.
- `/mnt/c` is slower than a native filesystem; `test_all` took 115 s here.
- `espeak-ng` and `arduino-cli` are not installed, and the suites pass without
  them; they are only needed to exercise a real speech engine or compile the sketch.
