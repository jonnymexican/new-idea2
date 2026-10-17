"""WiFi sensing: a motion and presence detector built on CSI, no hardware required.

A stream of channel state information frames goes in, and a live motion trace
comes out. The frames can come from a simulator (so the signal processing can be
built and tested before anything is bought), from a recording, or later from a
receiver on the desk — the detector cannot tell them apart, which is the point.

Start with `README.md` in this directory, then:

    python -m wifi_radar.watch          # live trace in the terminal
    python -m wifi_radar.live           # the same thing in a browser
    python -m unittest wifi_radar.test_radar
"""
