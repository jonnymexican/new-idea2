"""WiFi sensing: a motion and presence detector built on CSI, no hardware required.

A stream of channel state information frames goes in, and a live motion trace
comes out. The frames can come from a simulator (so the signal processing can be
built and tested before anything is bought), from a recording, or from a radio
on the desk — the detector cannot tell them apart, which is the point.

Start with `README.md` in this directory, then:

    python -m wifi_radar.watch              # live trace in the terminal
    python -m wifi_radar.live               # the same thing in a browser
    python -m wifi_radar.nexmon             # is a Pi sending CSI yet?
    python -m wifi_radar.doctor             # which board is attached, and is it printing?
    python -m wifi_radar.serial_node --port COM3   # is a serial board talking?
    python -m wifi_radar.thresholds --source sim:quiet   # is 4 sigma right on this input?
    python -m wifi_radar.selftest            # needs a board: break it, prove the doctor, restore it
    python -m unittest wifi_radar.test_radar

An Arduino cannot produce CSI — it has no radio — but serial_node.py turns its
sensor channels into frames the same detector measures, without a speed number
that a set of ADC readings cannot justify.
"""
