/*
  mega_channel_node — an Arduino Mega 2560 as a sensor node for wifi_radar.

  WHAT THIS IS
  Eight ADC channels sampled in lockstep, printed as one line per sample at
  20 Hz, 115200 baud:

      # mega_channel_node 8 channels 20 Hz
      S,1204,512,498,733,501,455,610,521,487
      S,1254,513,499,731,503,452,612,519,488
      ...

  wifi_radar.serial_node reads those lines and feeds them to the same detector
  the CSI sources use: same calibration, same threshold, same presence flag, and
  the same waterfall — one row per channel instead of one per subcarrier.

  WHAT THIS IS NOT
  A WiFi receiver. The ATmega2560 has no radio of any kind: no WiFi, no
  Bluetooth, and no channel state information to report. The 2.4 GHz modules
  often sold with Arduino kits (nRF24L01, and its clones) are packet radios that
  report a single bit of "something is transmitting", which is not the same
  measurement and cannot be turned into one. If you want CSI, the board to buy
  is an ESP32; see README.md in the directory above.

  So this is a different sensor with a familiar output. What it measures is
  whatever you wire to the pins — light on an LDR, sound on a piezo, distance
  from an HC-SR04, or the 50/60 Hz mains hum a bare wire picks out of the air,
  which shifts when a hand comes near it. The detector's metric asks "how much
  did this profile change", which is a real question for all of those. What it
  cannot answer is "how fast did something move": there is no wavelength and no
  Doppler shift in an ADC reading, so the views report no speed for this source
  rather than inventing one.

  WIRING
  Nothing is required: with all eight pins unconnected the readings still wander
  a little, which is enough to see the pipeline work. For a demonstration worth
  watching, put a bare 10 cm wire in A0 and wave a hand near it — the mains hum
  it couples to the pin changes measurably, and 20 Hz is fast enough to show the
  change frame by frame.

  The Mega has 16 analog pins (A0..A15). Bump CHANNELS to use more; the host
  takes whatever count arrives, and a change of count mid-session is treated as
  a resynchronisation rather than as motion.
*/

const uint8_t CHANNELS = 8;          // A0..A7
const uint16_t SAMPLE_HZ = 20;       // must match --rate on the host
const uint8_t AVERAGE = 4;           // readings per channel per sample
const unsigned long BAUD = 115200;

const unsigned long PERIOD_MS = 1000UL / SAMPLE_HZ;
unsigned long nextSample = 0;

void setup() {
  Serial.begin(BAUD);
  // The host ignores lines starting with '#', so a banner is free, and it is
  // the only way to tell from the log which firmware produced a recording.
  Serial.print(F("# mega_channel_node "));
  Serial.print(CHANNELS);
  Serial.print(F(" channels "));
  Serial.print(SAMPLE_HZ);
  Serial.println(F(" Hz"));
  nextSample = millis() + PERIOD_MS;
}

void loop() {
  // Non-blocking scheduling rather than delay(): delay() drifts by however long
  // the sampling itself took, and a drifting sample clock puts every frequency
  // the host computes in the wrong place.
  if ((long)(millis() - nextSample) < 0) {
    return;
  }
  nextSample += PERIOD_MS;

  Serial.print(F("S,"));
  Serial.print(millis());

  for (uint8_t pin = 0; pin < CHANNELS; pin++) {
    // The first conversion after the multiplexer moves is taken from the
    // previous pin's sample-and-hold, so it is thrown away. Reading a pin
    // repeatedly and averaging the rest is what turns a 10-bit reading with two
    // counts of jitter into something the noise floor can be measured from.
    analogRead(pin);
    unsigned long total = 0;
    for (uint8_t i = 0; i < AVERAGE; i++) {
      total += analogRead(pin);
    }
    Serial.print(',');
    Serial.print(total / AVERAGE);
  }
  Serial.println();
}
