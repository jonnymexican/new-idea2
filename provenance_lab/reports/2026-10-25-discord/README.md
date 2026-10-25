# Discord, 2026-10-25

One measurement of a real Discord webhook, kept because the weekly
`live-platforms` job skips every unconfigured platform and this was the first
run against a real chat service rather than the local harness.

```
platforms.py run --platforms discord \
    --image "C:\personal\pics\2008-09-22\2008-09-22 002.jpg" --out out/photo-discord
```

The photo is a 2048x1536, 1,512,877-byte JPEG; the lab re-encoded it at quality
92 (759,678 bytes, EXIF dropped) and signed it to 916,814 bytes. Discord stored
that file untouched and served the identical bytes from both
`cdn.discordapp.com` and `media.discordapp.net`, credential included. Only the
`&format=webp` rendition lost it — and that rendition is byte-identical to the
lab's simulated `webp` policy, so the simulation is calibrated against real
Discord.

`platforms.json` holds the redacted report the arm writes; the signature
parameters on the CDN URLs are stripped, and the 24-hour expiry on them
(issued 04:45Z, expiring 04:45Z on the 26th) has long passed anyway. The
message is still in the channel.
