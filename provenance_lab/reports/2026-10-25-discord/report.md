# Platform survival report

Generated 2026-10-25T04:45:03+00:00 · c2pa-python 0.38.0 (SDK 0.91.0) · fixture 2048x1536 · 916,814 bytes signed

## What each service served back

| platform | variant | status | served | format | dims | bytes | identical | credential | failures |
|---|---|---|---|---|---|---|---|---|---|
| `discord` | attachment | 200 | valid | jpeg | 2048x1536 | 916,814 | yes | Trusted | - |
| `discord` | proxy | 200 | valid | jpeg | 2048x1536 | 916,814 | yes | Trusted | - |
| `discord` | proxy_webp | 200 | gone | webp | 2048x1536 | 278,894 | no | - | - |

## Counts

| platform | variants | gone | invalid | valid | byte-identical | formats served |
|---|---|---|---|---|---|---|
| `discord` | 3 | 1 | 0 | 2 | 2 | jpeg, webp |

## Caveats

- The signing CA is a throwaway test anchor. `trusted` means trusted against that anchor; signature and binding checks are real.
- A platform that serves `gone` may still be showing the credential to some clients through an internal path this arm cannot see.
- `platforms.json` and `uploads/*.json` carry redacted URLs by design. `reuse` reads the owner-only refetch file recorded in `platforms.json`; a run copied without that file can only re-fetch anonymous-host URLs.
