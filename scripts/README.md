# scripts

Standalone utility / diagnostic scripts. Run from the repo root.

| Script | Purpose |
| --- | --- |
| `check_bedrock.py` | Five-check diagnostic that the running environment can reach & invoke Amazon Bedrock, and build the Strands agent end-to-end. `python scripts/check_bedrock.py [--json]` |
| `check_key.py`     | Stdlib-only check that the configured Gemini key exists and can reach the API. `python scripts/check_key.py` |
| `batch_test.py`    | Ad-hoc stress of a *running* server (defaults to `127.0.0.1:8000`); queues lessons via `/process-lesson`. `python scripts/batch_test.py` |
| `test_flagged.py`  | Manual one-shot POST to `/process-lesson` against a running server. |

Notes:

- `check_bedrock` and `check_key` never print secrets; they print masked tails
  only. `check_bedrock` uses the app's real configuration (`app.main`) and does
  its own network calls, so it works whether or not the app is running.
- `batch_test.py` requires the server to already be up (it is a client).
