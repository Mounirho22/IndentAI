"""
check_key.py — is the Gemini key actually usable?

Answers three questions in one shot:
  1. Can the app find a key at all (environment variable, else .env)?
  2. Does Google accept it?
  3. Is the model main.py asks for actually available to that key?

Standalone and stdlib-only on purpose. It does not import main.py, so it works even
if fastapi/litellm/strands are not installed yet, and it cannot be derailed by an
unrelated import error in the app.

    python check_key.py

Exits 0 only if the key works and the configured model is reachable. The key itself
is never printed — only a masked tail, which is enough to tell two keys apart.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent / ".env"
API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
TIMEOUT_SECONDS = 20


def read_env_file(path: Path) -> dict[str, str]:
    """Minimal .env reader — deliberately a copy of main.py's rules, not an import."""
    values: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return values

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        name, separator, value = line.partition("=")
        if not separator:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()
        values[name.strip()] = value
    return values


def mask(key: str) -> str:
    return f"{len(key)} chars, ending {key[-4:]}" if len(key) > 4 else "too short to mask"


def get_json(path: str, key: str) -> tuple[int, dict]:
    """
    GET a v1beta endpoint with the key in a header rather than the query string.

    Header auth keeps the secret out of URLs, which are the part most likely to end
    up in a proxy log or a shell history.
    """
    request = urllib.request.Request(
        f"{API_ROOT}/{path}",
        headers={"x-goog-api-key": key, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", "replace")
        try:
            return error.code, json.loads(body)
        except json.JSONDecodeError:
            return error.code, {"error": {"message": body[:400]}}


def main() -> int:
    # Same precedence the app uses: a real environment variable beats the file.
    file_values = read_env_file(ENV_FILE)
    key = os.getenv("GEMINI_API_KEY") or file_values.get("GEMINI_API_KEY", "")
    source = (
        "the environment"
        if os.getenv("GEMINI_API_KEY")
        else f"{ENV_FILE.name}" if key else "nowhere"
    )

    model_id = (
        os.getenv("GEMINI_MODEL_ID")
        or file_values.get("GEMINI_MODEL_ID")
        or "gemini/gemini-2.5-flash"
    )
    # LiteLLM needs the "gemini/" provider prefix; the REST API does not use it.
    bare_model = model_id.split("/", 1)[1] if "/" in model_id else model_id

    print("IdentAI — key check")
    print("-" * 52)

    if not key:
        print(f"  FAIL  no key found. Looked in the environment and {ENV_FILE}")
        print("        POST /generate will return 503 until one is set.")
        return 1
    print(f"  ok    key found in {source} ({mask(key)})")
    print(f"  ok    model configured: {model_id}  ->  asking for {bare_model}")

    # Shape check before the network call, because the shape explains the answer. Not
    # fatal on its own — Google's verdict is the evidence, this is just the diagnosis.
    if key.startswith("AQ."):
        print("  WARN  this looks like a short-lived Live API token, not an API key.")
        print("        Tokens beginning 'AQ.' are minted for browser Live API sessions")
        print("        and expire within minutes; the generateContent endpoint this")
        print("        app uses will refuse them. Expect a 400 or 401 below.")
        print("        Fix: create a key at https://aistudio.google.com/apikey")
        print("              -> looks like 'AIza...', 39 characters")
        print(f"              -> put it in {ENV_FILE.name} as GEMINI_API_KEY=AIza...")
    elif not key.startswith("AIza"):
        print("  WARN  an AI Studio key normally starts 'AIza' and is 39 characters.")
        print("        Testing it anyway — Google's answer is what counts.")

    try:
        status, payload = get_json("models", key)
    except urllib.error.URLError as error:
        print(f"  FAIL  could not reach Google: {error.reason}")
        print("        Check your connection, VPN or proxy. The key was not tested.")
        return 1

    if status != 200:
        message = (payload.get("error") or {}).get("message", "(no message)")
        print(f"  FAIL  Google rejected the request (HTTP {status})")
        print(f"        {message}")
        if status == 400:
            print("        400 usually means the key is malformed, or it is not an")
            print("        AI Studio key. Generate one at aistudio.google.com/apikey")
        elif status == 403:
            print("        403 usually means the key is real but the Generative")
            print("        Language API is not enabled for its project.")
        elif status == 429:
            print("        429 is a quota problem, not a key problem — the key works.")
        return 1

    names = [
        model.get("name", "").removeprefix("models/")
        for model in payload.get("models", [])
    ]
    usable = [
        model.get("name", "").removeprefix("models/")
        for model in payload.get("models", [])
        if "generateContent" in (model.get("supportedGenerationMethods") or [])
    ]
    print(f"  ok    key accepted — {len(names)} models visible")

    if bare_model in usable:
        print(f"  ok    {bare_model} is available and supports generateContent")
        print("-" * 52)
        print("PASS — start the server and the generate bar will work.")
        return 0

    print(f"  FAIL  {bare_model} is not available to this key for generateContent")
    if usable:
        print("        Models you could use instead (set GEMINI_MODEL_ID in .env,")
        print('        keeping the "gemini/" prefix):')
        for name in sorted(usable)[:12]:
            print(f"          gemini/{name}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
