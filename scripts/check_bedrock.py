"""
check_bedrock.py — can this environment actually reach Amazon Bedrock?

Answers five questions in one shot, in this order:
  1. Are AWS credentials resolvable?
  2. Is an AWS region configured?
  3. Can we reach the Bedrock control plane?
  4. Is the configured model invocable?
  5. Can the Strands agent build itself end-to-end?

Standalone on purpose: imports main.py for the configuration, but boto3 + the
strands provider do the network calls so the script still works when the app is
not running.

    python check_bedrock.py
    python check_bedrock.py --json

Exits 0 only when every check is green. Credentials and API keys are never
printed; only their shape and a masked tail.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app.main as app  # the FastAPI app, but only its config + model factory are used


def _mask(value: str | None) -> str:
    if not value:
        return "(none)"
    if len(value) <= 4:
        return "***"
    return f"***{value[-4:]} (len={len(value)})"


def _check_credentials() -> dict[str, Any]:
    """1. Are AWS credentials resolvable from this process?"""
    try:
        import boto3
    except ImportError:
        return {"ok": False, "detail": "boto3 is not installed. Run `pip install boto3`."}

    try:
        session = boto3.Session(region_name=app.BEDROCK_REGION)
        credentials = session.get_credentials()
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "detail": f"AWS credentials could not be resolved: {type(exc).__name__}: {exc}",
        }

    if credentials is None:
        return {
            "ok": False,
            "detail": (
                "No AWS credentials found. Run `aws configure`, or set "
                "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION in .env."
            ),
        }

    frozen = credentials.get_frozen_credentials()
    info = {
        "method": credentials.method or "unknown",
        "access_key_id": _mask(frozen.access_key) if frozen else "(unreadable)",
    }
    return {
        "ok": True,
        "detail": f"credentials resolved via {info['method']}",
        "info": info,
    }


def _check_region() -> dict[str, Any]:
    """2. Is an AWS region configured?"""
    if not app.BEDROCK_REGION:
        return {
            "ok": False,
            "detail": "No AWS region set. Set AWS_REGION (or BEDROCK_REGION) in .env.",
        }
    return {
        "ok": True,
        "detail": f"region configured: {app.BEDROCK_REGION}",
        "info": {"region": app.BEDROCK_REGION},
    }


def _check_bedrock_api() -> dict[str, Any]:
    """3. Can we reach the Bedrock control plane and list foundation models?"""
    try:
        import boto3
    except ImportError:
        return {"ok": False, "detail": "boto3 is not installed."}

    try:
        from botocore.config import Config
        boto_cfg = Config(connect_timeout=2, read_timeout=2, retries={"max_attempts": 1})
        client = boto3.client("bedrock", region_name=app.BEDROCK_REGION, config=boto_cfg)
        # boto3 parameter names differ between releases; try the filter if the
        # installed version accepts it, fall back to an unfiltered listing. The
        # only question this check answers is "can we reach the control plane".
        response = None
        for kwargs in (
            {"byOutputModality": "TEXT"},
            {"byOutputModalities": "TEXT"},
            {},
        ):
            try:
                response = client.list_foundation_models(**kwargs)
                break
            except TypeError:
                continue
        if response is None:
            return {
                "ok": False,
                "detail": "Installed boto3 does not know any list_foundation_models signature",
            }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "detail": (
                f"Bedrock control plane unreachable in {app.BEDROCK_REGION}: "
                f"{type(exc).__name__}: {str(exc)[:200]}"
            ),
        }

    models = response.get("modelSummaries", [])
    return {
        "ok": True,
        "detail": f"control plane answered — {len(models)} model(s) listed",
        "info": {"region": app.BEDROCK_REGION, "model_count": len(models)},
    }


def _check_model_access() -> dict[str, Any]:
    """4. Is the configured model invocable for on-demand inference?"""
    try:
        import boto3
    except ImportError:
        return {"ok": False, "detail": "boto3 is not installed."}

    if not app.BEDROCK_MODEL_ID:
        return {
            "ok": True,
            "detail": (
                "BEDROCK_MODEL_ID is not set; the strands default will be used at "
                "runtime. The actual model id is decided when the agent is built "
                "(see check 5)."
            ),
            "info": {"configured": None, "runtime": app.resolved_model_id()},
        }

    try:
        from botocore.config import Config
        boto_cfg = Config(connect_timeout=2, read_timeout=2, retries={"max_attempts": 1})
        client = boto3.client("bedrock", region_name=app.BEDROCK_REGION, config=boto_cfg)
        response = client.get_foundation_model(modelIdentifier=app.BEDROCK_MODEL_ID)
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "detail": (
                f"Model {app.BEDROCK_MODEL_ID!r} is not invocable in {app.BEDROCK_REGION}: "
                f"{type(exc).__name__}: {str(exc)[:200]}"
            ),
        }

    details = response.get("modelDetails") or {}
    return {
        "ok": True,
        "detail": f"model {app.BEDROCK_MODEL_ID!r} is invocable",
        "info": {
            "configured": app.BEDROCK_MODEL_ID,
            "provider": details.get("providerName"),
            "response_supported": "TEXT" in (details.get("outputModalities") or []),
        },
    }


def _check_agent_init() -> dict[str, Any]:
    """5. Can the Strands agent build itself with the resolved model?"""
    try:
        model = app.agent_model()
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "detail": (
                f"Strands agent could not initialise: {type(exc).__name__}: "
                f"{str(exc)[:200]}"
            ),
        }

    return {
        "ok": True,
        "detail": (
            f"Strands agent ready — model={app.resolved_model_id()} "
            f"region={app.BEDROCK_REGION}"
        ),
        "info": {
            "model_id": app.resolved_model_id(),
            "model_label": app.model_label(),
            "provider": app.MODEL_PROVIDER,
        },
    }


CHECKS = (
    ("credentials", _check_credentials),
    ("region", _check_region),
    ("bedrock_api", _check_bedrock_api),
    ("model_access", _check_model_access),
    ("agent_init", _check_agent_init),
)


def run_checks() -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for name, fn in CHECKS:
        try:
            results.append({"name": name, **fn()})
        except Exception as exc:  # noqa: BLE001 - one check must not abort the rest
            results.append({
                "name": name,
                "ok": False,
                "detail": f"check crashed: {type(exc).__name__}: {exc}",
            })

    overall_ok = all(result["ok"] for result in results)
    return {
        "ok": overall_ok,
        "provider": app.MODEL_PROVIDER,
        "region": app.BEDROCK_REGION,
        "configured_model": app.BEDROCK_MODEL_ID or None,
        "resolved_model": app.resolved_model_id(),
        "checks": results,
    }


def print_human(payload: dict[str, Any]) -> int:
    print("IdentAI — Bedrock access check")
    print("-" * 52)
    print(f"  provider     : {payload['provider']}")
    print(f"  region       : {payload['region']}")
    print(f"  configured   : {payload['configured_model'] or '(strands default)'}")
    print(f"  resolved     : {payload['resolved_model']}")
    print("-" * 52)

    for check in payload["checks"]:
        marker = "ok   " if check["ok"] else "FAIL "
        print(f"  {marker} {check['name']:<14} {check['detail']}")

    print("-" * 52)
    if payload["ok"]:
        print("PASS — Bedrock is reachable and the Strands agent is ready.")
        return 0
    print("FAIL — at least one check did not pass. See the FAIL lines above.")
    print("       Common fixes:")
    print("         - `aws configure` (or refresh SSO with `aws sso login`)")
    print("         - Set AWS_REGION in .env to a region where Bedrock is available")
    print("         - Request model access in the Bedrock console, or set")
    print("           BEDROCK_MODEL_ID to a model id this account can invoke.")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--json", action="store_true",
        help="Emit a JSON report on stdout instead of the human-readable table.",
    )
    args = parser.parse_args()

    try:
        payload = run_checks()
    except Exception:
        if args.json:
            print(json.dumps({
                "ok": False,
                "error": "diagnostic crashed",
                "traceback": traceback.format_exc(),
            }))
        else:
            print("FAIL — diagnostic crashed before any check could run.")
            traceback.print_exc()
        return 2

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return 0 if payload["ok"] else 1
    return print_human(payload)


if __name__ == "__main__":
    sys.exit(main())
