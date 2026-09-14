"""
agentcore/entrypoint.py — Amazon Bedrock AgentCore entrypoint for IdentAI.

Dual-mode entrypoint:

*Server mode* (no payload): validates the AWS environment and starts the
FastAPI server with uvicorn on $PORT (default 8080) — the long-running
autonomous runtime, where the learning agent's background loop lives.

*Task mode* (a JSON payload on stdin or via --payload): runs ONE agent task
end-to-end and prints exactly one structured JSON result on stdout — this is
what an AgentCore `InvokeAgentRuntime` call receives and parses. Logs go to
stderr so stdout stays machine-readable.

Supported task payloads
-----------------------
    {"action": "assess_submission",
     "student_id": "stu-42", "milestone_id": "while-loops",
     "pseudocode": "WHILE n < 3 ... END WHILE",
     "expected_output": ["3"],            # optional unit test
     "claimed_output": "3",               # optional hallucination tripwire
     "inputs": {"n": 2},                  # optional variable bindings
     "requires_permission": false}        # true = human-gated advanced module

    {"action": "resolve_decision",
     "decision_id": "<id from a HUMAN_DECISION_REQUIRED result>",
     "resolution": "approve" | "reject" | "advise",
     "human_input": "the mentor's note"}

    {"action": "status"}                     # queue + decisions snapshot

Responses differentiate the hackathon's two outcomes explicitly:

    {"status": "AUTONOMOUS_COMPLETION", ...}      # handled silently
    {"status": "HUMAN_DECISION_REQUIRED",
     "decision": { ...Escalation Decision Brief... }}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Project root on sys.path no matter how the container invokes us
# (/app/agentcore/entrypoint.py -> /app).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | agentcore | %(message)s",
    stream=sys.stderr,  # stdout is reserved for the task-mode JSON response
)
logger = logging.getLogger("agentcore")

VALID_RESOLUTIONS = ("approve", "reject", "advise")


# --------------------------------------------------------------------------- #
# AWS sanity checks — inline, cheap, and non-fatal by design: the engine
# degrades to deterministic mode when Bedrock is unreachable, so a missing
# credential is surfaced as a warning, never a crash.
# --------------------------------------------------------------------------- #

def sanity_check_environment() -> dict[str, Any]:
    """Validate model-provider configuration, credentials, and region."""
    report: dict[str, Any] = {
        "model_provider": os.getenv("MODEL_PROVIDER", "bedrock"),
        "region": os.getenv("BEDROCK_REGION", os.getenv("AWS_REGION", "us-east-1")),
        "credentials": "unknown",
        "region_valid": False,
        "warnings": [],
    }
    provider = str(report["model_provider"]).lower()

    try:
        import boto3

        valid_regions = set(boto3.session.Session().get_available_regions("bedrock"))
        region = str(report["region"])
        report["region_valid"] = region in valid_regions
        if not report["region_valid"]:
            report["warnings"].append(
                f"Region {region!r} is not a known Bedrock region; the model call "
                f"will fail and the engine will fall back to deterministic mode."
            )
        if provider == "bedrock":
            sts = boto3.client("sts", region_name=region)
            identity = sts.get_caller_identity()
            report["credentials"] = "ok"
            report["account_id"] = identity.get("Account")
        else:
            report["credentials"] = f"not required for provider={provider}"
    except ImportError:
        report["credentials"] = "boto3 missing"
        report["warnings"].append("boto3 is not installed — Bedrock reasoning unavailable.")
    except Exception as exc:  # noqa: BLE001 - diagnostics must never kill the runtime
        report["credentials"] = f"unresolvable: {type(exc).__name__}"
        logger.debug("credential probe failed", exc_info=exc)
        report["warnings"].append(
            "AWS credentials did not resolve. The server still starts and the "
            "learning agent still runs in deterministic escalation mode."
        )
    for warning in report["warnings"]:
        logger.warning("sanity: %s", warning)
    return report


# --------------------------------------------------------------------------- #
# Task mode — one payload in, one structured JSON result out
# --------------------------------------------------------------------------- #

def _result(status: str, **fields: Any) -> dict[str, Any]:
    """Uniform response envelope; `status` differentiates the two outcomes."""
    payload = {"status": status}
    payload.update(fields)
    return payload


async def run_task(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Execute one agent task and return its structured result.

    The engine here is the same AutonomousAgentEngine the FastAPI app runs:
    same silent pipeline, same decision gate, same SQLite state — the
    container is just another host for the identical agent behaviour.
    """
    from app.interpreter.engine import AutonomousAgentEngine

    action = str(payload.get("action", "")).strip()
    db_path = os.getenv("LESSONS_DB_PATH", str(PROJECT_ROOT / "data" / "lessons.db"))
    engine = AutonomousAgentEngine(
        db_path,
        poll_interval=float(os.getenv("AGENT_POLL_INTERVAL", "2.0")),
    )
    # Task mode owns its own draining, so migrate the schema but never spawn
    # the background worker inside an AgentCore invocation (idempotent).
    await engine.start(run_worker=False)

    if action == "assess_submission":
        try:
            submission_id = await engine.ingest_submission(
                str(payload["student_id"]),
                str(payload["milestone_id"]),
                str(payload["pseudocode"]),
                expected_output=payload.get("expected_output"),
                claimed_output=payload.get("claimed_output"),
                inputs=payload.get("inputs"),
                tier=payload.get("tier"),
                requires_permission=bool(payload.get("requires_permission", False)),
            )
        except KeyError as exc:
            return _result(
                "AUTONOMOUS_COMPLETION", action=action,
                error=f"missing field: {exc}",
                message="assess_submission requires student_id, milestone_id and pseudocode.",
            )
        await engine.drain_once()  # process this queue now, synchronously
        briefs = await engine.open_decisions()
        mine = [b for b in briefs if b.submission_id == submission_id]
        common = {
            "action": action,
            "submission_id": submission_id,
            "student_id": payload.get("student_id"),
            "milestone_id": payload.get("milestone_id"),
            "reasoning": "strands" if engine.reasoning_enabled else "deterministic",
        }
        if mine:
            return _result(
                "HUMAN_DECISION_REQUIRED",
                **common,
                outcome="escalated",
                decision=mine[0].to_dict(),
                message=(
                    "The agent paused and surfaced an Escalation Decision Brief; "
                    "resolve it with action=resolve_decision to resume."
                ),
            )
        return _result(
            "AUTONOMOUS_COMPLETION",
            **common,
            outcome=await submission_status(engine.db_path, submission_id),
            decision=None,
            message="Handled autonomously — no human attention was required.",
        )

    if action == "resolve_decision":
        resolution = str(payload.get("resolution", "approve"))
        if resolution not in VALID_RESOLUTIONS:
            return _result(
                "AUTONOMOUS_COMPLETION", action=action,
                error=f"resolution must be one of {VALID_RESOLUTIONS}",
            )
        try:
            brief = await engine.resolve_decision(
                str(payload["decision_id"]),
                str(payload.get("human_input", "")),
                resolution=resolution,  # type: ignore[arg-type]
            )
        except KeyError:
            return _result("AUTONOMOUS_COMPLETION", action=action,
                           error="missing field: decision_id")
        except LookupError as exc:
            return _result("AUTONOMOUS_COMPLETION", action=action, error=str(exc))
        await engine.drain_once()  # the agent resumes immediately
        return _result(
            "AUTONOMOUS_COMPLETION",
            action=action,
            decision=brief.to_dict(),
            message=f"Mentor '{resolution}' applied — background execution resumed.",
        )

    if action == "status":
        briefs = await engine.open_decisions()
        return _result(
            "AUTONOMOUS_COMPLETION",
            action=action,
            open_decisions=[b.to_dict() for b in briefs],
            reasoning="strands" if engine.reasoning_enabled else "deterministic",
        )

    return _result(
        "AUTONOMOUS_COMPLETION", action=action or None,
        error=(
            f"unknown action {action!r}; expected "
            "assess_submission, resolve_decision, or status"
        ),
    )


async def submission_status(db_path: str | Path, submission_id: int) -> str:
    """Read back one submission's terminal status after the drain."""
    import aiosqlite

    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            "SELECT status FROM student_submissions WHERE id = ?", (submission_id,)
        ) as cursor:
            row = await cursor.fetchone()
    return str(row[0]) if row else "unknown"


def emit(payload: dict[str, Any]) -> None:
    """The ONLY thing this process ever writes to stdout in task mode."""
    json.dump(payload, sys.stdout, ensure_ascii=False, default=str)
    sys.stdout.write("\n")


def serve() -> None:
    """Long-running mode: the FastAPI app with its embedded background agent."""
    report = sanity_check_environment()
    port = int(os.getenv("PORT", "8080"))
    host = os.getenv("HOST", "0.0.0.0")
    workers = int(os.getenv("WEB_CONCURRENCY", "1"))

    logger.info("Initializing IdentAI AgentCore runtime...")
    logger.info("Model provider: %s | region: %s | credentials: %s",
                report["model_provider"], report["region"], report["credentials"])
    if not report["region_valid"]:
        logger.warning(
            "BEDROCK_REGION=%s is not a known Bedrock region — fix before demoing.",
            report["region"],
        )
    logger.info("Starting uvicorn on %s:%d (workers=%d)...", host, port, workers)

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
        access_log=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="IdentAI AgentCore entrypoint")
    parser.add_argument(
        "--payload",
        help="Agent task payload as a JSON string; reads stdin when omitted.",
    )
    parser.add_argument(
        "--serve", action="store_true",
        help="Force server mode even when a payload is present.",
    )
    args = parser.parse_args()

    if args.serve:
        serve()
        return

    raw = args.payload
    if raw is None and not sys.stdin.isatty():
        raw = sys.stdin.read()

    if raw is None or not raw.strip():
        serve()
        return

    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("payload must be a JSON object")
    except (ValueError, TypeError) as exc:
        emit(_result("AUTONOMOUS_COMPLETION", error=f"invalid payload: {exc}"))
        sys.exit(1)

    sanity_check_environment()
    try:
        emit(asyncio.run(run_task(payload)))
    except Exception as exc:  # noqa: BLE001 - the container must always answer valid JSON
        logger.exception("Task failed")
        emit(_result("AUTONOMOUS_COMPLETION", error=f"{type(exc).__name__}: {exc}"))
        sys.exit(1)


if __name__ == "__main__":
    main()
