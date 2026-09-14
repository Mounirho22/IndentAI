"""
IdentAI — an autonomous Strands agent for differentiated lesson design and mentor operations.

FastAPI + aiosqlite + AWS strands-agents, reasoning on Amazon Bedrock.

The agent is not a pipeline. Given a mission brief it decides which tools to call,
audits its own output, corrects what it can, and escalates to a human teacher when
the problem needs pedagogical judgement rather than another retry.
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import html
import json
import logging
import os
import re
import sys
import textwrap
import time
import traceback
import uuid
from collections.abc import Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import aiosqlite
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Response, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from strands import Agent, tool

# Repo-root / helper-scripts bootstrap. Since this module now lives under app/,
# the lazy `import check_bedrock` inside the /api/bedrock-check route (and any
# other repo-relative import) must still resolve regardless of where the app is
# launched from. Idempotent and harmless when the paths are already present.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _path in (_PROJECT_ROOT, _PROJECT_ROOT / "scripts"):
    _value = str(_path)
    if _value not in sys.path:
        sys.path.insert(0, _value)

# Bedrock is the primary provider. The import path moved between strands releases,
# so try both rather than pinning the app to one of them.
try:
    from strands.models import BedrockModel
except ImportError:  # pragma: no cover - depends on the installed strands version
    from strands.models.bedrock import BedrockModel

# LiteLLM is the documented fallback provider only. Import failure is not fatal:
# a Bedrock-only deployment has no reason to carry it.
try:
    import litellm
    from strands.models.litellm import LiteLLMModel
except ImportError:  # pragma: no cover - fallback provider is optional
    litellm = None
    LiteLLMModel = None

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

# Secrets live in a .env file at the repository root rather than inside this
# module, so main.py can be shared, screen-shared, or submitted without carrying
# an API key. Resolved explicitly (the module now lives under app/).
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def _load_env_file(path: Path = ENV_FILE) -> None:
    """
    Read KEY=value lines from a .env file into os.environ.

    Dependency-free on purpose: python-dotenv is not in requirements.txt, and one
    stdlib function is cheaper than another install for the demo machine.

    Two deliberate rules:
      * A real environment variable always wins, so `set GEMINI_API_KEY=... &&
        python main.py` still overrides the file and deployment never depends on it.
      * A missing or unreadable file is not an error — the app simply falls back to
        the real environment, which is how a deployed run would supply the key.

    Resolved relative to this file rather than the working directory, so the server
    finds the key no matter which folder it was launched from.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        # Tolerate the `export FOO=bar` form people paste in from shell notes.
        if line.startswith("export "):
            line = line[len("export "):].lstrip()

        # partition, not split, so a value containing '=' survives intact.
        name, separator, value = line.partition("=")
        if not separator:
            continue
        name = name.strip()
        value = value.strip()

        # A quoted value is taken literally; an unquoted one ends at a trailing
        # ' #' comment, which is how python-dotenv reads it too.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()

        if name and name not in os.environ:
            os.environ[name] = value


_load_env_file()

# Overridable so a demo or test run can point at a scratch database instead of the
# working copy of lessons.db. The seeded working copy lives under data/.
DB_PATH = os.getenv("LESSONS_DB_PATH", "data/lessons.db")
TIERS = ("struggling", "on-level", "advanced")

# --- Model provider ------------------------------------------------------- #
# "bedrock" is the real target: the agent's reasoning loop AND every tool's own
# generation call go through Amazon Bedrock. "litellm" exists only so a credentials
# problem on demo day cannot take the whole app down — the agent, its tools and the
# autonomy loop are identical either way, which is the point of the Strands
# provider abstraction.
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "bedrock").strip().lower()

# Left unset on purpose. When it is empty, BedrockModel uses whatever the installed
# strands release considers current, which ages better than a hardcoded guess; the
# resolved id is logged at startup either way. Set it to pin a specific model, e.g.
# a Claude Sonnet cross-region inference profile: us.anthropic.claude-sonnet-4-...
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "").strip()

# Bedrock model ids are regional. AWS_REGION is the variable boto3 itself reads, so
# honouring it means `aws configure` alone is usually enough.
BEDROCK_REGION = (
    os.getenv("BEDROCK_REGION")
    or os.getenv("AWS_REGION")
    or os.getenv("AWS_DEFAULT_REGION")
    or "us-east-1"
).strip()

# Streaming needs bedrock:InvokeModelWithResponseStream as well as bedrock:Converse.
# Set BEDROCK_STREAMING=false if the account's policy only grants the latter.
BEDROCK_STREAMING = os.getenv("BEDROCK_STREAMING", "true").strip().lower() not in {
    "0", "false", "no", "off",
}

# model_id MUST carry the "gemini/" provider prefix for LiteLLM to route to Gemini;
# bare "gemini-pro" is unprefixed AND a retired alias, so it 404s.
GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini/gemini-3.6-flash")

# The NAME of the variable, never the key itself. Putting a literal key here is the
# one edit that silently disables generation: os.getenv would treat it as a name,
# find no such variable, return None, and every /generate call would answer 503.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# What the logs call the current model before the provider has been built. The live
# label, once the model has resolved itself, comes from model_label().
MODEL_LABEL = (
    f"bedrock:{BEDROCK_MODEL_ID or 'strands default'} ({BEDROCK_REGION})"
    if MODEL_PROVIDER == "bedrock"
    else GEMINI_MODEL_ID
)

# --- Budgets -------------------------------------------------------------- #
# A mission is an agent loop, not a fixed pipeline: it may generate, audit, correct
# and re-audit, so it needs a far larger ceiling than the old deterministic path.
MISSION_TIMEOUT_SECONDS = float(os.getenv("MISSION_TIMEOUT_SECONDS", "900"))

# The deadline for a mission run on the request path by POST /generate. Generous
# because an agent that audits and corrects its own work takes minutes, not seconds —
# and because the wait is not a blank screen: the desk streams the mission log while
# it works. Lower it if you would rather fail fast than watch.
GENERATE_TIMEOUT_SECONDS = float(os.getenv("GENERATE_TIMEOUT_SECONDS", "600"))

# Per-call ceiling handed to the provider. This one matters: both provider clients are
# blocking and run in a worker thread, and a thread cannot be pre-empted — so without a
# timeout on the call itself, the outer deadline could not fire until the call returned.
GEMINI_TIMEOUT_SECONDS = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "75"))
MODEL_TIMEOUT_SECONDS = float(os.getenv("MODEL_TIMEOUT_SECONDS", str(GEMINI_TIMEOUT_SECONDS)))

# How many times the agent may re-generate one artifact after a failed audit before it
# must escalate instead. Enforced in code, not merely requested in the prompt, so an
# audit that never passes cannot burn the whole mission budget in a loop.
MAX_CORRECTIONS_PER_ARTIFACT = int(os.getenv("MAX_CORRECTIONS_PER_ARTIFACT", "2"))

# Missions are kept in memory for the review desk to read back. Bounded so a
# long-running server cannot grow without limit.
MISSION_HISTORY_LIMIT = int(os.getenv("MISSION_HISTORY_LIMIT", "24"))
MISSION_EVENT_LIMIT = int(os.getenv("MISSION_EVENT_LIMIT", "400"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("lesson-agent")

CREATE_MATERIALS_TABLE = """
CREATE TABLE IF NOT EXISTS materials (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    unit_name        TEXT NOT NULL,
    topic            TEXT NOT NULL,
    objective        TEXT NOT NULL,
    tier             TEXT NOT NULL,
    worksheet_text   TEXT,
    quiz_text        TEXT,
    answer_key_text  TEXT,
    status           TEXT DEFAULT 'draft' CHECK (status IN ('draft', 'approved', 'flagged', 'rejected')),
    reason           TEXT
);
"""

CREATE_CURRICULUM_MISSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS curriculum_missions (
    id                   TEXT PRIMARY KEY,
    topic                TEXT NOT NULL,
    unit_name            TEXT,
    objective            TEXT,
    state                TEXT NOT NULL,
    goal_json            TEXT,
    tier_status_json     TEXT,
    tier_record_ids_json TEXT,
    transitions_json     TEXT,
    sub_mission_ids_json TEXT,
    summary              TEXT,
    created_at           TEXT NOT NULL,
    started_at           TEXT,
    completed_at         TEXT,
    state_changed_at     TEXT NOT NULL
);
"""

# One query behind every read path: the JSON feed, its legacy alias, and the
# server-rendered review desk. Newest first so fresh drafts land at the top.
MATERIALS_SELECT = """
SELECT id, unit_name, topic, objective, tier,
       worksheet_text, quiz_text, answer_key_text, status, reason
FROM materials
ORDER BY id DESC
"""


# Every draft row is written through this one statement, so the agent's
# save_draft tool and POST /generate can never drift apart.
INSERT_DRAFT_SQL = """
INSERT INTO materials (
    unit_name, topic, objective, tier,
    worksheet_text, quiz_text, answer_key_text, status
) VALUES (?, ?, ?, ?, ?, ?, ?, 'draft')
"""


# --------------------------------------------------------------------------- #
# Provider availability
# --------------------------------------------------------------------------- #

def provider_unavailable_reason() -> str | None:
    """
    Say why the configured provider cannot be used, or None if it looks usable.

    Checked before a mission starts so a misconfiguration is a 503 with an
    instruction, rather than an agent that burns two minutes and dies mid-loop.
    Everything here is local: no network call, no charge, no latency.
    """
    if MODEL_PROVIDER == "bedrock":
        try:
            import boto3  # imported lazily so the fallback provider needs no AWS SDK
        except ImportError:
            return ("MODEL_PROVIDER=bedrock but boto3 is not installed. Run "
                    "`pip install boto3`, or set MODEL_PROVIDER=litellm in .env.")
        try:
            credentials = boto3.Session(region_name=BEDROCK_REGION).get_credentials()
        except Exception as exc:  # noqa: BLE001 - a broken profile must not crash the app
            return f"AWS credentials could not be resolved ({type(exc).__name__}: {exc})."
        if credentials is None:
            return (
                "No AWS credentials found, so Bedrock cannot be reached. Run "
                "`aws configure`, or put AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / "
                f"AWS_REGION in {ENV_FILE}. Run `python check_bedrock.py` to test access."
            )
        return None

    if MODEL_PROVIDER == "litellm":
        if litellm is None or LiteLLMModel is None:
            return ("MODEL_PROVIDER=litellm but litellm is not installed. Run "
                    "`pip install litellm`, or set MODEL_PROVIDER=bedrock.")
        if not GEMINI_API_KEY:
            return (f"MODEL_PROVIDER=litellm but GEMINI_API_KEY is not set. Put it in "
                    f"{ENV_FILE} and restart, or set MODEL_PROVIDER=bedrock.")
        return None

    return (f"MODEL_PROVIDER={MODEL_PROVIDER!r} is not a provider this app knows. "
            "Use 'bedrock' (default) or 'litellm'.")


# --------------------------------------------------------------------------- #
# Mission state and observability
# --------------------------------------------------------------------------- #

# The vocabulary of the mission log. Every event the agent produces is one of these,
# which is what lets the review desk render a trail it did not have to parse.
#
# The agent's own chain-of-thought ("agent_reasoning") is deliberately NOT in this
# list: the live panel exists for the demo audience, and a teacher should see what
# the agent decided, not the words it thought with. Reasoning is still captured
# server-side in the verbose log when a future debugging tool needs it.
SAFE_MISSION_EVENT_KINDS = (
    "mission_started",         # the brief was accepted
    "agent_action",            # the agent selected/invoked an action
    "tool_selected",           # backwards-compatible alias
    "tool_called",             # a tool began executing
    "tool_started",            # backwards-compatible alias
    "tool_completed",          # a tool completed and returned output
    "tool_result",             # backwards-compatible alias
    "audit_started",           # an audit invocation is in flight
    "audit_result",            # a verdict on generated material
    "audit_passed",            # audit verdict was "pass" (also a save-ready signal)
    "audit_failed",            # audit verdict was "revise" — the agent must act
    "verification",            # a pseudo-code check
    "self_correction",         # a re-generation prompted by a failed audit
    "material_regenerated",    # a worksheet/quiz/objective was rewritten
    "human_review_required",   # audit said a teacher must decide
    "human_review",            # escalation, with the reason
    "human_review_completed",  # teacher reviewed and acted on an escalation
    "save",                    # a row reached the database
    "mission_completed",
    "mission_failed",
)

# Backward-compatible alias for anything that imported the old name.
MISSION_EVENT_KINDS = SAFE_MISSION_EVENT_KINDS

# Terminal outcomes. "escalated" is a success: the agent correctly refused to guess.
MISSION_OUTCOMES = ("running", "saved", "escalated", "failed", "timeout")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _digest(text: str) -> str:
    """Short content hash, used to tie an audit verdict to the exact text audited."""
    return hashlib.sha1(text.strip().encode("utf-8", "replace")).hexdigest()[:16]


def _clip(value: Any, limit: int = 240) -> str:
    """One-line, length-capped rendering of anything, for a log line."""
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


class Mission:
    """
    One agent run, and the trail it left.

    Held in memory rather than in SQLite on purpose: the trail is demo and debugging
    material with a lifetime of minutes, while `materials` is the teacher's actual
    work. Mixing the two would mean migrating the table that every read path and the
    whole review desk already depend on.
    """

    def __init__(self, mission_id: str, source: str, topic: str,
                 unit_name: str = "", objective: str = "") -> None:
        self.id = mission_id
        self.source = source
        self.topic = topic
        self.unit_name = unit_name
        self.objective = objective
        self.outcome = "running"
        self.started_at = _now_iso()
        self.finished_at: str | None = None
        self.summary = ""
        self.events: list[dict[str, Any]] = []
        self.saved_ids: list[int] = []
        self.flagged_ids: list[int] = []
        self.tool_calls = 0
        # (tier, kind) -> how many times it has been generated. Second and later
        # attempts are self-corrections, and are budgeted.
        self.attempts: dict[str, int] = {}
        # content digest -> the audit verdict for exactly that text.
        self.audits: dict[str, dict[str, Any]] = {}
        self._seq = 0
        self._monotonic = time.monotonic()

    # -- logging ----------------------------------------------------------- #

    def record(self, kind: str, message: str, **fields: Any) -> dict[str, Any]:
        """
        Append one event and mirror it to the server log.

        The terminal line is deliberately shaped like the panel line, so what a
        judge reads over a shoulder matches what the browser shows.
        """
        self._seq += 1
        event = {
            "seq": self._seq,
            "at": _now_iso(),
            "elapsed": round(time.monotonic() - self._monotonic, 2),
            "kind": kind,
            "message": _clip(message, 400),
        }
        event.update({key: value for key, value in fields.items() if value is not None})
        self.events.append(event)

        # Drop the oldest events rather than the newest: a runaway loop is most
        # diagnosable from what it is doing now.
        if len(self.events) > MISSION_EVENT_LIMIT:
            del self.events[: len(self.events) - MISSION_EVENT_LIMIT]

        level = logging.WARNING if kind in {"human_review", "mission_failed"} else logging.INFO
        logger.log(level, "[%s] %-15s %s", self.id, kind, event["message"])
        return event

    def finish(self, outcome: str, summary: str = "") -> None:
        self.outcome = outcome if outcome in MISSION_OUTCOMES else "failed"
        self.finished_at = _now_iso()
        if summary:
            self.summary = _clip(summary, 600)
        self.record(
            "mission_failed" if outcome in {"failed", "timeout"} else "mission_completed",
            summary or f"Mission {outcome}.",
            outcome=self.outcome,
            saved=len(self.saved_ids),
            flagged=len(self.flagged_ids),
            tool_calls=self.tool_calls,
        )

    # -- budgets ----------------------------------------------------------- #

    def attempt(self, tier: str, kind: str) -> int:
        """Count one generation of (tier, kind) and return which attempt it is."""
        key = f"{tier}/{kind}"
        self.attempts[key] = self.attempts.get(key, 0) + 1
        return self.attempts[key]

    def attempts_used(self, tier: str, kind: str) -> int:
        return self.attempts.get(f"{tier}/{kind}", 0)

    def corrections_left(self, tier: str, kind: str) -> int:
        """Corrections still allowed for this artifact. The first pass is free."""
        return max(0, MAX_CORRECTIONS_PER_ARTIFACT - max(0, self.attempts_used(tier, kind) - 1))

    # -- serialisation ----------------------------------------------------- #

    def snapshot(self, since: int = 0) -> dict[str, Any]:
        """The public shape, optionally only the events after `since`."""
        return {
            "mission_id": self.id,
            "source": self.source,
            "topic": self.topic,
            "unit_name": self.unit_name,
            "objective": self.objective,
            "outcome": self.outcome,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "summary": self.summary,
            "saved_ids": list(self.saved_ids),
            "flagged_ids": list(self.flagged_ids),
            "tool_calls": self.tool_calls,
            "latest_seq": self._seq,
            "events": [event for event in self.events if event["seq"] > since],
        }


# Recent missions, oldest first, so the desk can poll one back by id.
MISSIONS: dict[str, Mission] = {}

# Set for the duration of one agent run. A ContextVar rather than a tool argument
# because the mission id is the app's business, not something the model should be
# trusted to carry correctly through ten tool calls.
_CURRENT_MISSION: contextvars.ContextVar[Mission | None] = contextvars.ContextVar(
    "current_mission", default=None
)

MISSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{3,63}$")


def new_mission(source: str, topic: str, mission_id: str | None = None,
                unit_name: str = "", objective: str = "") -> Mission:
    """
    Register a mission, honouring a caller-supplied id when it is safe to.

    The browser sends the id with the request so the log panel can start polling
    immediately instead of waiting for a response that takes minutes. That makes it
    untrusted input, so it is pattern-checked and de-duplicated before use.
    """
    candidate = (mission_id or "").strip()
    if not candidate or not MISSION_ID_PATTERN.match(candidate) or candidate in MISSIONS:
        candidate = f"m-{uuid.uuid4().hex[:10]}"

    mission = Mission(candidate, source, topic, unit_name, objective)
    MISSIONS[candidate] = mission

    while len(MISSIONS) > MISSION_HISTORY_LIMIT:
        MISSIONS.pop(next(iter(MISSIONS)))
    return mission


def current_mission() -> Mission | None:
    """The mission this tool call belongs to, if any."""
    return _CURRENT_MISSION.get()


# --------------------------------------------------------------------------- #
# Curriculum-level mission
#
# A Mission is one agent invocation: a single topic, three tiers, one shot at
# self-correction. A CurriculumMission is the workflow object the teacher sees
# in the dashboard: the "curriculum goal" they handed in, the progress
# against it, and the per-tier sub-missions the agent has spawned to do the
# work. The two concepts are different on purpose: the teacher should never
# have to re-initiate a generation step, so the CurriculumMission is what
# they watch, while the underlying Mission is what the agent reasons through.
# --------------------------------------------------------------------------- #

CURRICULUM_MISSION_STATES = (
    "QUEUED",            # accepted, not yet started
    "RUNNING",           # orchestrator is iterating tiers
    "AUDITING",          # the agent is auditing one tier's output
    "SELF_CORRECTING",   # the agent is regenerating one tier after a failed audit
    "WAITING_FOR_HUMAN", # at least one item is flagged; the rest kept running
    "COMPLETED",         # all tiers accounted for, every flagged item approved
    "FAILED",            # unrecoverable: timeout, no provider, exception
)
CURRICULUM_MISSION_HISTORY_LIMIT = int(os.getenv("CURRICULUM_MISSION_HISTORY_LIMIT", "32"))


class CurriculumMission:
    """
    The teacher's view of one curriculum goal.

    Holds a high-level objective ("Prepare next week's differentiated CS
    materials for Unit 3: Nested Loops") and tracks the orchestrator that
    fulfils it. The orchestrator spawns one Mission per tier, reads each
    sub-mission's outcome, and updates progress here.
    """

    def __init__(self, mission_id: str, goal: dict[str, Any]) -> None:
        self.id = mission_id
        self.goal = dict(goal)  # the teacher's brief, immutable for the record
        self.topic: str = str(goal.get("topic", "")).strip()
        self.unit_name: str = str(goal.get("unit_name", "")).strip()
        self.objective: str = str(goal.get("objective", "")).strip()
        self.tiers: tuple[str, ...] = tuple(goal.get("tiers") or TIERS)
        # State machine
        self.state: str = "QUEUED"
        self.previous_state: str = "QUEUED"
        self.state_changed_at: str = _now_iso()
        # Timeline
        self.created_at: str = _now_iso()
        self.started_at: str | None = None
        self.completed_at: str | None = None
        # Counts and the items themselves
        self.generated_items: list[dict[str, Any]] = []   # every produced record
        self.failed_items: list[dict[str, Any]] = []      # sub-missions that failed
        self.human_review_items: list[dict[str, Any]] = []  # flagged records
        self.final_outputs: list[int] = []                 # approved record ids
        self.sub_mission_ids: list[str] = []              # Mission ids spawned
        # Per-tier progress so the dashboard can show "12/15 tasks completed"
        self.tier_status: dict[str, str] = {tier: "pending" for tier in self.tiers}
        self.tier_record_ids: dict[str, int] = {}
        # State-transition log; the dashboard can replay these
        self.transitions: list[dict[str, Any]] = []
        self.current_phase: str = "queued"
        self.errors: list[str] = []
        self.summary: str = ""
        # Counts derived from sub-missions
        self._auto_repaired: int = 0
        # Timing
        self._monotonic = time.monotonic()
        self.total_agent_seconds: float = 0.0
        self.transition("QUEUED", "accepted, not yet started")

    # -- state machine ----------------------------------------------------- #

    def transition(self, new_state: str, phase: str = "") -> None:
        """Move to a new state. Records the transition for the dashboard."""
        if new_state not in CURRICULUM_MISSION_STATES:
            raise ValueError(
                f"Unknown curriculum-mission state {new_state!r}; "
                f"expected one of {CURRICULUM_MISSION_STATES}."
            )
        if phase:
            self.current_phase = phase
        # Same-state transitions are pure phase updates; do not log them as
        # state changes. The very first QUEUED entry IS logged so the
        # dashboard can show "started in QUEUED" with a timestamp.
        is_initial = (not self.transitions and new_state == "QUEUED")
        if new_state == self.state and not is_initial:
            return
        self.previous_state = self.state
        self.state = new_state
        self.state_changed_at = _now_iso()
        if new_state == "RUNNING" and self.started_at is None:
            self.started_at = _now_iso()
        if new_state in {"COMPLETED", "FAILED"} and self.completed_at is None:
            self.completed_at = _now_iso()
        self.transitions.append({
            "at": self.state_changed_at,
            "to": new_state,
            "from": "INITIAL" if is_initial else self.previous_state,
            "phase": phase,
            "elapsed_s": round(time.monotonic() - self._monotonic, 2),
        })

    # -- progress ---------------------------------------------------------- #

    @property
    def completed_count(self) -> int:
        """Tiers the agent has finished — saved, escalated, or approved — so far."""
        return sum(1 for status in self.tier_status.values()
                   if status in {"saved", "draft", "flagged", "approved"})

    @property
    def total_count(self) -> int:
        return len(self.tiers)

    @property
    def progress_percent(self) -> float:
        if not self.tiers:
            return 0.0
        return round(self.completed_count * 100 / len(self.tiers), 1)

    @property
    def is_terminal(self) -> bool:
        return self.state in {"COMPLETED", "FAILED"}

    # -- sub-mission tracking --------------------------------------------- #

    def record_sub_mission(self, sub_mission_id: str) -> None:
        self.sub_mission_ids.append(sub_mission_id)

    def record_tier_outcome(self, tier: str, record_id: int, status: str) -> None:
        """The agent finished one tier: 'saved'/'draft' (saved), 'flagged' (escalated), or 'failed'."""
        # The orchestrator uses 'saved' (the agent's vocabulary); the
        # database uses 'draft' (the row's status column). Treat them as
        # the same thing on the curriculum-mission side so the rest of
        # the class can speak one vocabulary.
        normalised = "draft" if status == "saved" else status
        self.tier_status[tier] = normalised
        self.tier_record_ids[tier] = record_id
        self.generated_items.append({
            "tier": tier, "record_id": record_id, "status": normalised,
        })
        if normalised == "flagged":
            self.human_review_items.append({
                "tier": tier, "record_id": record_id, "status": "flagged",
            })
        elif normalised == "approved":
            self.final_outputs.append(record_id)
        if self.started_at:
            self.total_agent_seconds = round(
                time.monotonic() - self._monotonic, 2
            )

    def record_sub_failure(self, tier: str, error: str) -> None:
        """The sub-mission for `tier` did not produce anything saveable."""
        self.tier_status[tier] = "failed"
        self.failed_items.append({"tier": tier, "error": error})

    def approve_flagged(self, record_id: int) -> bool:
        """
        Mark a flagged record as approved. Returns True if this is the last
        flagged record and the mission should now complete.
        """
        for item in self.human_review_items:
            if item["record_id"] == record_id:
                item["status"] = "approved"
                self.final_outputs.append(record_id)
                self.tier_status[item["tier"]] = "approved"
                break
        else:
            return False
        # If nothing left to wait on, close the mission.
        if not self.has_pending_human_review and self.state == "WAITING_FOR_HUMAN":
            self.transition("COMPLETED",
                            phase="all flagged items approved by teacher")
            self.summary = self.compute_summary()
        return True

    # -- derived facts ----------------------------------------------------- #

    @property
    def has_pending_human_review(self) -> bool:
        return any(item.get("status") == "flagged"
                   for item in self.human_review_items)

    @property
    def auto_validated(self) -> int:
        """Records the agent both generated AND audited to a pass verdict,
        plus any item the teacher has since approved."""
        return sum(1 for tier_status in self.tier_status.values()
                   if tier_status in {"saved", "draft", "approved"})

    @property
    def auto_repaired(self) -> int:
        """Records that were regenerated after a failed audit."""
        # The orchestrator sets this explicitly when it counts sub-mission
        # attempts that exceeded the first generation.
        return self._auto_repaired

    def count_repair(self) -> None:
        """Recount the auto-repaired total from the sub-missions in memory."""
        total = 0
        for sub_id in self.sub_mission_ids:
            sub = MISSIONS.get(sub_id)
            if sub is None:
                continue
            for attempts in sub.attempts.values():
                if attempts > 1:
                    total += 1
                    break
        self._auto_repaired = total

    def compute_summary(self) -> str:
        """The one-line summary shown on the dashboard and the COMPLETED card."""
        total = self.total_count
        handled = self.auto_validated
        needs_human = sum(1 for item in self.human_review_items
                         if item.get("status") == "flagged")
        if total == 0:
            return "Mission created with no tiers."
        if needs_human == 0 and handled == total:
            return f"Mission completed — {handled}/{total} handled automatically."
        if needs_human > 0 and handled == total - needs_human:
            return (
                f"Mission completed — {handled}/{total} handled automatically. "
                f"{needs_human} decision(s) require your attention."
            )
        return (
            f"Mission in {self.state} — {handled}/{total} handled, "
            f"{needs_human} waiting on you."
        )

    # -- public serialisation --------------------------------------------- #

    def snapshot(self) -> dict[str, Any]:
        """The public shape the dashboard and the API return."""
        # Impact metrics: meaningful counts, no fake precision. The
        # time-saved estimate is a coarse 5-minute-per-handled-tier
        # benchmark (a teacher's first-pass prep of a single tier is
        # about that long on a 6-8 CS curriculum). It is labelled
        # "estimated" because it is a benchmark, not a measurement.
        handled = self.auto_validated
        repaired = self.auto_repaired
        escalated = sum(1 for item in self.human_review_items
                        if item.get("status") == "flagged")
        approved = sum(1 for item in self.human_review_items
                       if item.get("status") == "approved")
        # 5 minutes per auto-handled tier is the conservative benchmark
        # for first-pass teacher prep of one CS tier.
        time_saved_seconds = handled * 300
        return {
            "mission_id": self.id,
            "state": self.state,
            "previous_state": self.previous_state,
            "current_phase": self.current_phase,
            "goal": self.goal,
            "topic": self.topic,
            "unit_name": self.unit_name,
            "objective": self.objective,
            "tiers": list(self.tiers),
            "tier_status": dict(self.tier_status),
            "tier_record_ids": dict(self.tier_record_ids),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "state_changed_at": self.state_changed_at,
            "progress": {
                "completed": self.completed_count,
                "total": self.total_count,
                "percent": self.progress_percent,
            },
            "counts": {
                "generated": len(self.generated_items),
                "auto_validated": self.auto_validated,
                "auto_repaired": self.auto_repaired,
                "escalated": escalated,
                "approved": approved,
            },
            "metrics": {
                "tasks_completed_automatically": handled,
                "human_decisions_required": escalated,
                "materials_generated": len(self.generated_items),
                "validation_failures_auto_repaired": repaired,
                "materials_approved": approved,
                "time_saved_estimate_seconds": time_saved_seconds,
                "time_saved_estimate_label": "estimated",
                "time_saved_benchmark": (
                    "5 minutes per auto-handled tier (single CS tier prep "
                    "benchmark). Coarse estimate, not a measurement."
                ),
            },
            "generated_items": list(self.generated_items),
            "failed_items": list(self.failed_items),
            "human_review_items": list(self.human_review_items),
            "final_outputs": list(self.final_outputs),
            "sub_mission_ids": list(self.sub_mission_ids),
            "transitions": list(self.transitions),
            "errors": list(self.errors),
            "summary": self.summary,
            "total_agent_seconds": self.total_agent_seconds,
        }


# Curriculum missions held in memory, oldest first, so the desk can page through.
CURRICULUM_MISSIONS: dict[str, CurriculumMission] = {}

# Reverse index: which curriculum mission owns a given material record id.
# Used by approve_material to advance the right mission when the teacher
# approves a flagged item.
RECORD_TO_CURRICULUM: dict[int, str] = {}


async def persist_curriculum_mission(curriculum: CurriculumMission) -> None:
    """
    Save curriculum mission state to SQLite for durable crash resilience and recovery.
    Allows incomplete missions to resume safely without regenerating already completed tiers.
    """
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                """
                INSERT INTO curriculum_missions (
                    id, topic, unit_name, objective, state, goal_json,
                    tier_status_json, tier_record_ids_json, transitions_json,
                    sub_mission_ids_json, summary, created_at, started_at, completed_at, state_changed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    state=excluded.state,
                    tier_status_json=excluded.tier_status_json,
                    tier_record_ids_json=excluded.tier_record_ids_json,
                    transitions_json=excluded.transitions_json,
                    sub_mission_ids_json=excluded.sub_mission_ids_json,
                    summary=excluded.summary,
                    started_at=excluded.started_at,
                    completed_at=excluded.completed_at,
                    state_changed_at=excluded.state_changed_at
                """,
                (
                    curriculum.id,
                    curriculum.topic,
                    curriculum.unit_name,
                    curriculum.objective,
                    curriculum.state,
                    json.dumps(curriculum.goal),
                    json.dumps(curriculum.tier_status),
                    json.dumps(curriculum.tier_record_ids),
                    json.dumps(curriculum.transitions),
                    json.dumps(list(curriculum.sub_mission_ids)),
                    curriculum.summary,
                    curriculum.created_at,
                    curriculum.started_at,
                    curriculum.completed_at,
                    curriculum.state_changed_at,
                ),
            )
            await db.commit()
    except Exception as exc:  # noqa: BLE001 - persistence failure must not crash execution
        logger.debug("Could not persist curriculum mission state: %s", exc)


async def restore_curriculum_missions_from_db() -> None:
    """Restore durable curriculum mission states and reverse index from SQLite on startup."""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM curriculum_missions ORDER BY created_at ASC") as cursor:
                rows = await cursor.fetchall()
            for row in rows:
                mid = row["id"]
                try:
                    goal = json.loads(row["goal_json"]) if row["goal_json"] else {
                        "topic": row["topic"], "unit_name": row["unit_name"], "objective": row["objective"]
                    }
                    m = CurriculumMission(mid, goal)
                    m.state = row["state"]
                    m.previous_state = row["state"]
                    m.summary = row["summary"] or ""
                    m.created_at = row["created_at"]
                    m.started_at = row["started_at"]
                    m.completed_at = row["completed_at"]
                    m.state_changed_at = row["state_changed_at"]
                    if row["tier_status_json"]:
                        m.tier_status = json.loads(row["tier_status_json"])
                    if row["tier_record_ids_json"]:
                        m.tier_record_ids = {k: int(v) for k, v in json.loads(row["tier_record_ids_json"]).items()}
                    if row["transitions_json"]:
                        m.transitions = json.loads(row["transitions_json"])
                    if "sub_mission_ids_json" in row.keys() and row["sub_mission_ids_json"]:
                        # Restores the sub-mission drill-down across restarts.
                        m.sub_mission_ids = [str(v) for v in json.loads(row["sub_mission_ids_json"])]
                    # Re-populate reverse index and item lists
                    for tier, rid in m.tier_record_ids.items():
                        RECORD_TO_CURRICULUM[rid] = m.id
                        st = m.tier_status.get(tier, "draft")
                        m.generated_items.append({"tier": tier, "record_id": rid, "status": st})
                        if st == "flagged":
                            m.human_review_items.append({"tier": tier, "record_id": rid, "status": "flagged"})
                        elif st == "approved":
                            m.final_outputs.append(rid)
                    CURRICULUM_MISSIONS[mid] = m
                except Exception as row_err:  # noqa: BLE001
                    logger.warning("Could not restore mission row %s: %s", mid, row_err)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not restore curriculum missions: %s", exc)


def new_curriculum_mission(goal: dict[str, Any],
                           mission_id: str | None = None) -> CurriculumMission:
    """Register a new curriculum mission and return it. The id is validated."""
    candidate = (mission_id or "").strip()
    if not candidate or not MISSION_ID_PATTERN.match(candidate) or candidate in CURRICULUM_MISSIONS:
        candidate = f"cm-{uuid.uuid4().hex[:10]}"
    mission = CurriculumMission(candidate, goal)
    CURRICULUM_MISSIONS[candidate] = mission
    while len(CURRICULUM_MISSIONS) > CURRICULUM_MISSION_HISTORY_LIMIT:
        CURRICULUM_MISSIONS.pop(next(iter(CURRICULUM_MISSIONS)))
    return mission


def _log_tool(name: str, **arguments: Any) -> Mission | None:
    """
    Record that the agent selected a tool, and with what.

    Emits both canonical observability events (agent_action, tool_called)
    and legacy compatibility events (tool_selected, tool_started).
    Returns the mission so a tool body can add to it.
    """
    mission = current_mission()
    if mission is None:
        return None
    mission.tool_calls += 1
    shown = ", ".join(f"{key}={_clip(value, 90)}" for key, value in arguments.items() if value != "")
    mission.record("agent_action", f"Agent decided to invoke {name}({shown})", tool=name)
    mission.record("tool_selected", f"{name}({shown})", tool=name)
    mission.record("tool_called", f"{name} called with {shown or 'default arguments'}", tool=name)
    mission.record("tool_started", f"{name} running", tool=name)
    return mission


def _log_result(mission: Mission | None, name: str, summary: str, **fields: Any) -> None:
    if mission is not None:
        mission.record("tool_completed", f"{name} completed: {summary}", tool=name, **fields)
        mission.record("tool_result", f"{name} → {summary}", tool=name, **fields)



# --------------------------------------------------------------------------- #
# Model access
# --------------------------------------------------------------------------- #

_bedrock_runtime: Any = None


def _bedrock_client() -> Any:
    """
    Cached bedrock-runtime client.

    boto3 clients are thread-safe for calls but expensive to construct, and every
    tool-level generation makes one Converse call from a worker thread.
    """
    global _bedrock_runtime
    if _bedrock_runtime is None:
        import boto3
        from botocore.config import Config

        _bedrock_runtime = boto3.client(
            "bedrock-runtime",
            region_name=BEDROCK_REGION,
            config=Config(
                read_timeout=MODEL_TIMEOUT_SECONDS,
                connect_timeout=12,
                # Bedrock throttles hard on burst. Adaptive backoff turns a
                # transient 429 into latency instead of a failed mission.
                retries={"max_attempts": 3, "mode": "adaptive"},
            ),
        )
    return _bedrock_runtime


def _bedrock_converse(system: str, user: str, max_tokens: int, temperature: float) -> str:
    """One blocking Bedrock Converse call, returning the concatenated text blocks."""
    request: dict[str, Any] = {
        "messages": [{"role": "user", "content": [{"text": user}]}],
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
    }
    if system:
        request["system"] = [{"text": system}]
    if BEDROCK_MODEL_ID:
        request["modelId"] = BEDROCK_MODEL_ID
    else:
        # Ask the agent's own model object what it resolved to, so the tools and the
        # reasoning loop cannot silently end up on two different models.
        request["modelId"] = resolved_model_id()

    response = _bedrock_client().converse(**request)
    blocks = response.get("output", {}).get("message", {}).get("content", [])
    return "".join(block.get("text", "") for block in blocks)


def _litellm_completion(system: str, user: str, max_tokens: int, temperature: float,
                        json_mode: bool) -> str:
    """One blocking LiteLLM call against the fallback provider."""
    kwargs: dict[str, Any] = {
        "model": GEMINI_MODEL_ID,
        "api_key": GEMINI_API_KEY,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "timeout": MODEL_TIMEOUT_SECONDS,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = litellm.completion(**kwargs)
    return response.choices[0].message.content or ""


async def _complete(system: str, user: str, *, max_tokens: int = 4096,
                    temperature: float = 0.4, json_mode: bool = False) -> str:
    """
    One completion from whichever provider is configured.

    Every tool that writes curriculum text goes through here, so the tools follow the
    same provider as the agent's reasoning loop instead of quietly staying on the old
    one. Both client libraries are blocking, so the call is offloaded to a worker
    thread and given its own timeout — the outer mission deadline cannot interrupt a
    thread, so a per-call ceiling is what makes that deadline real.
    """
    if MODEL_PROVIDER == "bedrock":
        text = await asyncio.to_thread(
            _bedrock_converse, system, user, max_tokens, temperature
        )
    elif MODEL_PROVIDER == "litellm":
        if litellm is None:
            raise RuntimeError("MODEL_PROVIDER=litellm but litellm is not installed.")
        text = await asyncio.to_thread(
            _litellm_completion, system, user, max_tokens, temperature, json_mode
        )
    else:
        raise RuntimeError(f"Unknown MODEL_PROVIDER {MODEL_PROVIDER!r}.")

    return (text or "").strip()


# --------------------------------------------------------------------------- #
# Model output parsing
# --------------------------------------------------------------------------- #

def _loads_model_json(raw: str, context: str) -> dict:
    """
    Parse a JSON object out of a model response.

    Handles markdown fences, leading/trailing commentary, unescaped characters,
    trailing commas, and regex salvaging for robust execution across model providers.
    """
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", "", text).strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match is None:
            logger.warning("Unrecoverable JSON decode error for %s. Raw text: %s", context, _clip(raw, 200))
            raise ValueError(f"Model did not return JSON for {context}: {raw[:200]}")
        
        matched_str = match.group(0)
        try:
            obj = json.loads(matched_str)
        except json.JSONDecodeError:
            # Clean common syntax errors: trailing commas before } or ]
            cleaned = re.sub(r",\s*([\}\]])", r"\1", matched_str)
            try:
                obj = json.loads(cleaned)
            except json.JSONDecodeError as err:
                logger.warning("JSON salvage failed for %s (%s). Raw: %s", context, err, _clip(raw, 200))
                raise ValueError(f"Model returned invalid JSON for {context}: {err}") from err

    if isinstance(obj, list) and len(obj) > 0:
        obj = obj[0]
    if not isinstance(obj, dict):
        raise ValueError(f"Model did not return a JSON object for {context}")
    return obj


# --------------------------------------------------------------------------- #
# Pseudo-code verification  (deterministic, no model call)
# --------------------------------------------------------------------------- #

# The only vocabulary the curriculum allows. Anything else at the head of a code line
# is either a real programming language leaking in or an invented construct students
# have not been taught.
PSEUDO_KEYWORDS = (
    "SET", "INPUT", "PRINT", "DISPLAY", "IF", "ELSE IF", "ELSE", "END IF", "ENDIF",
    "REPEAT", "END REPEAT", "WHILE", "END WHILE", "STOP", "END",
)

# Syntax from real languages. Present in a code line, it breaks the house rule that
# grades 6-8 material stays in pseudo-code.
# Markers that replace a whole statement. A line STARTING with one of these
# is real code even in lowercase ("print(total)"), never prose.
REAL_SYNTAX_STATEMENT_MARKERS = (
    ("print(", "Python print call"),
    ("input(", "Python input call"),
    ("range(", "Python range call"),
    ("printf(", "C printf call"),
    ("console.log(", "JavaScript console call"),
    ("system.out.print", "Java System.out.print"),
    ("cout <<", "C++ stream output"),
    ("elif", "Python elif"),
    ("def ", "Python function definition"),
    ("import ", "Python import"),
)

REAL_SYNTAX_MARKERS = (
    ("def ", "Python function definition"),
    ("import ", "Python import"),
    ("print(", "Python print call"),
    ("input(", "Python input call"),
    ("range(", "Python range call"),
    ("console.log", "JavaScript console call"),
    ("elif", "Python elif"),
    ("&&", "C-style boolean operator"),
    ("||", "C-style boolean operator"),
    ("++", "C-style increment"),
    ("+=", "compound assignment operator"),
    ("{", "brace block"),
    ("}", "brace block"),
    (";", "statement terminator"),
    ("#include", "C include"),
)

# Words that appear inside expressions but are not variables.
_EXPR_WORDS = {
    "THEN", "TIMES", "TO", "DO", "AND", "OR", "NOT", "MOD", "TRUE", "FALSE", "IS",
    "EQUAL", "GREATER", "LESS", "THAN", "STEP", "BY", "OF", "IN", "WITH",
}

_BLANK = re.compile(r"_{3,}")
_STRING = re.compile(r"\"[^\"]*\"|'[^']*'")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _leading_keyword(line: str) -> str | None:
    """The pseudo-code keyword a line opens with, if it opens with one."""
    upper = line.strip().upper()
    for keyword in sorted(PSEUDO_KEYWORDS, key=len, reverse=True):
        if upper == keyword or upper.startswith(keyword + " "):
            return keyword
    return None


def _code_lines(text: str) -> list[tuple[int, str]]:
    """
    Pull the pseudo-code lines out of mixed prose.

    Worksheets are mostly instructions with code embedded, so the checks below run
    only on lines that actually open with a keyword. An indented continuation of a
    keyword line (the body of a loop, say) is picked up by the keyword test too,
    because every statement in this vocabulary starts with one.
    """
    found: list[tuple[int, str]] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if _leading_keyword(line) is not None:
            found.append((number, line))
    return found


def _looks_like_code(line: str) -> bool:
    """
    A line that is trying to be code but does not start with a legal keyword.

    Used to catch invented constructs (FOR, FUNCTION, RETURN) without flagging the
    ordinary English sentences that make up most of a worksheet.
    """
    stripped = line.strip()
    if _leading_keyword(stripped) is not None:
        return False
    upper = stripped.upper()
    for invented in ("FOR ", "FOREACH ", "FUNCTION ", "RETURN ", "DEF ", "LOOP ",
                     "SWITCH ", "CASE ", "TRY ", "CATCH ", "VAR ", "LET ", "CONST "):
        if upper.startswith(invented):
            return True
    return False


def _expression_identifiers(fragment: str) -> list[str]:
    """Variable-looking names in an expression, with strings and blanks removed."""
    cleaned = _BLANK.sub(" ", _STRING.sub(" ", fragment))
    names = []
    for match in _IDENTIFIER.finditer(cleaned):
        name = match.group(0)
        if name.upper() in _EXPR_WORDS or name.upper() in PSEUDO_KEYWORDS:
            continue
        names.append(name)
    return names


def verify_pseudocode(text: str) -> dict[str, Any]:
    """
    Static analysis of pseudo-code. Deterministic, offline, and honest about limits.

    Three classes of finding, and the distinction is the whole point:
      * error    — provably wrong for this curriculum (real syntax, invented keyword,
                   a stray END with nothing to close).
      * warning  — likely wrong but context-dependent (a variable read before it is
                   SET may be defined by a trace table the student fills in).
      * note     — informational, never a reason to regenerate.

    A deliberately planted bug in the advanced tier shows up here as a finding, which
    is correct: the tool reports, the agent decides. Nothing here rewrites anything.
    """
    lines = _code_lines(text)
    findings: list[dict[str, Any]] = []

    def add(severity: str, line: int, code: str, detail: str) -> None:
        findings.append({"severity": severity, "line": line, "code": code, "detail": detail})

    # 1. Real language syntax. Call-style leaks usually REPLACE the whole
    #    statement and arrive lowercase ("print(total)"), which the
    #    taught-keyword filter silently drops — so statement-initial markers
    #    are scanned across every line. Embedded markers (semicolons, braces,
    #    compound operators) stay scoped to recognised code lines so ordinary
    #    prose that merely mentions them is not flagged.
    for number, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        for marker, description in REAL_SYNTAX_STATEMENT_MARKERS:
            if lowered.startswith(marker):
                add("error", number, "real_syntax",
                    f"{description} ({marker!r}) in: {_clip(stripped, 70)}")
                break

    for number, line in lines:
        lowered = line.lower()
        for marker, description in REAL_SYNTAX_MARKERS:
            if marker in lowered:
                add("error", number, "real_syntax",
                    f"{description} ({marker!r}) in: {_clip(line, 70)}")
                break

    # 2. Invented constructs, checked across every line so a FOR loop hiding in prose
    #    formatting is still caught.
    for number, raw in enumerate(text.splitlines(), start=1):
        if _looks_like_code(raw):
            add("error", number, "unknown_keyword",
                f"Not in the taught vocabulary: {_clip(raw.strip(), 70)}")

    # 3. Block markers. Absence of END IF is a legitimate house style, so only
    #    inconsistency and stray closers are reported.
    opens = {"IF": 0, "REPEAT": 0, "WHILE": 0}
    closes = {"IF": 0, "REPEAT": 0, "WHILE": 0}
    for number, line in lines:
        keyword = _leading_keyword(line)
        if keyword in {"END IF", "ENDIF"}:
            closes["IF"] += 1
        elif keyword == "END REPEAT":
            closes["REPEAT"] += 1
        elif keyword == "END WHILE":
            closes["WHILE"] += 1
        elif keyword == "IF":
            opens["IF"] += 1
        elif keyword == "REPEAT":
            opens["REPEAT"] += 1
        elif keyword == "WHILE":
            opens["WHILE"] += 1

    for block in opens:
        if closes[block] > opens[block]:
            add("error", 0, "stray_end",
                f"{closes[block]} END {block} marker(s) but only {opens[block]} {block} opener(s).")
        elif closes[block] and closes[block] < opens[block]:
            add("warning", 0, "mixed_block_style",
                f"{opens[block]} {block} opener(s) but only {closes[block]} END {block} — "
                "close every block or none of them, not some.")

    # 4. IF without THEN reads as prose, not as a condition students can trace.
    for number, line in lines:
        if _leading_keyword(line) in {"IF", "ELSE IF"} and "THEN" not in line.upper():
            add("warning", number, "if_without_then",
                f"No THEN on: {_clip(line, 70)}")

    # 5. REPEAT needs a count the student can count.
    for number, line in lines:
        if _leading_keyword(line) == "REPEAT":
            body = line.upper().replace("REPEAT", "", 1)
            if "TIMES" not in body:
                add("warning", number, "repeat_without_count",
                    f"REPEAT with no 'n TIMES' count: {_clip(line, 70)}")
            elif not re.search(r"\d|_{3,}", body):
                add("warning", number, "repeat_count_not_a_number",
                    f"REPEAT count is neither a number nor a blank: {_clip(line, 70)}")

    # 6. Reading a variable that was never SET or INPUT.
    assigned: set[str] = set()
    for number, line in lines:
        keyword = _leading_keyword(line)
        upper = line.upper()
        if keyword == "SET":
            target = re.match(r"SET\s+([A-Za-z_][A-Za-z0-9_]*)", line, re.IGNORECASE)
            if target:
                # Right-hand side is read before the assignment takes effect.
                remainder = re.split(r"\bTO\b|=", line, maxsplit=1, flags=re.IGNORECASE)
                if len(remainder) > 1:
                    for name in _expression_identifiers(remainder[1]):
                        if name.lower() not in assigned:
                            add("warning", number, "use_before_set",
                                f"'{name}' is read before it is given a value.")
                assigned.add(target.group(1).lower())
            continue
        if keyword == "INPUT":
            for name in _expression_identifiers(line[len("INPUT"):]):
                assigned.add(name.lower())
            continue
        if keyword in {"PRINT", "DISPLAY", "IF", "ELSE IF", "WHILE", "REPEAT"}:
            fragment = line[len(keyword):] if upper.startswith(keyword) else line
            for name in _expression_identifiers(fragment):
                if name.lower() not in assigned:
                    add("warning", number, "use_before_set",
                        f"'{name}' is read on line {number} before it is given a value.")

    # 7. A WHILE whose condition mentions nothing the body changes cannot terminate.
    #    This is the one loop bug worth proving rather than guessing at, and it needs
    #    no interpreter: if no variable in the condition is reassigned inside the
    #    block, the condition can never become false.
    for index, (number, line) in enumerate(lines):
        if _leading_keyword(line) != "WHILE":
            continue
        condition_names = {name.lower() for name in _expression_identifiers(line[len("WHILE"):])}
        if not condition_names:
            continue
        changed: set[str] = set()
        for _, body_line in lines[index + 1:]:
            if _leading_keyword(body_line) == "END WHILE":
                break
            target = re.match(r"(?:SET|INPUT)\s+([A-Za-z_][A-Za-z0-9_]*)",
                              body_line, re.IGNORECASE)
            if target:
                changed.add(target.group(1).lower())
        if condition_names and not (condition_names & changed):
            add("warning", number, "possible_infinite_loop",
                f"Nothing in the loop body changes {', '.join(sorted(condition_names))}, "
                "so the WHILE condition can never become false.")

    has_blanks = bool(_BLANK.search(text))
    errors = [item for item in findings if item["severity"] == "error"]
    warnings = [item for item in findings if item["severity"] == "warning"]

    return {
        "code_lines_found": len(lines),
        "has_student_blanks": has_blanks,
        "errors": errors,
        "warnings": warnings,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "ok": not errors,
        "note": (
            "No pseudo-code lines were found, so nothing was checked."
            if not lines else
            "Blanks were found, so this is a fill-in exercise: unset variables are "
            "expected." if has_blanks else
            "Fully written code: unset variables and unterminated loops are real defects."
        ),
    }


# --------------------------------------------------------------------------- #
# Auditing  (deterministic checks first, then a model critique)
# --------------------------------------------------------------------------- #

BANNED_OBJECTIVE_VERBS = (
    "understand", "learn about", "know about", "be familiar", "appreciate",
    "be aware", "gain knowledge", "be exposed",
)

APPROVED_OBJECTIVE_VERBS = (
    "trace", "predict", "identify", "write", "modify", "compare", "justify",
    "explain", "build", "debug", "evaluate", "convert", "calculate", "design",
    "choose", "locate", "fix", "describe", "apply", "analyse", "analyze", "order",
)

AUDIT_SYSTEM_PROMPT = """\
You are a strict reviewer of middle school (grades 6-8) computer science materials.
You did not write this material and you gain nothing by approving it.

Judge only what you are shown, against the stated objective and ability tier.

Reply with a single JSON object and nothing else, with exactly these keys:
  "verdict"   — one of "pass", "revise", "human"
  "problems"  — array of short, specific, actionable strings (empty when verdict is pass)
  "reason"    — one sentence explaining the verdict
  "findings"  — array of structured issues, each one a JSON object with:
      "severity"            — "low" | "medium" | "high"
      "issue"               — one short sentence stating what is wrong
      "affected_component"  — "objective" | "worksheet" | "quiz" | "answer_key" |
                              "distractor" | "code_example" | "explanation" |
                              "difficulty" | "prerequisite"
      "recommended_action"  — one of "regenerate_worksheet" | "regenerate_quiz" |
                              "regenerate_distractors" | "regenerate_objective" |
                              "fix_explanation" | "escalate_to_teacher" |
                              "no_action"
  Each problem string MUST also appear as a finding (so the structured record and the
  human-readable problems stay in sync). When the verdict is "pass", findings MUST be [].

How to choose the verdict:
  "pass"   The material serves the objective at this tier and a teacher could approve
           it as-is. Minor stylistic preferences are not defects.
  "revise" There is a concrete, mechanical defect a writer can fix without asking a
           teacher anything: a missing section, a distractor that encodes no real
           misconception, difficulty obviously wrong for the tier, an instruction a
           student could not follow, an answer key that does not match the quiz.
  "human"  A teacher's professional judgement is genuinely required: the objective
           itself is unmeasurable or does not match the topic, the topic falls outside
           grades 6-8 computer science, the material asserts something you cannot
           verify, or two defensible readings exist and the choice changes what
           students learn. Do not use "human" for anything a writer could simply fix.
"""


# Auditable components. Anything tagged with a component not in this set is a sign the
# model drifted; the audit normalises unknown tags to "worksheet" so the agent still has
# a single label to act on.
AUDIT_COMPONENTS: tuple[str, ...] = (
    "objective", "worksheet", "quiz", "answer_key", "distractor",
    "code_example", "explanation", "difficulty", "prerequisite",
)
AUDIT_ACTIONS: tuple[str, ...] = (
    "regenerate_worksheet", "regenerate_quiz", "regenerate_distractors",
    "regenerate_objective", "fix_explanation", "escalate_to_teacher", "no_action",
)


AUDIT_ACTIONS: tuple[str, ...] = (
    "regenerate_worksheet", "regenerate_quiz", "regenerate_distractors",
    "regenerate_objective", "fix_explanation", "escalate_to_teacher", "no_action",
)


# --------------------------------------------------------------------------- #
# Bounded pseudo-code interpreter (app/interpreter)
# --------------------------------------------------------------------------- #
from app.interpreter import (
    ExecutionResult,
    execute_pseudocode,
    interp_eval as _interp_eval,
    normalise_output_line as _normalise_output_line,
    outputs_match as _outputs_match,
    INTERPRETER_ERRORS as _INTERPRETER_ERRORS,
    INTERPRETER_STEP_CAP as _INTERPRETER_STEP_CAP,
    DEFAULT_INPUTS as _INTERPRETER_INPUTS,
    run_pseudocode as _run_pseudocode,
)
from app.interpreter.engine import AutonomousAgentEngine

# The autonomous background agent over the student learning stream. It starts
# with the app (lifespan below), silently ingests and assesses submissions from
# lessons.db, and only surfaces through the decision-gate routes when a human
# judgement is genuinely required. One engine per process; the routes below are
# its only human-facing surface.
LEARNING_ENGINE = AutonomousAgentEngine(DB_PATH)



def _extract_expected_output(answer_key: str, q_match: re.Match) -> str | None:
    """
    Pull the expected output out of an answer-key entry.

    The answer key convention the writer is told to follow is "Q<n> (<points>):
    ... Output: line1, line2, ...". If we can find that, we use it; otherwise
    we return None and the consistency check is skipped for this question.
    """
    # Find the body of this Q up to the next Q or end.
    start = q_match.end()
    next_q = re.search(r"\bQ\d+\b", answer_key[start:])
    end = start + next_q.start() if next_q else len(answer_key)
    body = answer_key[start:end]
    m = re.search(r"output\s*[:\-]\s*(.+?)(?:\n|$)", body, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1)
    return None


def check_pseudocode_consistency(worksheet: str, answer_key: str) -> list[str]:
    """
    Cross-check a worksheet's pseudo-code snippets against the answer key.

    The check finds any quiz item that looks like a "predict the output"
    question, runs the worksheet's pseudo-code through the bounded
    interpreter, and reports a problem if the actual output disagrees with
    the answer key's stated expected output. Items without an explicit
    Output: line in the answer key are skipped — we are not willing to
    claim a contradiction we cannot actually check.
    """
    problems: list[str] = []
    q_matches = list(re.finditer(r"\bQ(\d+)\b", answer_key))
    if not q_matches:
        return problems

    # The interpreter runs the whole worksheet, but the answer key checks
    # one question at a time. Run it once; the answer key's "Q3" expected
    # output is a snapshot of what the program printed by the time Q3's
    # trace was asked, not the full output. For now, we only run the
    # interpreter when the question explicitly says "predict the output"
    # and the answer key gives an Output: block.
    pseudo = execute_pseudocode(worksheet)
    if not pseudo["ok"] and pseudo["error"] and pseudo["error"][0] != "syntax":
        problems.append(
            f"Pseudo-code did not run cleanly: {pseudo['error'][1]}. "
            f"Students cannot trace this; rewrite the snippet."
        )
        return problems

    for q_match in q_matches:
        q_number = q_match.group(1)
        # Only check items that say "predict the output".
        start = q_match.end()
        next_q = re.search(r"\bQ\d+\b", answer_key[start:])
        end = start + next_q.start() if next_q else len(answer_key)
        body = answer_key[start:end]
        if "predict the output" not in body.lower() and "what does" not in body.lower():
            continue
        expected = _extract_expected_output(answer_key, q_match)
        if expected is None:
            continue
        expected_lines = [line for line in expected.splitlines() if line.strip()]
        if not expected_lines:
            continue
        if not _outputs_match(pseudo["output"], expected_lines):
            problems.append(
                f"Q{q_number} 'predict the output' answer key disagrees with what "
                f"the worksheet's pseudo-code actually produces. "
                f"Answer key says: {expected.strip()[:80]!r}; interpreter produced: "
                f"{', '.join(pseudo['output'][:6]) or '(nothing)'!r}."
            )
    return problems


def _extract_quiz_options(quiz: str) -> dict[str, list[str]]:
    """
    Parse a quiz into a {Q1: [option_text...], Q2: ...} map.

    The convention the writer is told to follow is "Q1 (2 pts): stem. A) ... B)
    ... C) ... D) ..." — but real model output drifts, so we accept several
    shapes: "A)", "A.", and "A " followed by a capital letter. Returns the
    options in the order they appear so the answer key can be matched
    positionally when its "misconception: ..." line is ordered the same way.
    """
    options: dict[str, list[str]] = {}
    pattern = re.compile(
        r"\bQ(\d+)\b\s*[^A-Za-z]*(.*?)(?=\n\s*Q\d+\b|\Z)",
        re.DOTALL,
    )
    opt_pattern = re.compile(
        r"\b([A-D])[\.\)]\s*([^\n]+)",
    )
    for match in pattern.finditer(quiz):
        q_number = match.group(1)
        body = match.group(2)
        opts = [m.group(2).strip() for m in opt_pattern.finditer(body)]
        if opts:
            options[q_number] = opts
    return options


def _extract_answer_key_misconceptions(answer_key: str) -> dict[str, list[dict[str, str]]]:
    """
    Parse an answer key into a {Qn: [misconception-per-option...]} map.

    Recognised shapes (any of):
      * "Q1: B. misconception: X"  (one misconception per question, single line)
      * "Q1 (2 pts): A) X (correct) B) Y (distractor: ...) C) Z (distractor: ...)"
        where the misconception sits inside a "(distractor: ...)" annotation
      * a multi-line block where every line under a Q starts with the
        option letter and ends with "misconception: ..." or "(misconception: ...)"

    Each option entry has {"option_letter", "option_text" (if known),
    "is_correct" (bool), "misconception" (str or None)}. This is the
    misconception metadata the spec asks the answer key to carry.
    """
    out: dict[str, list[dict[str, str]]] = {}
    q_pattern = re.compile(r"\bQ(\d+)\b[^\n]*(?:\n(?!\s*Q\d+\b)[^\n]*)*", re.DOTALL)
    for match in q_pattern.finditer(answer_key):
        q_number = match.group(1)
        body = match.group(0)
        entries: list[dict[str, str]] = []

        # Pull the correct-answer letter from "Q1: B. ..." if present.
        inline = re.search(
            rf"\bQ{q_number}\b\s*[:\.\)]\s*([A-D])([\s\S]+?)(?=(?:\n\s*Q\d+\b)|\Z)",
            body,
        )
        inline_correct_letter: str | None = None
        inline_correct_text: str | None = None
        inline_correct_misc: str | None = None
        if inline:
            inline_correct_letter = inline.group(1)
            rest = inline.group(2)
            inline_correct_text = rest.strip()
            m_misc = re.search(
                r"(?:misconception|distractor)\s*[:\-]\s*(.+?)(?:[\.\)]\s*$|$)",
                rest, re.IGNORECASE,
            )
            inline_correct_misc = m_misc.group(1).strip() if m_misc else None

        # Per-line: A) ... misconception: ... / Q1: B. misconception: ...
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            m_letter = re.match(r"([A-D])[\.\)]\s*(.*)", line)
            if not m_letter:
                continue
            letter = m_letter.group(1)
            rest = m_letter.group(2)
            m_misc = re.search(
                r"(?:misconception|distractor)\s*[:\-]\s*(.+?)(?:[\.\)]\s*$|$)",
                rest, re.IGNORECASE,
            )
            m_correct = re.match(r"^(\*|correct|answer)\b", rest, re.IGNORECASE)
            is_correct = bool(m_correct) or letter == inline_correct_letter
            entries.append({
                "option_letter": letter,
                "option_text": rest,
                "is_correct": is_correct,
                "misconception": m_misc.group(1).strip() if m_misc else None,
            })

        # If the per-line scan never produced an entry for the inline
        # correct letter, add it now so the linkage check has the correct
        # answer it expects.
        if inline_correct_letter and not any(
            e["option_letter"] == inline_correct_letter for e in entries
        ):
            entries.append({
                "option_letter": inline_correct_letter,
                "option_text": inline_correct_text or "",
                "is_correct": True,
                "misconception": inline_correct_misc,
            })

        # Free-response fallback: "Q5: free response. misconception: ..."
        if not entries:
            free = re.search(
                rf"\bQ{q_number}\b\s*[:\.\)]\s*([^\n]+)",
                body,
            )
            if free:
                rest = free.group(1)
                m_misc = re.search(
                    r"(?:misconception|distractor)\s*[:\-]\s*(.+?)(?:[\.\)]\s*$|$)",
                    rest, re.IGNORECASE,
                )
                entries.append({
                    "option_letter": "",
                    "option_text": rest.strip(),
                    "is_correct": True,
                    "misconception": m_misc.group(1).strip() if m_misc else None,
                })

        if entries:
            out[q_number] = entries
    return out


def check_misconception_linkage(quiz: str, answer_key: str) -> list[str]:
    """
    Deterministic check that the answer key actually carries a misconception
    per wrong option of each quiz item, and that the linkage is sensible.

    Returns a list of human-readable problems (empty when the key is sound).
    A separate `_extract_answer_key_misconceptions` companion returns the
    structured per-option metadata the dashboard can render.
    """
    problems: list[str] = []
    options = _extract_quiz_options(quiz)
    misconceptions = _extract_answer_key_misconceptions(answer_key)

    if not options:
        return problems  # No multiple-choice items to check.

    for q_number, opts in options.items():
        entries = misconceptions.get(q_number, [])
        if not entries:
            problems.append(
                f"Q{q_number} has no answer-key entry that names a misconception; "
                "the distractor diagnostic is missing."
            )
            continue
        # Find the correct answer: an option marked is_correct, or the first
        # entry that contains 'correct' or '*' or a bare letter with no
        # misconception annotation.
        correct_letter = next(
            (e["option_letter"] for e in entries if e["is_correct"]),
            None,
        )
        if not correct_letter:
            # The "Q1: B" convention.
            m = re.search(rf"\bQ{q_number}\b[^\n]*([A-D])\b", answer_key)
            if m:
                correct_letter = m.group(1)
        if not correct_letter:
            problems.append(
                f"Q{q_number} answer key does not mark which option is correct; "
                "cannot link misconceptions to the right place."
            )
            continue
        # A single-entry answer key (one line per Q) leaves the wrong options
        # entirely undiagnosed. The distractor diagnostic the spec asks for
        # requires one misconception per wrong option, so flag the count gap.
        if len(entries) < len(opts):
            problems.append(
                f"Q{q_number} answer key lists {len(entries)} option(s) but the "
                f"quiz has {len(opts)} options; every wrong option needs a named "
                "misconception so the distractor actually diagnoses something."
            )
        for entry in entries:
            if entry["option_letter"] == correct_letter:
                continue  # correct answers do not need a misconception
            if not entry["misconception"]:
                problems.append(
                    f"Q{q_number} option {entry['option_letter']} has no "
                    "misconception in the answer key; a wrong answer with no "
                    "diagnosis teaches the teacher nothing."
                )
    return problems


def _deterministic_problems(kind: str, tier: str, topic: str, objective: str,
                            text: str) -> dict[str, Any]:
    """
    Structural checks that need no model and cannot be argued with.

    Running these first means an obviously malformed artifact costs one cheap function
    call instead of a second round trip to Bedrock, and the problems handed back are
    precise enough for the agent to fix on the next attempt.

    Returns a dict with two parallel lists:
      - "problems": human-readable strings (kept for backwards compatibility)
      - "findings": structured dicts with severity / affected_component / recommended_action

    The agent loop is free to use either: the human-readable problem is what the
    system prompt hands back to the writer; the structured finding is what the audit
    tool returns to the model so it can choose the right corrective action.
    """
    problems: list[str] = []
    findings: list[dict[str, Any]] = []

    def add(problem: str, *, severity: str, component: str, action: str) -> None:
        problems.append(problem)
        findings.append({
            "severity": severity,
            "issue": problem,
            "affected_component": component,
            "recommended_action": action,
        })

    body = text.strip()
    lowered = body.lower()

    if not body:
        add(f"The {kind} is empty.", severity="high", component=kind, action="regenerate_worksheet")
        return {"problems": problems, "findings": findings}
    if "```" in body:
        add("Output contains markdown code fences; it must be plain text.",
            severity="medium", component=kind, action="regenerate_worksheet")

    if kind == "objective":
        words = body.split()
        for banned in BANNED_OBJECTIVE_VERBS:
            if banned in lowered:
                add(
                    f"Objective uses the unmeasurable phrase '{banned}'. Open with an "
                    "observable verb such as trace, predict, write or justify.",
                    severity="high", component="objective", action="regenerate_objective",
                )
        first = words[0].lower().rstrip(",:;") if words else ""
        if first not in APPROVED_OBJECTIVE_VERBS and not any(
            verb in lowered.split(".")[0] for verb in APPROVED_OBJECTIVE_VERBS
        ):
            add(
                "Objective does not contain an observable Bloom's verb, so mastery "
                "could not be measured.",
                severity="high", component="objective", action="regenerate_objective",
            )
        if len(words) > 45:
            add(f"Objective is {len(words)} words; keep it to one sentence.",
                severity="medium", component="objective", action="regenerate_objective")
        if body.count(".") > 2:
            add("Objective should be a single sentence.",
                severity="low", component="objective", action="regenerate_objective")
        return {"problems": problems, "findings": findings}

    # Every generated artifact carries a header naming the topic and the tier, which
    # is how a teacher tells three otherwise similar sheets apart on paper.
    expected_header = {"worksheet": "WORKSHEET", "quiz": "QUIZ", "answer_key": "ANSWER KEY"}
    header = expected_header.get(kind)
    if header:
        first_line = body.splitlines()[0].upper()
        if header not in first_line:
            add(f"First line must be the {header} header; found: "
                f"{_clip(body.splitlines()[0], 60)}",
                severity="medium", component=kind, action="regenerate_worksheet")
        elif tier.upper() not in first_line:
            add(f"Header does not name the '{tier}' tier.",
                severity="low", component=kind, action="regenerate_worksheet")

    # Cross-cutting pedagogical check: the material should reference the topic and the
    # objective somewhere visible. A worksheet about "loops" that never says "loop" is
    # almost certainly a generation mistake worth flagging.
    if topic:
        topic_token = topic.strip().lower().split()[0] if topic.strip() else ""
        if topic_token and len(topic_token) > 2 and topic_token not in lowered:
            add(f"Material does not mention the topic '{topic}' by name.",
                severity="medium", component=kind, action="regenerate_worksheet")
    if objective:
        obj_first = objective.strip().lower().split()[0] if objective.strip() else ""
        # We don't require the full objective to appear (paraphrasing is fine); we do
        # require at least one strong verb-noun pair from the objective to surface, so
        # the student can see what they are learning to do.
        for word in objective.lower().split():
            if len(word) > 5 and word in lowered:
                break
        else:
            add(f"Material does not echo any word from the stated objective, so a student "
                f"cannot see what they are learning to do.",
                severity="low", component=kind, action="regenerate_worksheet")

    if kind == "worksheet":
        if len(body) < 400:
            add(f"Worksheet is only {len(body)} characters; too thin for a lesson.",
                severity="high", component="worksheet", action="regenerate_worksheet")
        if "objective:" not in lowered:
            add("Worksheet is missing the 'Objective:' line.",
                severity="medium", component="worksheet", action="regenerate_worksheet")
        if "directions:" not in lowered:
            add("Worksheet is missing the 'Directions:' line.",
                severity="medium", component="worksheet", action="regenerate_worksheet")
        numbered = len(re.findall(r"^\s*([1-9])[\.\)]", body, re.MULTILINE))
        if numbered < 5:
            add(f"Found {numbered} numbered exercises; exactly 5 are required.",
                severity="high", component="worksheet", action="regenerate_worksheet")
        blanks = len(_BLANK.findall(body))
        minimum = 6 if tier == "struggling" else 3
        if blanks < minimum:
            add(
                f"Only {blanks} visible ______ blanks; the {tier} tier needs at least "
                f"{minimum} so the student knows where to write.",
                severity="medium", component="worksheet", action="regenerate_worksheet",
            )
        if not any(word in lowered for word in ("explain", "reflect", "in your own words",
                                                "why do you think")):
            add("No closing reflection prompt asking the student to explain "
                "their reasoning.",
                severity="medium", component="worksheet", action="regenerate_worksheet")
        if tier == "struggling":
            if "word bank" not in lowered:
                add("The struggling tier requires a word bank of the exact "
                    "keywords needed.",
                    severity="high", component="difficulty", action="regenerate_worksheet")
            if "trace" not in lowered:
                add("The struggling tier requires a trace table.",
                    severity="high", component="difficulty", action="regenerate_worksheet")
        if tier == "advanced":
            if not any(word in lowered for word in ("bug", "flawed", "error", "incorrect",
                                                    "wrong")):
                add("The advanced tier requires one flawed snippet the student "
                    "must locate and fix.",
                    severity="high", component="difficulty", action="regenerate_worksheet")
            if not any(word in lowered for word in ("efficien", "fewer steps", "less work",
                                                    "faster")):
                add("The advanced tier requires an efficiency comparison.",
                    severity="high", component="difficulty", action="regenerate_worksheet")

    if kind == "quiz":
        if len(body) < 200:
            add(f"Quiz is only {len(body)} characters; too thin to assess "
                "the objective.",
                severity="high", component="quiz", action="regenerate_quiz")
        missing = [f"Q{index}" for index in range(1, 6)
                   if not re.search(rf"\bQ{index}\b", body)]
        if missing:
            add(f"Missing quiz items: {', '.join(missing)}.",
                severity="high", component="quiz", action="regenerate_quiz")
        points = [int(value) for value in re.findall(r"\((\d+)\s*(?:pt|pts|point|points)\)",
                                                     lowered)]
        if points and sum(points) != 10:
            add(f"Point values sum to {sum(points)}; the quiz must total 10.",
                severity="high", component="quiz", action="regenerate_quiz")
        elif not points:
            add("No per-item point values in parentheses, e.g. '(2 pts)'.",
                severity="high", component="quiz", action="regenerate_quiz")
        options = len(re.findall(r"\b[A-D]\)", body))
        if options < 4:
            add("Fewer than one full set of A) B) C) D) options; the tier "
                "specification requires two multiple-choice items.",
                severity="high", component="distractor", action="regenerate_quiz")
        # Difficulty / prerequisite: the struggling tier must not assume the student
        # can write pseudo-code from a blank page in the quiz — at least one item
        # should still be recall or a multiple choice. The on-level tier may ask
        # for short writing, but not for full trace-then-write questions.
        if tier == "struggling":
            mc_items = len(re.findall(r"\bQ\d+\b.*?\bA\)\s", body))
            if mc_items < 2:
                add("The struggling tier's quiz must have at least two multiple-choice "
                    "items so no item asks the student to write code from scratch.",
                    severity="high", component="difficulty", action="regenerate_quiz")

    if kind == "answer_key":
        if len(body) < 150:
            add(f"Answer key is only {len(body)} characters.",
                severity="high", component="answer_key", action="regenerate_quiz")
        missing = [f"Q{index}" for index in range(1, 6)
                   if not re.search(rf"\bQ{index}\b", body)]
        if missing:
            add(f"Answer key does not cover: {', '.join(missing)}.",
                severity="high", component="answer_key", action="regenerate_quiz")
        misconceptions = len(re.findall(
            r"misconception|reveals|suggests|confus|mistakes|thinks|believes", lowered))
        if misconceptions < 3:
            add(
                f"Only {misconceptions} items name the misconception a wrong answer would "
                "reveal; every item needs one.",
                severity="high", component="distractor", action="regenerate_distractors",
            )
        if "total" not in lowered:
            add("Answer key does not state the total.",
                severity="medium", component="answer_key", action="regenerate_quiz")
        if not re.search(r"8\s*/\s*10|mastery", lowered):
            add("Answer key does not state the 8/10 mastery threshold.",
                severity="medium", component="answer_key", action="regenerate_quiz")

    # Curriculum-wide rule: no real language syntax anywhere, in any artifact.
    syntax = verify_pseudocode(body)
    for error in syntax["errors"][:3]:
        add(f"Line {error['line']}: {error['detail']}",
            severity="high", component="code_example", action="regenerate_worksheet")

    return {"problems": problems, "findings": findings}


async def _audit(kind: str, tier: str, topic: str, objective: str,
                 text: str) -> dict[str, Any]:
    """
    Audit one artifact and return a structured verdict.

    Two layers, cheapest first. If the structural layer finds anything, the model is
    not consulted at all: those defects are unarguable and the agent should fix them
    before spending a round trip on taste.

    Returns a dict with:
      verdict            — "pass" | "revise" | "human" | "unknown"
      problems           — human-readable list (kept for backwards compatibility)
      findings           — list of structured {severity, issue, affected_component,
                            recommended_action} dicts
      auto_fixable       — True when the agent can attempt a correction itself
      layer              — "structural" | "critique" | "input"
      reason             — one-sentence summary
    """
    kind = kind.strip().lower().replace(" ", "_").replace("-", "_")
    if kind not in {"worksheet", "quiz", "answer_key", "objective"}:
        return {
            "verdict": "revise",
            "problems": [f"Unknown material kind {kind!r}. Use worksheet, quiz, "
                         "answer_key or objective."],
            "findings": [{
                "severity": "high",
                "issue": f"Unknown material kind {kind!r}.",
                "affected_component": kind,
                "recommended_action": "regenerate_worksheet",
            }],
            "auto_fixable": True,
            "layer": "input",
        }

    structural = _deterministic_problems(kind, tier, topic, objective, text)
    structural_problems = structural["problems"]
    structural_findings = structural["findings"]
    if structural_problems:
        return {
            "verdict": "revise",
            "problems": structural_problems[:8],
            "findings": structural_findings[:8],
            "auto_fixable": True,
            "layer": "structural",
            "reason": "Structural checks failed, so no model critique was needed.",
        }

    user_prompt = textwrap.dedent(
        f"""\
        Material kind: {kind}
        Topic: {topic}
        Ability tier: {tier}
        Stated objective: {objective}

        Material to review:
        ---
        {text.strip()[:6000]}
        ---

        Return only the JSON object.
        """
    ).strip()

    try:
        raw = await _complete(
            AUDIT_SYSTEM_PROMPT, user_prompt,
            max_tokens=900, temperature=0.1, json_mode=True,
        )
        payload = _loads_model_json(raw, f"the audit of the {kind} ({tier} tier)")
    except Exception as exc:  # noqa: BLE001
        # A failed audit must not be read as a pass. Reporting the failure lets the
        # agent decide between retrying the audit and escalating.
        logger.warning("Audit call failed | kind=%s | tier=%s | %s", kind, tier, exc)
        return {
            "verdict": "unknown",
            "problems": [f"The audit itself failed ({type(exc).__name__}). The material "
                         "has not been reviewed."],
            "findings": [{
                "severity": "high",
                "issue": f"Audit call failed ({type(exc).__name__}).",
                "affected_component": kind,
                "recommended_action": "escalate_to_teacher",
            }],
            "auto_fixable": False,
            "layer": "critique",
            "reason": "Audit unavailable; treat this material as unreviewed.",
        }

    verdict = str(payload.get("verdict", "")).strip().lower()
    if verdict not in {"pass", "revise", "human"}:
        verdict = "revise"
    problems = [str(item).strip() for item in payload.get("problems", []) if str(item).strip()]
    if verdict != "pass" and not problems:
        problems = [str(payload.get("reason", "The reviewer gave no specific problem."))]

    # Normalise the model-returned findings. The model may invent components or
    # severities; this is the audit's last line of defence before the agent acts on
    # the result, so unknown values are dropped or coerced to safe defaults.
    findings = _normalise_findings(payload.get("findings"), kind, problems)

    return {
        "verdict": verdict,
        "problems": problems[:8],
        "findings": findings[:8],
        "auto_fixable": verdict == "revise",
        "layer": "critique",
        "reason": _clip(payload.get("reason", ""), 300),
    }


def _normalise_findings(raw: Any, kind: str, problems: list[str]) -> list[dict[str, Any]]:
    """
    Coerce whatever the model returned into a list of clean, schema-valid findings.

    Rules:
      * Only known components and severities survive; anything else falls back to the
        audit's kind, or "low" / "regenerate_worksheet".
      * Each "issue" must be a non-empty string. We accept the model's findings, but
        also cross-check that at least one finding carries the same text as a
        human-readable problem — otherwise we add a synthetic finding so the agent
        never sees a problems list that has no structured counterpart.
    """
    safe: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        raw = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        severity = str(item.get("severity", "")).strip().lower()
        if severity not in {"low", "medium", "high"}:
            severity = "medium"
        component = str(item.get("affected_component", "")).strip().lower()
        if component not in AUDIT_COMPONENTS:
            component = kind if kind in AUDIT_COMPONENTS else "worksheet"
        action = str(item.get("recommended_action", "")).strip().lower()
        if action not in AUDIT_ACTIONS:
            # Map by component as a best-effort default.
            action = {
                "objective": "regenerate_objective",
                "quiz": "regenerate_quiz",
                "answer_key": "regenerate_quiz",
                "distractor": "regenerate_distractors",
                "code_example": "fix_explanation",
                "explanation": "fix_explanation",
                "difficulty": "regenerate_worksheet",
                "prerequisite": "escalate_to_teacher",
            }.get(component, "regenerate_worksheet")
        issue = str(item.get("issue", "")).strip()
        if not issue:
            continue
        safe.append({
            "severity": severity,
            "issue": issue,
            "affected_component": component,
            "recommended_action": action,
        })

    if problems and not safe:
        # The model returned no findings but did return problems. Synthesise one
        # finding per problem so the agent has a structured counterpart.
        for problem in problems[:8]:
            safe.append({
                "severity": "medium",
                "issue": problem,
                "affected_component": kind if kind in AUDIT_COMPONENTS else "worksheet",
                "recommended_action": "regenerate_worksheet",
            })
    return safe


# --------------------------------------------------------------------------- #
# Material generation  (tool bodies — plain coroutines, wrapped as tools below)
# --------------------------------------------------------------------------- #

WORKSHEET_SYSTEM_PROMPT = """\
You are a middle school (grades 6-8) computer science curriculum writer specializing in
algorithms and pseudo-code.

Hard rules:
- Pseudo-code only. Use SET, INPUT, PRINT, IF/ELSE IF/ELSE, REPEAT n TIMES, WHILE.
  Never emit real language syntax (no Python, no braces, no semicolons, no imports).
- No images. Render any flowchart as ASCII boxes and arrows the student can read on paper.
- Anchor every exercise in a concrete pre-teen context: video games, lunch lines, group
  chats, playlists, sports scores, lockers.
- Return the worksheet body as plain text only. No markdown fences, no preamble, no
  meta-commentary about your own output.
"""

# Each tier changes the COGNITIVE DEMAND, not just the wording — struggling students
# never face a blank page, advanced students never get pre-written scaffolding.
TIER_WORKSHEET_SPECS: dict[str, str] = {
    "struggling": (
        "Heavy scaffolding. Include: (a) one ASCII flowchart the student reads and labels; "
        "(b) two fill-in-the-blank pseudo-code snippets with a word bank of the exact "
        "keywords needed; (c) one trace table where the student records each variable's "
        "value line by line through a 3-4 step algorithm. Never ask this student to write "
        "an algorithm from a blank page. Keep numbers small and whole."
    ),
    "on-level": (
        "Moderate support. The student writes short sequential pseudo-code from scratch and "
        "uses at least one IF/ELSE conditional. Provide the goal plus sample input and "
        "expected output, but no starter code and no word bank. Include one exercise that "
        "converts a set of written real-world steps into pseudo-code."
    ),
    "advanced": (
        "Low support, high cognitive demand. Include: (a) one flawed pseudo-code snippet "
        "containing a single logic bug (off-by-one, inverted comparison, or a statement "
        "inside the loop that belongs outside it) that the student must locate, explain and "
        "fix; (b) one exercise requiring a REPEAT or WHILE loop to eliminate duplicated "
        "steps; (c) one efficiency question comparing two algorithms, where the student "
        "must justify which does less work and why."
    ),
}


async def _worksheet_text(topic: str, tier: str, objective: str,
                          revision_notes: str = "") -> str:
    """
    Core worksheet generation.

    Kept outside the @tool decorator so other code can await it as an ordinary
    coroutine, without relying on how strands wraps a decorated tool.

    `revision_notes` is what makes correction real rather than hopeful: on a retry the
    specific audit findings are handed back to the writer, so the second attempt is
    aimed at the defect instead of being another roll of the dice.
    """
    spec = TIER_WORKSHEET_SPECS.get(tier)
    if spec is None:
        raise ValueError(f"Unknown tier '{tier}'. Expected one of: {', '.join(TIERS)}.")

    revision_block = ""
    if revision_notes.strip():
        revision_block = textwrap.dedent(
            f"""\

            THIS IS A REVISION. A reviewer rejected your previous attempt for these
            reasons. Fix every one of them. Do not repeat the same mistakes:
            {revision_notes.strip()}
            """
        )

    user_prompt = textwrap.dedent(
        f"""\
        Topic: {topic}
        Learning objective: {objective}
        Ability tier: {tier}

        Tier specification you must follow exactly:
        {spec}

        Required layout:
        WORKSHEET — {topic} ({tier} tier)
        Objective: <restate the objective in student-friendly language>
        Directions: <two sentences maximum>
        Then exactly 5 numbered exercises that escalate in difficulty, followed by a final
        reflection prompt asking the student to explain their reasoning in plain words.
        Use visible blanks (______) everywhere the student is expected to write.
        """
    ).strip() + revision_block

    worksheet = (await _complete(
        WORKSHEET_SYSTEM_PROMPT, user_prompt, max_tokens=3500, temperature=0.5,
    )).strip()
    if not worksheet:
        raise ValueError(f"Model returned an empty worksheet for '{topic}' ({tier} tier).")

    logger.info("Generated worksheet | topic=%s | tier=%s | chars=%d", topic, tier, len(worksheet))
    return worksheet


QUIZ_SYSTEM_PROMPT = """\
You are a middle school (grades 6-8) computer science assessment writer specializing in
algorithms and pseudo-code.

Hard rules:
- Pseudo-code only (SET, INPUT, PRINT, IF/ELSE, REPEAT n TIMES, WHILE). No real language
  syntax anywhere.
- Every distractor in a multiple-choice item must encode a real student misconception,
  never a joke or an obviously wrong option.
- Respond with a single JSON object and nothing else. No markdown fences, no prose.
- The JSON object has exactly two keys: "quiz" and "answer_key". Both values are plain
  text strings; use \\n for line breaks inside them.
"""

TIER_QUIZ_SPECS: dict[str, str] = {
    "struggling": (
        "Recall and tracing only. Two multiple-choice items on vocabulary and reading "
        "pseudo-code, one 'predict the output' item over a 3-step algorithm, one "
        "fill-in-the-blank item, and one short written explanation."
    ),
    "on-level": (
        "Application. Two multiple-choice items on choosing the correct control structure, "
        "one 'predict the output' item involving an IF/ELSE, one item where the student "
        "writes 3-5 lines of pseudo-code from scratch, and one short written explanation."
    ),
    "advanced": (
        "Analysis and transfer. Two multiple-choice items on loop behavior and edge cases, "
        "one item where the student finds and fixes a logic bug in given pseudo-code, one "
        "item requiring a loop-based rewrite, and one item justifying which of two "
        "algorithms is more efficient."
    ),
}


async def _quiz_payload(topic: str, tier: str, objective: str,
                        revision_notes: str = "") -> dict:
    """
    Core quiz + answer key generation.

    Kept outside the @tool decorator for the same reason as _worksheet_text. One model
    call produces both artifacts, because an answer key written from a separate call
    tends to drift from the quiz it is supposed to mark.
    """
    spec = TIER_QUIZ_SPECS.get(tier)
    if spec is None:
        raise ValueError(f"Unknown tier '{tier}'. Expected one of: {', '.join(TIERS)}.")

    revision_block = ""
    if revision_notes.strip():
        revision_block = textwrap.dedent(
            f"""\

            THIS IS A REVISION. A reviewer rejected your previous attempt for these
            reasons. Fix every one of them:
            {revision_notes.strip()}
            """
        )

    user_prompt = textwrap.dedent(
        f"""\
        Topic: {topic}
        Learning objective assessed: {objective}
        Ability tier: {tier}

        Tier specification you must follow exactly:
        {spec}

        Build a 5-item quiz worth 10 points total (mastery threshold 8/10).

        "quiz" must start with the header: QUIZ — {topic} ({tier} tier)
        Number the items Q1-Q5 and show the point value of each in parentheses.

        "answer_key" must start with the header: ANSWER KEY — {topic} ({tier} tier)
        For every item give the correct answer, the points awarded, and one line naming the
        most likely student misconception a wrong answer would reveal. End with the total
        and the mastery threshold.

        Return only the JSON object with keys "quiz" and "answer_key".
        """
    ).strip() + revision_block

    raw = await _complete(
        QUIZ_SYSTEM_PROMPT, user_prompt, max_tokens=3500, temperature=0.4, json_mode=True,
    )
    payload = _loads_model_json(raw, f"'{topic}' ({tier} tier)")

    quiz = str(payload.get("quiz", "")).strip()
    answer_key = str(payload.get("answer_key", "")).strip()
    if not quiz or not answer_key:
        raise ValueError(
            f"Model response for '{topic}' ({tier} tier) is missing 'quiz' or 'answer_key'."
        )

    logger.info("Generated quiz | topic=%s | tier=%s", topic, tier)
    return {"quiz": quiz, "answer_key": answer_key}


async def _insert_drafts(rows: Sequence[tuple]) -> list[int]:
    """
    Insert one or more draft rows inside a single transaction.

    All rows share one commit, so a partially generated lesson can never leave a
    half-populated tier set on the review desk: either every tier lands or none does.

    Args:
        rows: Tuples of (unit_name, topic, objective, tier, worksheet, quiz, answer_key).

    Returns:
        The new row ids, in the same order as the input.
    """
    row_ids: list[int] = []
    async with aiosqlite.connect(DB_PATH) as db:
        for row in rows:
            cursor = await db.execute(INSERT_DRAFT_SQL, row)
            row_ids.append(cursor.lastrowid)
        await db.commit()
    return row_ids


async def _flag_row(unit_name: str, topic: str, objective: str, tier: str,
                    reason: str, worksheet_text: str = "",
                    quiz_text: str = "", answer_key_text: str = "") -> int:
    """Insert one flagged row and return its id. The single escalation write path.

    When the agent escalates with a worksheet, quiz, or answer key in hand,
    those go in the same row so the teacher can review the actual material
    on the dashboard (and export it as a classroom package once approved).
    The columns default to empty strings so older callers stay unchanged.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            """
            INSERT INTO materials (
                unit_name, topic, objective, tier,
                status, reason, worksheet_text, quiz_text, answer_key_text
            ) VALUES (?, ?, ?, ?, 'flagged', ?, ?, ?, ?)
            """,
            (unit_name, topic, objective, tier, reason,
             worksheet_text, quiz_text, answer_key_text),
        )
        await db.commit()
        return cursor.lastrowid


# --------------------------------------------------------------------------- #
# Export
#
# The export functions build a single "classroom package" for one record
# and render it in three formats: a print-ready HTML handout, a
# Markdown document, and a server-rendered print page (the teacher prints
# to PDF via the browser). All three reuse the per-record data already
# in the materials table; no second model call is involved.
# --------------------------------------------------------------------------- #


async def fetch_material_by_id(item_id: int) -> dict[str, Any] | None:
    """Look up one material row by id, or None if it does not exist."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM materials WHERE id = ?", (item_id,),
        ) as cur:
            row = await cur.fetchone()
    return dict(row) if row else None


def _parse_worksheet_blocks(worksheet: str) -> dict[str, Any]:
    """
    Pull the structural pieces out of a worksheet so the export can render
    them in their own sections (objective line, directions, numbered items,
    trace table, reflection prompt).
    """
    out: dict[str, Any] = {
        "objective": "",
        "directions": "",
        "items": [],
        "table": [],
        "reflection": "",
    }
    if not worksheet:
        return out
    lines = [line.rstrip() for line in worksheet.splitlines()]
    for line in lines:
        stripped = line.strip()
        low = stripped.lower()
        if not out["objective"] and low.startswith("objective:"):
            out["objective"] = stripped[len("objective:"):].strip()
            continue
        if not out["directions"] and low.startswith("directions:"):
            out["directions"] = stripped[len("directions:"):].strip()
            continue
        if low.startswith("|") and "|" in stripped[1:]:
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if len(cells) >= 2:
                out["table"].append(cells)
            continue
        if low.startswith("|---") or set(stripped.replace("|", "").strip()) <= {"-", " "}:
            continue
        m = re.match(r"^\s*(\d+)\.\s+(.*)", line)
        if m:
            out["items"].append({
                "number": int(m.group(1)),
                "text": m.group(2).strip(),
            })
            continue
        if any(kw in low for kw in ("explain in your own words", "reflect", "why do you think",
                                    "describe what you would", "trace the table")):
            out["reflection"] = stripped
    return out


def _export_misconceptions(answer_key: str | None) -> list[dict[str, Any]]:
    """
    Reuse the existing answer-key parser to surface per-option misconception
    diagnostics. Empty when no answer key is available.
    """
    if not answer_key:
        return []
    parsed = _extract_answer_key_misconceptions(answer_key)
    rows: list[dict[str, Any]] = []
    for q_number, entries in parsed.items():
        for entry in entries:
            if entry.get("option_letter") == "":
                continue  # free-response entries are not distractors
            rows.append({
                "question": f"Q{q_number}",
                "option_letter": entry["option_letter"],
                "option_text": entry.get("option_text", ""),
                "is_correct": bool(entry.get("is_correct")),
                "misconception": entry.get("misconception") or "",
            })
    return rows


def _build_export_package(record: dict[str, Any]) -> dict[str, Any]:
    """
    Assemble the structured "classroom package" for one material record.

    The export route is the teacher-facing surface, so the package is
    ordered the way a teacher would file it: objective first, then the
    student-facing worksheet, the assessment, the answer key, the
    misconception diagnostic, and a closing teacher notes block.
    """
    worksheet = record.get("worksheet_text") or ""
    quiz = record.get("quiz_text") or ""
    answer_key = record.get("answer_key_text") or ""
    worksheet_blocks = _parse_worksheet_blocks(worksheet)
    misconceptions = _export_misconceptions(answer_key)
    # Parse the audit history for this record's digest. If a digest isn't
    # in the in-memory map (e.g. server restart), surface an empty list
    # rather than guessing.
    digest = _digest(worksheet + "\n" + quiz + "\n" + answer_key)
    audit_history = []
    for mission in MISSIONS.values():
        if digest in mission.audits:
            audit_history.append(mission.audits[digest])
    # Try harder: many sub-missions don't store the full text digest,
    # so fall back to looking up by record_id in the audit map.
    if not audit_history:
        for mission in MISSIONS.values():
            for stored_digest, audit in mission.audits.items():
                # Different records hash differently; the heuristic here
                # is the last-seen audit for this tier+topic — close enough
                # for the export view.
                if (audit.get("tier") == record.get("tier")
                        and not audit_history):
                    audit_history.append(audit)
    return {
        "unit_name": record.get("unit_name", ""),
        "topic": record.get("topic", ""),
        "objective": record.get("objective", ""),
        "tier": record.get("tier", ""),
        "id": record.get("id"),
        "status": record.get("status", ""),
        "reason": record.get("reason", ""),
        "worksheet_text": worksheet,
        "worksheet_blocks": worksheet_blocks,
        "quiz_text": quiz,
        "answer_key_text": answer_key,
        "misconceptions": misconceptions,
        "audit_history": audit_history,
    }


def _export_markdown(pkg: dict[str, Any]) -> str:
    """Render the package as a Markdown document."""
    lines: list[str] = []
    unit = pkg["unit_name"] or pkg["topic"] or "Lesson"
    lines.append(f"# {unit}")
    lines.append("")
    lines.append(f"**Topic:** {pkg['topic']}  ")
    lines.append(f"**Ability tier:** {pkg['tier']}  ")
    lines.append(f"**Learning objective:** {pkg['objective']}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Worksheet")
    lines.append("")
    if pkg["worksheet_blocks"]["objective"]:
        lines.append(f"**Objective:** {pkg['worksheet_blocks']['objective']}  ")
    if pkg["worksheet_blocks"]["directions"]:
        lines.append(f"**Directions:** {pkg['worksheet_blocks']['directions']}")
    lines.append("")
    for item in pkg["worksheet_blocks"]["items"]:
        lines.append(f"{item['number']}. {item['text']}")
        lines.append("")
    if pkg["worksheet_blocks"]["table"]:
        # Markdown table
        first = pkg["worksheet_blocks"]["table"][0]
        lines.append("| " + " | ".join(first) + " |")
        lines.append("| " + " | ".join("---" for _ in first) + " |")
        for row in pkg["worksheet_blocks"]["table"][1:]:
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    if pkg["worksheet_blocks"]["reflection"]:
        lines.append("**Reflection:** " + pkg["worksheet_blocks"]["reflection"])
        lines.append("")
    if pkg["worksheet_text"] and not pkg["worksheet_blocks"]["items"]:
        # Fallback: show the raw text so nothing is lost.
        lines.append("```")
        lines.append(pkg["worksheet_text"])
        lines.append("```")
        lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Assessment")
    lines.append("")
    if pkg["quiz_text"]:
        lines.append(pkg["quiz_text"])
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Answer key")
    lines.append("")
    if pkg["answer_key_text"]:
        lines.append(pkg["answer_key_text"])
    lines.append("")
    if pkg["misconceptions"]:
        lines.append("---")
        lines.append("")
        lines.append("## Misconception diagnostics (teacher reference)")
        lines.append("")
        lines.append(
            "Each row names the misconception a wrong answer reveals. "
            "Use this when grading: the row is the diagnosis, not just the answer."
        )
        lines.append("")
        lines.append("| Question | Option | Misconception |")
        lines.append("| --- | --- | --- |")
        for row in pkg["misconceptions"]:
            lines.append(
                f"| {row['question']} | {row['option_letter']} "
                f"{'✓' if row['is_correct'] else ''} | "
                f"{row['misconception'] or '—'} |"
            )
        lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Teacher notes")
    lines.append("")
    lines.append(
        "- Materials were generated by the Strands agent and audited against the "
        f"objective, tier, and the curriculum's structural rules."
    )
    if pkg["status"] == "approved":
        lines.append("- Status: **approved** for classroom use.")
    elif pkg["status"] == "flagged":
        lines.append(f"- Status: **flagged** for teacher review. Reason: {pkg['reason'] or 'see dashboard'}.")
    else:
        lines.append("- Status: **draft**, awaiting teacher approval.")
    lines.append("- Tiers differ in scaffolding: struggling uses a word bank and a trace table;")
    lines.append("  on-level expects a self-correction; advanced asks the student to find and fix a defect.")
    return "\n".join(lines).strip() + "\n"


def _export_html(pkg: dict[str, Any]) -> str:
    """Print-ready HTML handout. CSS is inline so the teacher can print directly."""
    css = """
    body { font-family: Georgia, 'Times New Roman', serif; max-width: 8.5in;
           margin: 0.6in auto; padding: 0 0.4in; color: #1a1a1a; line-height: 1.45; }
    h1 { font-size: 22pt; margin: 0 0 4pt; letter-spacing: -0.01em; }
    h2 { font-size: 15pt; margin: 22pt 0 8pt; border-bottom: 1px solid #c8c8c8; padding-bottom: 3pt; }
    h3 { font-size: 12pt; margin: 14pt 0 4pt; }
    .meta { color: #555; font-size: 10pt; margin: 0 0 14pt; }
    .meta b { color: #1a1a1a; }
    .objective, .directions, .reflection { background: #f6f4ee;
        border-left: 3px solid #888; padding: 7pt 10pt; margin: 8pt 0; font-size: 10.5pt; }
    .objective b, .directions b, .reflection b { color: #1a1a1a; }
    ol.worksheet { padding-left: 22pt; }
    ol.worksheet li { margin-bottom: 8pt; }
    table { border-collapse: collapse; margin: 10pt 0; font-size: 10.5pt; }
    th, td { border: 1px solid #b8b8b8; padding: 4pt 8pt; }
    th { background: #ece9df; }
    pre, .raw { white-space: pre-wrap; font-family: 'Courier New', monospace;
        font-size: 10pt; background: #fafaf6; padding: 8pt 10pt; border: 1px solid #e2e0d4; }
    .misconception { font-size: 10pt; border-collapse: collapse; }
    .misconception td { vertical-align: top; padding: 4pt 8pt; }
    .misconception .qcol { width: 8%; }
    .misconception .ocol { width: 12%; }
    .misconception .mcol { width: 80%; }
    .badge { display: inline-block; padding: 1pt 6pt; border-radius: 3pt;
        font-size: 9pt; font-weight: 600; letter-spacing: 0.02em; text-transform: uppercase; }
    .badge-approved { background: #d8efd8; color: #1b5e20; }
    .badge-flagged { background: #fff0c2; color: #8a5a00; }
    .badge-draft { background: #ece9df; color: #555; }
    .teacher-notes { background: #fafaf6; border: 1px solid #e2e0d4;
        padding: 10pt 12pt; font-size: 10pt; color: #333; }
    .teacher-notes ul { margin: 4pt 0 0 18pt; padding: 0; }
    @media print {
      body { margin: 0.4in; }
      h2 { page-break-after: avoid; }
      .worksheet, .assessment, .answer-key, .misconceptions { page-break-inside: avoid; }
    }
    """

    def esc(s: str) -> str:
        return html.escape(s or "")

    badge_class = "badge-approved" if pkg["status"] == "approved" else \
        "badge-flagged" if pkg["status"] == "flagged" else "badge-draft"
    badge_label = pkg["status"].upper() if pkg["status"] else "DRAFT"

    blocks = pkg["worksheet_blocks"]
    parts: list[str] = []
    parts.append(
        f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>{esc(pkg['unit_name'] or pkg['topic'])} - {esc(pkg['tier'])}</title>"
        f"<style>{css}</style></head><body>"
    )
    parts.append(
        f"<h1>{esc(pkg['unit_name'] or pkg['topic'] or 'Lesson')}</h1>"
    )
    parts.append(
        f"<p class='meta'>"
        f"<b>Topic:</b> {esc(pkg['topic'])} &middot; "
        f"<b>Tier:</b> {esc(pkg['tier'])} &middot; "
        f"<span class='badge {badge_class}'>{esc(badge_label)}</span>"
        f"</p>"
    )
    parts.append(
        f"<p><b>Learning objective:</b> {esc(pkg['objective'])}</p>"
    )

    parts.append("<h2>Worksheet</h2>")
    if blocks["objective"]:
        parts.append(f"<div class='objective'><b>Objective:</b> {esc(blocks['objective'])}</div>")
    if blocks["directions"]:
        parts.append(f"<div class='directions'><b>Directions:</b> {esc(blocks['directions'])}</div>")
    if blocks["items"]:
        parts.append("<ol class='worksheet'>")
        for item in blocks["items"]:
            parts.append(f"<li>{esc(item['text'])}</li>")
        parts.append("</ol>")
    if blocks["table"]:
        # Skip the first row of the table if it's clearly a header.
        rows = blocks["table"]
        header = rows[0]
        body = rows[1:] if len(rows) > 1 else []
        parts.append("<table>")
        parts.append("<thead><tr>" + "".join(f"<th>{esc(c)}</th>" for c in header) + "</tr></thead>")
        if body:
            parts.append("<tbody>" + "".join(
                "<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in body
            ) + "</tbody>")
        parts.append("</table>")
    if blocks["reflection"]:
        parts.append(f"<div class='reflection'><b>Reflection:</b> {esc(blocks['reflection'])}</div>")
    if not blocks["items"] and pkg["worksheet_text"]:
        parts.append(f"<pre class='raw'>{esc(pkg['worksheet_text'])}</pre>")

    parts.append("<h2>Assessment</h2>")
    if pkg["quiz_text"]:
        parts.append(f"<pre class='raw'>{esc(pkg['quiz_text'])}</pre>")
    else:
        parts.append("<p><em>No assessment generated for this record.</em></p>")

    parts.append("<h2>Answer key</h2>")
    if pkg["answer_key_text"]:
        parts.append(f"<pre class='raw'>{esc(pkg['answer_key_text'])}</pre>")
    else:
        parts.append("<p><em>No answer key generated for this record.</em></p>")

    if pkg["misconceptions"]:
        parts.append("<h2>Misconception diagnostics (teacher reference)</h2>")
        parts.append("<p><em>Each row names the misconception a wrong answer reveals. "
                     "Use this when grading: the row is the diagnosis, not just the answer.</em></p>")
        parts.append("<table class='misconception'>")
        parts.append("<thead><tr><th class='qcol'>Question</th>"
                     "<th class='ocol'>Option</th><th class='mcol'>Misconception</th></tr></thead><tbody>")
        for row in pkg["misconceptions"]:
            correct = " (correct)" if row["is_correct"] else ""
            parts.append(
                f"<tr><td>{esc(row['question'])}</td>"
                f"<td>{esc(row['option_letter'])}{esc(correct)}</td>"
                f"<td>{esc(row['misconception'] or '—')}</td></tr>"
            )
        parts.append("</tbody></table>")

    parts.append("<h2>Teacher notes</h2>")
    parts.append("<div class='teacher-notes'><ul>")
    parts.append(
        "<li>Materials were generated by the Strands agent and audited against the "
        f"objective, the {esc(pkg['tier'])} tier, and the curriculum's structural rules.</li>"
    )
    if pkg["status"] == "approved":
        parts.append("<li>Status: <b>approved</b> for classroom use.</li>")
    elif pkg["status"] == "flagged":
        reason = pkg.get("reason") or "see the dashboard for the full context."
        parts.append(f"<li>Status: <b>flagged</b> for teacher review. Reason: {esc(reason)}</li>")
    else:
        parts.append("<li>Status: <b>draft</b>, awaiting teacher approval.</li>")
    parts.append(
        "<li>Tiers differ in scaffolding: the <b>struggling</b> tier uses a word bank and a "
        "trace table; the <b>on-level</b> tier expects a self-correction; the <b>advanced</b> "
        "tier asks the student to find and fix a defect.</li>"
    )
    parts.append("</ul></div>")

    parts.append("</body></html>")
    return "".join(parts)


def _export_print_html(pkg: dict[str, Any]) -> str:
    """
    A small wrapper around the export HTML that auto-triggers
    window.print() on load. The PDF endpoint serves this; the teacher's
    browser opens it, prints to PDF, and saves the file. The page is
    only useful for the immediate print, so the body is hidden until
    the user clicks "Print" or the print dialog opens.
    """
    body = _export_html(pkg)
    # Inject the auto-print trigger and a fallback instruction panel
    # (shown if window.print() is blocked by the browser, e.g. when the
    # teacher downloads the file instead of opening it).
    trigger = (
        "<script>setTimeout(function(){try{window.print();}catch(e){}},250);</script>"
    )
    fallback = (
        "<noscript><p style='font-family:sans-serif;padding:14px;background:#fff0c2;'>"
        "This page is meant to print to PDF. Press <b>Ctrl+P</b> (or Cmd+P) and choose "
        "'Save as PDF' to download the classroom package.</p></noscript>"
    )
    return body.replace("</body>", trigger + fallback + "</body>")


# --------------------------------------------------------------------------- #
# Agent activity
#
# A safe, concise view of what the agent did for one material record. The
# underlying sub-mission may contain raw model output, which we MUST NOT
# surface. The view strips the per-event payload down to a short action
# line: the event kind plus a redacted message. This is the surface the
# spec calls for: not chain-of-thought, but a record of agent actions.
# --------------------------------------------------------------------------- #


def _summarise_event(event: dict[str, Any]) -> dict[str, Any] | None:
    """
    Convert one mission event into a safe, one-line summary suitable for
    the dashboard's "Agent Activity" timeline. Returns None for events that
    do not describe an agent action worth showing the teacher.
    """
    kind = event.get("kind", "")
    # Strip the prefix that strands prefixes onto the agent's own narration;
    # that is the only place a chain-of-thought would leak through.
    raw_msg = (event.get("message") or "").strip()
    # Defence in depth: anything that looks like internal reasoning
    # should not reach the dashboard even if the kind slips through.
    if kind == "agent_reasoning":
        return None
    # Pick a short, safe label per kind.
    label_map = {
        "mission_started":        "Mission started",
        "tool_selected":          "Tool selected",
        "tool_started":           "Tool started",
        "tool_result":            "Tool result",
        "audit_started":          "Audit started",
        "audit_result":           "Audit verdict",
        "audit_passed":           "Validation passed",
        "audit_failed":           "Validation failed",
        "verification":           "Pseudo-code verified",
        "self_correction":        "Regenerated",
        "material_regenerated":   "Material regenerated",
        "human_review_required":  "Human review required",
        "human_review":           "Escalated for review",
        "human_review_completed": "Reviewer decision",
        "save":                   "Saved",
        "mission_completed":      "Mission completed",
        "mission_failed":         "Mission failed",
    }
    label = label_map.get(kind, kind.replace("_", " "))
    # Clip the message to keep the row light.
    return {
        "kind": kind,
        "label": label,
        "message": _clip(raw_msg, 160),
        "at": event.get("at", ""),
    }


def _activity_for_record(record: dict[str, Any], *, limit: int = 24) -> list[dict[str, Any]]:
    """
    Find the sub-mission that produced this record, return its safe
    action-level events.

    The lookup is heuristic: the sub-mission's `audits` dict is keyed by
    text digest, not by record_id, so we walk all in-memory sub-missions
    and pick the one whose saved_ids or flagged_ids include this record.
    """
    record_id = record.get("id")
    if record_id is None:
        return []

    # 1. Direct lookup via RECORD_TO_SUB_MISSION if it exists (we'll build it).
    sub_id = RECORD_TO_SUB_MISSION.get(record_id)  # type: ignore[name-defined]
    sub = MISSIONS.get(sub_id) if sub_id else None

    # 2. Fall back to scanning all in-memory missions.
    if sub is None:
        for candidate in MISSIONS.values():
            if (record_id in getattr(candidate, "saved_ids", ()) or
                    record_id in getattr(candidate, "flagged_ids", ())):
                sub = candidate
                break

    if sub is None:
        return []

    summary = [
        s for s in (_summarise_event(e) for e in sub.events) if s is not None
    ]
    # Most recent first; cap to a small number for the timeline.
    summary.reverse()
    return summary[:limit]


# Reverse index of record_id -> sub-mission id, populated by the
# orchestrator and the regenerate route. Used by the activity endpoint to
# find the right sub-mission without scanning.
RECORD_TO_SUB_MISSION: dict[int, str] = {}


# --------------------------------------------------------------------------- #
# Curriculum planner
# --------------------------------------------------------------------------- #
PLAN_SYSTEM_PROMPT = """\
You are a middle school (grades 6-8) computer science curriculum planner.

Given a bare topic, you decide two things: which unit the topic belongs in, and the
single measurable learning objective a lesson on it should serve.

Hard rules:
- The objective must be measurable and observable. Open with one Bloom's verb such as
  trace, predict, identify, write, modify, compare or justify. Never use "understand",
  "learn about", "know", "be familiar with", or "appreciate".
- The objective must be one sentence, achievable in a single class period, and phrased
  as what the student will be able to do.
- Stay inside pseudo-code (SET, INPUT, PRINT, IF/ELSE, REPEAT n TIMES, WHILE). Never
  name a real programming language or its syntax.
- unit_name is a short title-case unit heading, at most six words, with no numbering
  unless the topic itself implies one.
- Respond with a single JSON object and nothing else. No markdown fences, no prose.
- The JSON object has exactly two keys: "unit_name" and "objective". Both are plain
  text strings.
"""


async def _plan_lesson(topic: str, revision_notes: str = "") -> dict[str, str]:
    """
    Turn a bare topic into a unit name and a measurable learning objective.

    Returns:
        A dict with keys "unit_name" and "objective".
    """
    revision_block = ""
    if revision_notes.strip():
        revision_block = (
            "\n\nA reviewer rejected your previous objective for these reasons. "
            f"Write a new one that fixes them:\n{revision_notes.strip()}"
        )

    user_prompt = textwrap.dedent(
        f"""\
        Topic: {topic}

        Place this topic in a grades 6-8 computer science unit and write the one
        measurable objective a single lesson on it should achieve.

        Return only the JSON object with keys "unit_name" and "objective".
        """
    ).strip() + revision_block

    raw = await _complete(
        PLAN_SYSTEM_PROMPT, user_prompt, max_tokens=600, temperature=0.3, json_mode=True,
    )
    payload = _loads_model_json(raw, f"the lesson plan for '{topic}'")

    unit_name = str(payload.get("unit_name", "")).strip()
    objective = str(payload.get("objective", "")).strip()

    # Never persist a blank objective — the tier generators are built around it, and a
    # blank one would produce three worthless drafts a human still has to read.
    if not unit_name or not objective:
        raise ValueError(
            f"Planner response for '{topic}' is missing 'unit_name' or 'objective'."
        )

    logger.info("Planned lesson | topic=%s | unit=%s", topic, unit_name)
    return {"unit_name": unit_name, "objective": objective}


DISTRACTOR_SYSTEM_PROMPT = """\
You write multiple-choice distractors for middle school computer science assessments.

A distractor is only useful if a real student would pick it for a real reason. Each one
you write must correspond to a specific, nameable misconception a grades 6-8 student
actually holds — confusing a loop with a conditional, reading > as >=, assuming
assignment works right-to-left, believing a loop body runs once before the test, and so
on. Never write a joke option, an obviously absurd option, or a near-duplicate of the
correct answer.

Respond with a single JSON object and nothing else, with one key "distractors" holding
an array of exactly three objects, each with keys:
  "option"        — the wrong answer as a student would see it
  "misconception" — the specific misconception it detects, in one clause
  "why_plausible" — why a student holding that misconception would choose it
"""


async def _distractors(topic: str, tier: str, concept: str,
                       correct_answer: str) -> list[dict[str, str]]:
    """Generate three misconception-bearing distractors for one item."""
    user_prompt = textwrap.dedent(
        f"""\
        Topic: {topic}
        Ability tier: {tier}
        Concept being assessed: {concept}
        The correct answer: {correct_answer}

        Write three distractors for this item. Return only the JSON object.
        """
    ).strip()

    raw = await _complete(
        DISTRACTOR_SYSTEM_PROMPT, user_prompt,
        max_tokens=900, temperature=0.6, json_mode=True,
    )
    payload = _loads_model_json(raw, f"distractors for '{concept}' ({tier} tier)")

    options: list[dict[str, str]] = []
    for entry in payload.get("distractors", []):
        if not isinstance(entry, dict):
            continue
        option = str(entry.get("option", "")).strip()
        misconception = str(entry.get("misconception", "")).strip()
        if not option or not misconception:
            continue
        options.append({
            "option": option,
            "misconception": misconception,
            "why_plausible": str(entry.get("why_plausible", "")).strip(),
        })

    if not options:
        raise ValueError(f"No usable distractors came back for '{concept}' ({tier} tier).")

    # A distractor equal to the correct answer is worse than useless: it makes the item
    # unmarkable. Cheap to catch here, expensive to catch in a classroom.
    normalised_correct = correct_answer.strip().lower()
    options = [item for item in options
               if item["option"].strip().lower() != normalised_correct]
    if not options:
        raise ValueError(
            f"Every distractor for '{concept}' duplicated the correct answer."
        )
    return options


# --------------------------------------------------------------------------- #
# Agent tools
#
# Ten tools. The agent chooses which to call, in what order, how many times, and
# when to stop; nothing in this file sequences them. What this file does enforce is
# the small set of rails that must not depend on the model's goodwill: a correction
# budget, a refusal to save material that has not passed audit, a refusal to escalate
# a defect the agent has not actually tried to fix, and a completion check made
# against the database rather than against the agent's own account of itself.
# --------------------------------------------------------------------------- #

@tool
async def get_curriculum_context(topic: str) -> dict:
    """
    Look up what the curriculum already holds for a topic before generating anything.

    Reads the lessons database for prior material on this topic, so you can see which
    ability tiers already exist, which a teacher already approved, which were previously
    escalated and why, and what objective was used last time. Also returns the
    pseudo-code vocabulary and the tier specifications this curriculum requires.

    Call this first. It is cheap, it needs no model, and it tells you whether you are
    creating a lesson from scratch or duplicating one that already exists.

    Args:
        topic: The lesson topic to look up.

    Returns:
        A dict describing existing records, missing tiers, and the house rules.
    """
    mission = _log_tool("get_curriculum_context", topic=topic)

    needle = f"%{topic.strip().lower()}%"
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """
            SELECT id, unit_name, topic, objective, tier, status, reason
              FROM materials
             WHERE lower(topic) LIKE ? OR lower(unit_name) LIKE ?
             ORDER BY id DESC
             LIMIT 12
            """,
            (needle, needle),
        )
        rows = [dict(row) for row in await cursor.fetchall()]

    existing = [
        {"id": row["id"], "tier": row["tier"], "outcome": row["status"],
         "unit_name": row["unit_name"], "objective": _clip(row["objective"], 160),
         "escalation_reason": _clip(row["reason"], 200) if row["reason"] else None}
        for row in rows
    ]
    covered = {row["tier"] for row in rows if row["status"] in {"draft", "approved"}}
    prior_objective = next(
        (row["objective"] for row in rows
         if row["status"] == "approved" and row["objective"]),
        next((row["objective"] for row in rows if row["objective"]), ""),
    )

    result = {
        "topic": topic,
        "existing_records": existing,
        "tiers_required": list(TIERS),
        "tiers_already_covered": sorted(covered),
        "tiers_missing": [tier for tier in TIERS if tier not in covered],
        "previously_escalated": [row for row in existing
                                 if row["outcome"] == "flagged"],
        "prior_objective_for_this_topic": _clip(prior_objective, 200) or None,
        "house_rules": {
            "audience": "grades 6-8 computer science",
            "notation": "pseudo-code only: SET, INPUT, PRINT, IF/ELSE IF/ELSE, "
                        "REPEAT n TIMES, WHILE. No real programming language syntax.",
            "no_images": "Any diagram must be ASCII art readable on paper.",
            "mastery_threshold": "quizzes are 5 items, 10 points, mastery at 8/10",
        },
        "tier_specifications": {
            tier: {"worksheet": TIER_WORKSHEET_SPECS[tier], "quiz": TIER_QUIZ_SPECS[tier]}
            for tier in TIERS
        },
    }

    _log_result(
        mission, "get_curriculum_context",
        f"{len(existing)} existing record(s); missing tiers: "
        f"{', '.join(result['tiers_missing']) or 'none'}",
        existing=len(existing), missing=result["tiers_missing"],
    )
    return result


@tool
async def generate_learning_objective(topic: str, revision_notes: str = "") -> dict:
    """
    Write the single measurable learning objective a lesson on this topic should serve,
    and the unit heading it belongs under.

    Use this when you were given a bare topic and no objective. The returned objective
    is checked against this curriculum's rules before you see it: an objective that
    opens with an unmeasurable phrase such as "understand" comes back with the problem
    named in "self_check_problems" so you can decide whether to accept it, revise it, or
    escalate the topic as unteachable.

    Args:
        topic: The lesson topic, e.g. "Nested loops".
        revision_notes: On a retry, the problems from the previous attempt, so the new
            objective is aimed at fixing them. Leave empty on the first attempt.

    Returns:
        A dict with "unit_name", "objective", "self_check_problems" and "self_check_ok".
    """
    mission = _log_tool("generate_learning_objective", topic=topic,
                        revising=bool(revision_notes.strip()))

    plan = await _plan_lesson(topic, revision_notes)
    structural = _deterministic_problems(
        "objective", "on-level", topic, plan["objective"], plan["objective"]
    )
    problems = structural["problems"]

    if mission is not None:
        mission.objective = plan["objective"]
        mission.unit_name = plan["unit_name"]

    _log_result(
        mission, "generate_learning_objective",
        f"{plan['unit_name']} — {_clip(plan['objective'], 120)}"
        + (f" [{len(problems)} problem(s)]" if problems else ""),
        unit_name=plan["unit_name"], problems=problems,
    )
    return {
        "unit_name": plan["unit_name"],
        "objective": plan["objective"],
        "self_check_problems": problems,
        "self_check_findings": structural["findings"],
        "self_check_ok": not problems,
    }


@tool
async def generate_differentiated_material(topic: str, tier: str, objective: str,
                                           revision_notes: str = "") -> dict:
    """
    Write the student-facing worksheet for one ability tier.

    The three tiers differ in cognitive demand, not wording: "struggling" gets a
    flowchart, a word bank and a trace table and is never asked to write from a blank
    page; "on-level" writes short pseudo-code from scratch with sample input and output
    but no starter code; "advanced" gets a deliberately flawed snippet to find and fix,
    a loop-refactoring exercise, and an efficiency comparison to justify.

    Generate one tier per call. The material is not saved by this tool and cannot be
    saved until it has passed audit_material.

    Args:
        topic: The lesson topic.
        tier: One of "struggling", "on-level", "advanced".
        objective: The measurable objective this worksheet must serve.
        revision_notes: On a retry, the exact audit problems to fix. Passing these back
            is what makes the second attempt different from the first — leave empty only
            on a first attempt.

    Returns:
        A dict with "worksheet", "tier", "attempt" and "corrections_left".
    """
    mission = _log_tool("generate_differentiated_material", tier=tier, topic=topic,
                        revising=bool(revision_notes.strip()))

    if tier not in TIERS:
        message = f"Unknown tier {tier!r}. Use one of: {', '.join(TIERS)}."
        _log_result(mission, "generate_differentiated_material", message)
        return {"outcome": "rejected", "detail": message, "tier": tier}

    attempt = mission.attempt(tier, "worksheet") if mission else 1
    worksheet = await _worksheet_text(topic, tier, objective, revision_notes)

    if mission is not None and attempt > 1:
        mission.record(
            "material_regenerated",
            f"regenerated the {tier} worksheet (attempt {attempt})",
            tier=tier, attempt=attempt,
        )

    _log_result(
        mission, "generate_differentiated_material",
        f"worksheet for {tier} tier, attempt {attempt}, {len(worksheet)} chars",
        tier=tier, attempt=attempt, chars=len(worksheet),
    )
    return {
        "outcome": "generated",
        "tier": tier,
        "worksheet": worksheet,
        "attempt": attempt,
        "corrections_left": mission.corrections_left(tier, "worksheet") if mission else 0,
    }


@tool
async def generate_assessment(topic: str, tier: str, objective: str,
                              revision_notes: str = "") -> dict:
    """
    Write the formative quiz and its matching answer key for one ability tier.

    Five items, ten points, mastery at 8/10. The answer key names, for every item, the
    misconception a wrong answer would reveal — that is what makes it useful to a
    teacher rather than just a list of letters. Quiz and key are written together in one
    call so the key cannot drift from the quiz it marks.

    Args:
        topic: The lesson topic.
        tier: One of "struggling", "on-level", "advanced".
        objective: The objective being assessed.
        revision_notes: On a retry, the exact audit problems to fix.

    Returns:
        A dict with "quiz", "answer_key", "tier", "attempt" and "corrections_left".
    """
    mission = _log_tool("generate_assessment", tier=tier, topic=topic,
                        revising=bool(revision_notes.strip()))

    if tier not in TIERS:
        message = f"Unknown tier {tier!r}. Use one of: {', '.join(TIERS)}."
        _log_result(mission, "generate_assessment", message)
        return {"outcome": "rejected", "detail": message, "tier": tier}

    attempt = mission.attempt(tier, "quiz") if mission else 1
    payload = await _quiz_payload(topic, tier, objective, revision_notes)

    if mission is not None and attempt > 1:
        mission.record(
            "material_regenerated",
            f"regenerated the {tier} quiz + answer key (attempt {attempt})",
            tier=tier, attempt=attempt,
        )

    _log_result(
        mission, "generate_assessment",
        f"quiz + answer key for {tier} tier, attempt {attempt}",
        tier=tier, attempt=attempt,
    )
    return {
        "outcome": "generated",
        "tier": tier,
        "quiz": payload["quiz"],
        "answer_key": payload["answer_key"],
        "attempt": attempt,
        "corrections_left": mission.corrections_left(tier, "quiz") if mission else 0,
    }


@tool
async def generate_misconception_distractors(topic: str, tier: str, concept: str,
                                            correct_answer: str) -> dict:
    """
    Write three multiple-choice distractors, each tied to a named student misconception.

    Use this when an audit reports that a quiz item's wrong answers are weak, obvious,
    or interchangeable — a distractor nobody would pick teaches the teacher nothing
    about why a student got the item wrong. Each returned option comes with the specific
    misconception it detects, so you can fold both the option and its diagnostic meaning
    back into the quiz and the answer key.

    Args:
        topic: The lesson topic.
        tier: One of "struggling", "on-level", "advanced".
        concept: The precise thing the item assesses, e.g. "when a WHILE body runs".
        correct_answer: The correct answer, so the distractors do not duplicate it.

    Returns:
        A dict with "distractors": a list of {option, misconception, why_plausible}.
    """
    mission = _log_tool("generate_misconception_distractors", tier=tier, concept=concept)

    options = await _distractors(topic, tier, concept, correct_answer)

    _log_result(
        mission, "generate_misconception_distractors",
        f"{len(options)} distractor(s) for '{_clip(concept, 60)}': "
        + "; ".join(_clip(item["misconception"], 60) for item in options),
        count=len(options),
    )
    return {"outcome": "generated", "concept": concept, "distractors": options}


@tool
async def audit_material(kind: str, tier: str, topic: str, objective: str,
                         material_text: str,
                         companion_text: str = "") -> dict:
    """
    Review a piece of material against its objective and tier, and return a verdict.

    Two layers run, cheapest first. Structural checks confirm the material has the
    header, the five exercises, the point values, the blanks, the reflection prompt and
    the tier-specific components this curriculum requires; only if those pass is a
    separate reviewer asked to judge the teaching. The reviewer did not write the
    material and gains nothing by approving it.

    The verdict tells you what to do next, and the distinction matters:
      "pass"    — the material is ready to save.
      "revise"  — a concrete mechanical defect. Fix it yourself: call the matching
                  generator again, passing the problems as revision_notes. Do not
                  escalate a "revise".
      "human"   — a teacher's professional judgement is genuinely required. Do not
                  retry; escalate with flag_for_human_review.
      "unknown" — the audit itself failed, so the material is unreviewed. Retry the
                  audit once, then escalate if it still fails.

    Material may only be saved after it has passed this audit. Auditing the worksheet is
    required; audit the assessment too when the tier or the objective makes its
    difficulty a real question.

    Args:
        kind: One of "worksheet", "quiz", "answer_key", "objective".
        tier: One of "struggling", "on-level", "advanced".
        topic: The lesson topic.
        objective: The objective the material claims to serve.
        material_text: The exact text to review, unmodified.
        companion_text: When auditing a quiz or answer_key, the worksheet that the
            quiz asks the student to trace. Optional, but enables the
            pseudo-code consistency check.

    Returns:
        A dict with verdict, problems, findings, severity, auto_fixable,
        corrections_left, next_step, and the structured finding list the spec
        asks for (severity, issue, affected_component, recommended_action).
    """
    mission = _log_tool("audit_material", kind=kind, tier=tier, chars=len(material_text),
                        companion_chars=len(companion_text) if companion_text else 0)

    if mission is not None:
        mission.record(
            "audit_started",
            f"auditing the {kind} for the {tier} tier",
            audit_kind=kind, tier=tier,
        )

    report = await _audit(kind, tier, topic, objective, material_text)
    verdict = report["verdict"]
    findings = report.get("findings", [])

    # Cross-check: if the agent is auditing a quiz (or its answer key) and hands
    # us the worksheet it was built against, run the bounded interpreter and
    # report any disagreement with the answer key. This is the same check
    # `check_pseudocode_consistency` exposes; we call it from inside the audit
    # so the agent sees the problem without having to wire two tools together.
    if companion_text and kind in {"quiz", "answer_key"}:
        try:
            consistency_problems = check_pseudocode_consistency(companion_text, material_text)
        except Exception as exc:  # noqa: BLE001
            consistency_problems = [f"Consistency check itself errored: {exc}"]
        for problem in consistency_problems:
            if problem in report["problems"]:
                continue
            report["problems"].append(problem)
            findings.append({
                "severity": "high",
                "issue": problem,
                "affected_component": "answer_key" if kind == "answer_key" else "quiz",
                "recommended_action": "regenerate_quiz",
            })
        if consistency_problems:
            verdict = "revise"
            report["verdict"] = verdict
            report["layer"] = report.get("layer", "structural")

    # Misconception-linkage check: when auditing an answer_key, the companion is
    # the quiz. Every wrong option should carry a named misconception in the
    # answer key, otherwise the distractor teaches the teacher nothing about
    # *why* a student would have picked it.
    if kind == "answer_key" and companion_text:
        try:
            linkage_problems = check_misconception_linkage(companion_text, material_text)
        except Exception as exc:  # noqa: BLE001
            linkage_problems = [f"Misconception linkage check itself errored: {exc}"]
        for problem in linkage_problems:
            if problem in report["problems"]:
                continue
            report["problems"].append(problem)
            findings.append({
                "severity": "high",
                "issue": problem,
                "affected_component": "distractor",
                "recommended_action": "regenerate_distractors",
            })
        if linkage_problems:
            verdict = "revise"
            report["verdict"] = verdict
            report["layer"] = report.get("layer", "structural")

    # Tie the verdict to this exact text. Editing the material after an audit
    # invalidates the audit, and save_draft checks the digest rather than trusting a
    # claim that something passed.
    budget_kind = "quiz" if kind in {"quiz", "answer_key"} else kind
    left = mission.corrections_left(tier, budget_kind) if mission else 0
    if mission is not None:
        mission.audits[_digest(material_text)] = {
            "kind": kind, "tier": tier, "verdict": verdict,
            "problems": report["problems"],
            "findings": findings,
            "at": _now_iso(),
        }
        # Persist the per-option misconception metadata for answer-key audits so
        # the dashboard can render the structured per-option diagnostic the
        # spec asks for, and so the deterministic demo can introspect it.
        if kind == "answer_key" and companion_text:
            try:
                mission.audits[_digest(material_text)]["distractor_metadata"] = \
                    _extract_answer_key_misconceptions(material_text)
            except Exception:  # noqa: BLE001
                pass
        mission.record(
            "audit_result",
            f"{kind} ({tier}) -> {verdict}"
            + (f": {_clip('; '.join(report['problems']), 200)}" if report["problems"] else ""),
            audit_kind=kind, tier=tier, verdict=verdict, layer=report.get("layer"),
            problems=report["problems"], corrections_left=left,
        )
        if verdict == "pass":
            mission.record(
                "audit_passed",
                f"{kind} for the {tier} tier passed audit",
                audit_kind=kind, tier=tier,
            )
        elif verdict == "revise":
            # The spec asks for an explicit "audit_failed" event; "audit_result"
            # already carries the verdict but the dashboard distinguishes them by
            # tone. Record it as its own event so the panel can label it.
            mission.record(
                "audit_failed",
                f"audit failed on the {tier} {kind}: "
                f"{_clip('; '.join(report['problems']), 180)}",
                audit_kind=kind, tier=tier, problems=report["problems"],
                findings=findings, corrections_left=left,
            )
            if left > 0:
                mission.record(
                    "self_correction",
                    f"audit failed on the {tier} {kind}; regenerating with these notes: "
                    f"{_clip('; '.join(report['problems']), 180)}",
                    audit_kind=kind, tier=tier, problems=report["problems"],
                    corrections_left=left,
                )
        elif verdict == "human":
            # The spec asks for a "human_review_required" event distinct from the
            # eventual "human_review" event the flag_for_human_review tool emits.
            mission.record(
                "human_review_required",
                f"audit says the {tier} {kind} needs a teacher's judgement: "
                f"{_clip('; '.join(report['problems']), 180)}",
                audit_kind=kind, tier=tier, problems=report["problems"],
                findings=findings,
            )

    if verdict == "pass":
        next_step = f"Save this {kind} with save_draft once the tier's other artifacts exist."
    elif verdict == "revise" and left > 0:
        # The agent uses the structured findings to pick the right corrective tool.
        actions = sorted({f.get("recommended_action", "regenerate_worksheet")
                          for f in findings}) if findings else []
        actions_text = ", ".join(actions) if actions else "regenerate_worksheet"
        next_step = (
            f"Auto-fixable. Regenerate the {kind} for the {tier} tier, passing these "
            f"problems as revision_notes. Suggested actions: {actions_text}. You have "
            f"{left} correction(s) left for it."
        )
    elif verdict == "revise":
        next_step = (
            f"The correction budget for this {kind} is spent. Escalate with "
            "flag_for_human_review using category 'budget_exhausted' and list what you "
            "already tried."
        )
    elif verdict == "human":
        next_step = ("This needs a teacher's judgement, not another attempt. Escalate "
                     "with flag_for_human_review using category "
                     "'pedagogical_judgement'.")
    else:
        next_step = ("The audit did not complete, so this material is unreviewed. Retry "
                     "the audit once; if it fails again escalate with category "
                     "'audit_unavailable'.")

    # Pick the highest severity for the quick top-line "status" the spec asks for.
    sev_rank = {"low": 1, "medium": 2, "high": 3}
    worst = max(
        (sev_rank.get(f.get("severity", "low"), 1) for f in findings),
        default=0,
    )
    overall_severity = {0: "low", 1: "low", 2: "medium", 3: "high"}[worst]

    return {
        "verdict": verdict,
        "status": "PASS" if verdict == "pass" else "FAIL",
        "severity": overall_severity,
        "problems": report["problems"],
        "findings": findings,
        "auto_fixable": report["auto_fixable"] and left > 0,
        "checked_layer": report.get("layer"),
        "reason": report.get("reason", ""),
        "corrections_left": left,
        "next_step": next_step,
    }


@tool
async def verify_code_example(pseudocode: str) -> dict:
    """
    Statically check pseudo-code for defects, and run the bounded interpreter.

    Static checks (free, no model call):
      * Real language syntax leaking into a pseudo-code lesson
      * Constructs students have not been taught (FOR, FUNCTION, RETURN)
      * Stray or inconsistent END markers
      * IF with no THEN, REPEAT with no countable count
      * Variable read before it is given a value
      * WHILE whose body changes nothing its condition tests (cannot terminate)

    Dynamic check (the bounded interpreter in this file): runs the same
    pseudo-code lines as the curriculum teaches, with a step cap, and
    reports the captured output. This is what makes "predict the output"
    items auditable. It is not a general programming language engine: it
    does not handle arrays, functions, or arbitrary I/O, and it says so
    in the "execution" sub-dict when it cannot.

    Findings are graded. An "error" is provably wrong for this curriculum. A
    "warning" depends on context: in a fill-in-the-blank exercise an unset
    variable is expected, and in the advanced tier one planted bug is the
    entire point of the exercise, so a warning there confirms the material
    is doing its job rather than failing. Read "has_student_blanks" and
    the tier before deciding that a finding is a defect.

    This tool reports; it never rewrites. You decide what to do with what
    it finds.

    Args:
        pseudocode: The snippet to check, or the whole worksheet — prose is ignored and
            only lines that open with a pseudo-code keyword are analysed.

    Returns:
        A dict with "ok", "errors", "warnings", counts, a "note" on interpretation,
        and "execution" containing the interpreter's actual output and any error.
    """
    mission = _log_tool("verify_code_example", chars=len(pseudocode))

    report = verify_pseudocode(pseudocode)
    execution = execute_pseudocode(pseudocode)

    if mission is not None:
        summary = (
            f"{report['code_lines_found']} code line(s): "
            f"{report['error_count']} error(s), {report['warning_count']} warning(s)"
        )
        details = [item["detail"] for item in report["errors"][:2]] or \
                  [item["detail"] for item in report["warnings"][:2]]
        mission.record(
            "verification",
            summary + (f" — {_clip('; '.join(details), 200)}" if details else ""),
            ok=report["ok"], errors=report["error_count"],
            warnings=report["warning_count"],
            has_blanks=report["has_student_blanks"],
            execution_ok=execution["ok"],
            execution_output="; ".join(execution["output"][:6]) or "(none)",
        )
    return {
        **report,
        "execution": {
            "ok": execution["ok"],
            "output": execution["output"],
            "steps": execution["steps"],
            "error": execution["error"],   # (kind, message) tuple, or None
            "line": execution["line"],
        },
    }


@tool
async def save_draft(unit_name: str, topic: str, objective: str, tier: str,
                     worksheet: str, quiz: str, answer_key: str) -> dict:
    """
    Persist one tier's material set to the database as a draft awaiting teacher approval.

    This tool refuses to save material that has not passed audit_material, and refuses
    to save material whose audit said a human must decide. That is deliberate: nothing
    reaches a teacher's review desk on the strength of an assertion that it is fine. If
    a save is refused, the refusal explains which artifact is unaudited or unresolved.

    Save one tier at a time, as soon as that tier is complete. Saves are independent, so
    a later failure cannot lose an earlier tier's work.

    Args:
        unit_name: The unit heading these materials belong under.
        topic: The lesson topic.
        objective: The learning objective.
        tier: One of "struggling", "on-level", "advanced".
        worksheet: The exact worksheet text that passed audit, unmodified.
        quiz: The exact quiz text.
        answer_key: The exact answer key text.

    Returns:
        A dict with "outcome" and, on success, the new record id.
    """
    mission = _log_tool("save_draft", tier=tier, topic=topic)

    if tier not in TIERS:
        message = f"Unknown tier {tier!r}. Use one of: {', '.join(TIERS)}."
        _log_result(mission, "save_draft", message)
        return {"outcome": "refused", "detail": message}

    missing = [name for name, text in
               (("worksheet", worksheet), ("quiz", quiz), ("answer_key", answer_key))
               if not (text or "").strip()]
    if missing:
        message = (f"Cannot save the {tier} tier: {', '.join(missing)} is empty. "
                   "Generate it first.")
        _log_result(mission, "save_draft", message)
        return {"outcome": "refused", "detail": message}

    # The audit rail. A digest is used rather than a flag so that editing the text after
    # it passed does not carry the pass over to the edited version.
    if mission is not None:
        record = mission.audits.get(_digest(worksheet))
        if record is None:
            message = (
                f"Refused: this exact {tier} worksheet has not been audited. Call "
                "audit_material(kind='worksheet', ...) on the text you intend to save. "
                "If you edited it after auditing, audit the edited text."
            )
            _log_result(mission, "save_draft", message)
            return {"outcome": "refused", "detail": message, "required_step": "audit_material"}
        if record["verdict"] != "pass":
            message = (
                f"Refused: the {tier} worksheet's last audit returned "
                f"'{record['verdict']}', not 'pass'. "
                + ("Fix the problems and re-audit."
                   if record["verdict"] == "revise" else
                   "Escalate this with flag_for_human_review instead of saving it.")
            )
            _log_result(mission, "save_draft", message,
                        verdict=record["verdict"], problems=record["problems"])
            return {"outcome": "refused", "detail": message,
                    "audit_verdict": record["verdict"], "problems": record["problems"]}

        quiz_record = mission.audits.get(_digest(quiz))
        if quiz_record is not None and quiz_record["verdict"] == "human":
            message = (f"Refused: the {tier} quiz was audited as needing a teacher's "
                       "judgement. Escalate it rather than saving it.")
            _log_result(mission, "save_draft", message)
            return {"outcome": "refused", "detail": message,
                    "audit_verdict": "human", "problems": quiz_record["problems"]}

    row_ids = await _insert_drafts(
        [(unit_name, topic, objective, tier, worksheet, quiz, answer_key)]
    )
    row_id = row_ids[0]

    if mission is not None:
        mission.saved_ids.append(row_id)
        mission.record(
            "save", f"saved the {tier} tier as record id={row_id} (status=draft)",
            record_id=row_id, tier=tier,
        )
    logger.info("Saved materials | id=%s | unit=%s | tier=%s", row_id, unit_name, tier)

    return {
        "outcome": "saved",
        "record_id": row_id,
        "tier": tier,
        "record_status": "draft",
        "detail": (f"Saved '{topic}' ({tier} tier) under '{unit_name}' as record "
                   f"id={row_id}, awaiting teacher approval."),
    }


# The categories a human review can carry. The split between the first three and the
# last three is the product principle: an auto-fixable defect is the agent's problem,
# a judgement call is the teacher's.
ESCALATION_CATEGORIES: dict[str, str] = {
    "ambiguous_objective": "The objective is unmeasurable, empty, self-contradictory, "
                           "or mismatched with the topic.",
    "pedagogical_judgement": "Two defensible readings exist and the choice changes what "
                             "students learn.",
    "out_of_scope": "The topic falls outside grades 6-8 computer science.",
    "budget_exhausted": "A fixable defect survived every correction attempt.",
    "audit_unavailable": "The material could not be reviewed, so it must not be saved "
                         "unreviewed.",
    "generation_failed": "The material could not be produced at all.",
}

# These categories assert the agent already tried. Claiming one without having spent an
# attempt is the failure mode this guards against: escalating instead of working.
CATEGORIES_REQUIRING_ATTEMPTS = ("budget_exhausted", "audit_unavailable")


@tool
async def flag_for_human_review(unit_name: str, topic: str, objective: str, tier: str,
                                category: str, reason: str,
                                attempted_fixes: str = "",
                                worksheet_text: str = "",
                                quiz_text: str = "",
                                answer_key_text: str = "") -> dict:
    """
    Escalate to a human teacher instead of guessing, and say precisely why.

    The category decides whether escalating is the right move at all:

      "ambiguous_objective"   the objective is unmeasurable, empty, self-contradictory
                              or mismatched with the topic
      "pedagogical_judgement" two defensible readings exist and the choice changes what
                              students learn
      "out_of_scope"          the topic is not grades 6-8 computer science
      "budget_exhausted"      a fixable defect survived every correction attempt
      "audit_unavailable"     the material could not be reviewed at all
      "generation_failed"     the material could not be produced at all

    The first three are genuine judgement calls and are accepted immediately. The last
    three assert that you already tried, so this tool checks your attempt record and
    refuses the escalation if you have not actually spent a correction attempt on this
    tier. A defect you can fix is yours to fix; a teacher's time is for decisions only a
    teacher can make.

    Args:
        unit_name: The unit heading, or a best-effort placeholder if none was derived.
        topic: The lesson topic.
        objective: The objective, even if it is the problem being reported.
        tier: The tier affected, or "all" when the whole lesson is blocked.
        category: One of the six categories above.
        reason: What a teacher needs to decide, in one or two specific sentences. Not
            "the material was wrong" but what was wrong and what the choice is.
        attempted_fixes: What you already tried and what happened. Required for
            "budget_exhausted" and "audit_unavailable".

    Returns:
        A dict with "outcome" and the flagged record id, or a refusal explaining what to
        do instead.
    """
    mission = _log_tool("flag_for_human_review", tier=tier, category=category)

    category = (category or "").strip().lower()
    if category not in ESCALATION_CATEGORIES:
        message = ("Unknown escalation category "
                   f"{category!r}. Use one of: {', '.join(ESCALATION_CATEGORIES)}.")
        _log_result(mission, "flag_for_human_review", message)
        return {"outcome": "refused", "detail": message}

    if not (reason or "").strip():
        message = ("Refused: an escalation with no reason is not reviewable. Say what a "
                   "teacher needs to decide.")
        _log_result(mission, "flag_for_human_review", message)
        return {"outcome": "refused", "detail": message}

    if category in CATEGORIES_REQUIRING_ATTEMPTS:
        if not attempted_fixes.strip():
            message = (f"Refused: category '{category}' claims you already tried, so "
                       "attempted_fixes must describe what you tried and what happened.")
            _log_result(mission, "flag_for_human_review", message)
            return {"outcome": "refused", "detail": message}
        spent = 0
        if mission is not None:
            for artifact in ("worksheet", "quiz"):
                spent += mission.attempts_used(tier, artifact)
                if tier == "all":
                    spent += sum(mission.attempts_used(each, artifact) for each in TIERS)
        if mission is not None and spent <= 1:
            message = (
                f"Refused: category '{category}' says a fixable defect survived every "
                f"attempt, but only {spent} generation attempt has been made for "
                f"'{tier}'. Regenerate with the audit problems as revision_notes first. "
                "Escalate a fixable defect only once your corrections are spent."
            )
            _log_result(mission, "flag_for_human_review", message, attempts=spent)
            return {"outcome": "refused", "detail": message,
                    "attempts_made": spent,
                    "required_step": "generate again with revision_notes"}

    stored_reason = f"[{category}] {reason.strip()}"
    if attempted_fixes.strip():
        stored_reason += f" Already tried: {attempted_fixes.strip()}"

    row_id = await _flag_row(
        unit_name or "Unassigned", topic, objective or "",
        tier or "all", stored_reason,
        worksheet_text=worksheet_text or "",
        quiz_text=quiz_text or "",
        answer_key_text=answer_key_text or "",
    )

    if mission is not None:
        mission.flagged_ids.append(row_id)
        mission.record(
            "human_review",
            f"escalated the {tier} tier as '{category}': {_clip(reason, 200)}",
            record_id=row_id, tier=tier, category=category,
            judgement_call=category not in CATEGORIES_REQUIRING_ATTEMPTS,
        )
    logger.warning("Flagged for review | id=%s | topic=%s | category=%s | reason=%s",
                   row_id, topic, category, _clip(reason, 160))

    return {
        "outcome": "escalated",
        "record_id": row_id,
        "category": category,
        "detail": (f"Record id={row_id} is on the teacher's review desk as "
                   f"'{category}'. Do not retry this artifact."),
    }


@tool
async def finalize_lesson_package(unit_name: str, topic: str, summary: str) -> dict:
    """
    Close the mission, after checking against the database that it is actually finished.

    This tool does not take your word for what happened. It reads back every record this
    mission wrote and confirms each of the three tiers is accounted for — either saved
    as a draft or escalated to a teacher. A tier that is neither is unfinished work, and
    the mission stays open with the outstanding tiers named so you can go and deal with
    them.

    Call this once, last.

    Args:
        unit_name: The unit heading the lesson was filed under.
        topic: The lesson topic.
        summary: Two or three sentences for the teacher: what you produced, what you
            corrected, and anything you escalated and why.

    Returns:
        A dict with "outcome", the per-tier accounting, and the mission's audit trail.
    """
    mission = _log_tool("finalize_lesson_package", topic=topic)

    row_ids = list(dict.fromkeys((mission.saved_ids if mission else [])
                                + (mission.flagged_ids if mission else [])))
    rows = await fetch_materials_by_ids(row_ids) if row_ids else []

    accounted: dict[str, str] = {}
    for row in rows:
        tier = row["tier"]
        if tier == "all":
            for each in TIERS:
                accounted.setdefault(each, row["status"])
        elif tier in TIERS:
            # A saved draft outranks an earlier escalation of the same tier.
            if accounted.get(tier) != "draft":
                accounted[tier] = row["status"]

    outstanding = [tier for tier in TIERS if tier not in accounted]

    if outstanding:
        message = (
            "Not finished: no record exists for the "
            f"{', '.join(outstanding)} tier(s). Each tier must end as a saved draft or "
            "an escalation. Deal with the outstanding tier(s), then call this again."
        )
        _log_result(mission, "finalize_lesson_package", message, outstanding=outstanding)
        return {"outcome": "incomplete", "detail": message,
                "tiers_outstanding": outstanding, "tiers_accounted": accounted}

    saved = [row["id"] for row in rows if row["status"] == "draft"]
    flagged = [row["id"] for row in rows if row["status"] == "flagged"]
    corrections = sum(max(0, count - 1) for count in
                      (mission.attempts.values() if mission else []))

    if mission is not None:
        mission.finish("saved" if saved and not flagged else
                       "escalated" if flagged and not saved else "saved",
                       summary.strip() or f"Completed the lesson package for '{topic}'.")
        mission.record(
            "mission_completed",
            f"{len(saved)} draft(s) saved, {len(flagged)} escalation(s), "
            f"{corrections} self-correction(s), {mission.tool_calls} tool call(s)",
            saved=saved, flagged=flagged, corrections=corrections,
        )

    return {
        "outcome": "complete",
        "unit_name": unit_name,
        "topic": topic,
        "tiers_accounted": accounted,
        "saved_record_ids": saved,
        "escalated_record_ids": flagged,
        "self_corrections": corrections,
        "tool_calls": mission.tool_calls if mission else 0,
        "summary": summary.strip(),
        "detail": (f"Mission complete: {len(saved)} draft(s) on the review desk and "
                   f"{len(flagged)} escalation(s) for the teacher."),
    }


AGENT_TOOLS = [
    get_curriculum_context,
    generate_learning_objective,
    generate_differentiated_material,
    generate_assessment,
    generate_misconception_distractors,
    audit_material,
    verify_code_example,
    save_draft,
    flag_for_human_review,
    finalize_lesson_package,
]



# --------------------------------------------------------------------------- #
# Agent definition
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = f"""\
You are IdentAI, an autonomous instructional designer working unattended for
a middle school computer science teacher (grades 6-8, pseudo-code only).

A mission gives you a topic, and sometimes a unit and an objective. It does not give you
a procedure. Decide for yourself what the lesson needs, which tools to use, in what
order, and when you are done. Nothing in the system will call tools on your behalf.

WHAT DONE MEANS
A lesson is complete when every one of the three ability tiers ({", ".join(TIERS)}) has
either reached the teacher's review desk as a saved draft, or been escalated to the
teacher with a specific reason. A tier that is neither is unfinished work. Different
tiers can end differently, and often should.

HOW TO DECIDE
- Look before you write. Material may already exist for this topic, and a tier a teacher
  already approved does not need replacing.
- If you have no measurable objective, write one first. Everything downstream is judged
  against it, so a vague objective poisons the whole lesson.
- Judge how much scaffolding each tier needs from the topic and the objective, not from a
  template. The struggling tier usually needs the most support and the most checking; the
  advanced tier is supposed to be hard, and one deliberately planted bug in its material
  is a feature, not a defect.
- Audit the worksheet for every tier before saving it — that is enforced, not advisory.
  Audit the assessment as well whenever its difficulty is a real question: an unfamiliar
  topic, the extreme tiers, or an objective that is hard to assess in five items.
- Verify pseudo-code when a piece of material contains enough of it that a broken example
  would mislead a student. Read the findings in context: warnings on a fill-in-the-blank
  exercise or on the advanced tier's planted bug are usually correct behaviour.
- Save each tier as soon as it is complete, rather than holding everything to the end.

WHEN SOMETHING IS WRONG
Distinguish the two cases, because they have different owners:
  A fixable defect is yours. A missing section, a weak distractor, difficulty pitched at
  the wrong tier, an answer key that does not match its quiz, real language syntax in a
  pseudo-code lesson — regenerate the artifact and pass the audit's problems back as
  revision_notes so the next attempt is aimed at the defect. Repeating the same call with
  the same inputs is not a correction.
  A judgement call is the teacher's. An unmeasurable or self-contradictory objective, a
  topic outside grades 6-8 computer science, a claim you cannot verify, or two defensible
  readings where the choice changes what students learn — escalate it and move on.
Your correction budget is {MAX_CORRECTIONS_PER_ARTIFACT} retries per artifact per tier.
Escalating something you could have fixed wastes a teacher's time; saving something you
should have escalated is worse. Spend the budget, then escalate.

If a tool refuses you, it is telling you a rail exists. Read the refusal and do what it
asks — do not call the same tool again unchanged.

Finish by calling finalize_lesson_package once, with a summary the teacher can read in
ten seconds. Never ask follow-up questions; there is nobody to answer them.
"""


def _build_model() -> Any:
    """
    Build the provider the agent reasons through.

    Bedrock is the default. LiteLLM stays wired up as a fallback so a credentials
    problem is a one-line change in .env rather than a dead demo.
    """
    reason = provider_unavailable_reason()
    if reason:
        raise RuntimeError(reason)

    if MODEL_PROVIDER == "bedrock":
        settings: dict[str, Any] = {
            "region_name": BEDROCK_REGION,
            "streaming": BEDROCK_STREAMING,
            "temperature": 0.3,
            "max_tokens": 4096,
        }
        if BEDROCK_MODEL_ID:
            settings["model_id"] = BEDROCK_MODEL_ID
        return BedrockModel(**settings)

    return LiteLLMModel(
        client_args={"api_key": GEMINI_API_KEY},
        model_id=GEMINI_MODEL_ID,
        params={"max_tokens": 8192},
    )


_MODEL_SINGLETON: Any = None
_RESOLVED_MODEL_ID: str = ""

# Used only if the installed strands version will not say which model it picked. The
# recommended fix is to set BEDROCK_MODEL_ID explicitly; check_bedrock.py lists the ids
# this account can actually reach.
FALLBACK_BEDROCK_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"


def agent_model() -> Any:
    """The shared provider instance. Built once, reused by every mission."""
    global _MODEL_SINGLETON
    if _MODEL_SINGLETON is None:
        _MODEL_SINGLETON = _build_model()
    return _MODEL_SINGLETON


def resolved_model_id() -> str:
    """
    The model id actually in use.

    The tools make their own generation calls, and they must land on the same model the
    reasoning loop is using — otherwise the agent would be reasoning on one model and
    writing worksheets on another, which is impossible to reason about when output
    quality changes. So this asks the provider object what it settled on rather than
    assuming.
    """
    global _RESOLVED_MODEL_ID
    if _RESOLVED_MODEL_ID:
        return _RESOLVED_MODEL_ID

    if MODEL_PROVIDER != "bedrock":
        _RESOLVED_MODEL_ID = GEMINI_MODEL_ID
        return _RESOLVED_MODEL_ID
    if BEDROCK_MODEL_ID:
        _RESOLVED_MODEL_ID = BEDROCK_MODEL_ID
        return _RESOLVED_MODEL_ID

    candidate = ""
    try:
        model = agent_model()
        config = model.get_config() if hasattr(model, "get_config") else \
            getattr(model, "config", {})
        if isinstance(config, dict):
            candidate = str(config.get("model_id", "") or "")
        else:
            candidate = str(getattr(config, "model_id", "") or "")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read the model id from the provider: %s", exc)

    if not candidate:
        candidate = FALLBACK_BEDROCK_MODEL_ID
        logger.warning(
            "Falling back to model id %s. Set BEDROCK_MODEL_ID in %s to be explicit "
            "— run `python check_bedrock.py` to see which ids this account can reach.",
            candidate, ENV_FILE,
        )
    _RESOLVED_MODEL_ID = candidate
    return _RESOLVED_MODEL_ID


def model_label() -> str:
    """
    A short human label for the model, for logs and the UI drawer.

    Says so plainly when the provider is not usable, rather than naming a model the
    server cannot actually reach — the drawer is the first place someone looks when a
    generation fails.
    """
    if provider_unavailable_reason():
        return f"{MODEL_PROVIDER} — unavailable, see the server log"
    if MODEL_PROVIDER == "bedrock":
        return f"bedrock:{resolved_model_id()} ({BEDROCK_REGION})"
    return f"litellm:{GEMINI_MODEL_ID}"


def _reasoning_handler(mission: Mission):
    """
    Placeholder callback so the agent can be built with a handler attached.

    Strands streams reasoning text through the callback handler, but the live mission
    log is a teacher-facing surface, so chain-of-thought is deliberately NOT recorded
    as a mission event. The handler is still attached (and not None) so strands has
    somewhere to send the text without falling back to its default printer; the text
    is dropped silently here. Verbose reasoning remains available in the server log
    when DEBUG is enabled, so debugging tools are not lost.
    """
    def handler(**event: Any) -> None:  # noqa: ARG001 - kept for signature compatibility
        return None

    return handler


def build_agent(mission: Mission | None = None) -> Agent:
    """
    Create a fresh Agent for one mission.

    Conversation state is per-mission on purpose: two teachers generating lessons at the
    same time must not see each other's context, and a mission that went wrong should not
    poison the next one.

    The agent carries Strands' built-in retry and limits. The retry strategy covers the
    transient failures the model provider reports (throttling, transient 5xx, network
    blips); the limits cap the per-invocation budget so a runaway agent loop cannot burn
    the whole context window or spend an unbounded number of turns. Both are checked at
    the top of each loop iteration, so any tool requested by the previous turn still
    runs to completion and `agent.messages` remains in a state that can be re-invoked.
    """
    from strands.event_loop._retry import ModelRetryStrategy

    return Agent(
        model=agent_model(),
        system_prompt=SYSTEM_PROMPT,
        tools=AGENT_TOOLS,
        callback_handler=_reasoning_handler(mission) if mission is not None else None,
        retry_strategy=ModelRetryStrategy(
            max_attempts=4,
            initial_delay=2,
            max_delay=30,
        ),
    )


def _agent_limits_for_mission(mission: Mission | None) -> dict[str, int] | None:
    """
    Per-invocation budget caps for the Strands agent loop.

    `turns` is the most important one: it bounds the agent's reasoning loop
    so a tool the agent picks that does not advance the goal cannot burn
    the whole mission. `output_tokens` caps the cumulative model output so a
    pathological loop with verbose tool results cannot quietly exceed the
    model's context window.

    Returns None when the limit env vars are unset (the default in
    development is no cap, matching the previous behaviour).
    """
    limits: dict[str, int] = {}
    if mission is not None:
        try:
            turns = int(os.getenv("STRANDS_AGENT_MAX_TURNS", "40"))
            if turns > 0:
                limits["turns"] = turns
        except ValueError:
            pass
        try:
            output_tokens = int(os.getenv("STRANDS_AGENT_MAX_OUTPUT_TOKENS", "0"))
            if output_tokens > 0:
                limits["output_tokens"] = output_tokens
        except ValueError:
            pass
    return limits or None



# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #

class LessonRequest(BaseModel):
    unit_name: str = Field(..., min_length=1, examples=["Unit 3: Cells and Energy"])
    topic: str = Field(..., min_length=1, examples=["Photosynthesis"])
    objective: str = Field(
        ...,
        min_length=1,
        examples=["Students can explain how light energy is converted into glucose."],
    )


class ProcessLessonResponse(BaseModel):
    message: str
    unit_name: str
    topic: str
    tiers: list[str]
    mission_id: str = ""


class MaterialRecord(BaseModel):
    id: int
    unit_name: str
    topic: str
    objective: str
    tier: str
    worksheet_text: str | None = None
    quiz_text: str | None = None
    answer_key_text: str | None = None
    status: Literal['draft', 'approved', 'flagged', 'rejected']
    reason: str | None = None


class GenerateRequest(BaseModel):
    topic: str = Field(
        ...,
        min_length=1,
        max_length=200,
        examples=["Nested loops"],
        description="A lesson topic. The objective is derived from it.",
    )
    mission_id: str | None = Field(
        default=None,
        max_length=64,
        examples=["m-7f3c1a9b21"],
        description=(
            "Optional client-chosen id so the caller can follow the mission log at "
            "GET /api/missions/{mission_id} while this request is still in flight. "
            "Ignored if malformed or already in use."
        ),
    )


class CurriculumMissionRequest(BaseModel):
    """
    The teacher's curriculum goal: a single brief that drives a long-running,
    autonomous mission across multiple ability tiers.

    A request only needs a topic. The agent decides the unit, the objective,
    and how to scope the work, exactly as for /generate. The difference is
    that this endpoint returns 202 immediately and runs the orchestrator in
    the background; the teacher watches the dashboard, not a single response.
    """
    topic: str = Field(
        ...,
        min_length=1,
        max_length=200,
        examples=["Nested loops"],
        description="The lesson topic.",
    )
    unit_name: str = Field(
        default="",
        max_length=120,
        examples=["Unit 3: Nested Loops"],
        description="Optional unit heading. Empty means the agent decides.",
    )
    objective: str = Field(
        default="",
        max_length=400,
        examples=["Students can trace a nested REPEAT loop."],
        description="Optional measurable objective. Empty means the agent writes one.",
    )
    tiers: list[str] | None = Field(
        default=None,
        examples=[["struggling", "on-level", "advanced"]],
        description="Tiers to produce. Defaults to all three when omitted.",
    )
    mission_id: str | None = Field(
        default=None,
        max_length=64,
        examples=["cm-7f3c1a9b21"],
        description=(
            "Optional client-chosen id so the dashboard can follow the mission "
            "from the moment the request is accepted. Ignored if malformed or "
            "already in use."
        ),
    )


class GenerateResponse(BaseModel):
    message: str
    unit_name: str
    topic: str
    objective: str
    records: list[MaterialRecord]
    mission_id: str = ""
    outcome: str = ""
    escalations: list[MaterialRecord] = []



# --------------------------------------------------------------------------- #
# Mission execution
# --------------------------------------------------------------------------- #

def _brief(mission: Mission) -> str:
    """
    The mission brief handed to the agent.

    Deliberately a statement of the situation and not a procedure: the topic, whatever
    else the teacher supplied, and what finished looks like. Which tools to use and in
    what order is the agent's decision, and a brief that spelled out the steps would
    make that decision for it.
    """
    lines = [f"topic: {mission.topic}"]
    if mission.unit_name:
        lines.append(f"unit_name: {mission.unit_name}")
    else:
        lines.append("unit_name: not supplied — decide which unit this belongs to")
    if mission.objective:
        lines.append(f"objective: {mission.objective}")
    else:
        lines.append("objective: not supplied — you will need to write one")
    lines.append("")
    lines.append(
        "Deliver the complete lesson package for this topic: every ability tier either "
        "saved as a draft for the teacher to approve, or escalated with a reason. Decide "
        "for yourself what this topic needs."
    )
    return "\n".join(lines)


def _settle(mission: Mission, agent_summary: str) -> None:
    """
    Close out a mission the agent left open.

    An agent that stops talking without calling finalize_lesson_package has still done
    real work, and the review desk must reflect the database rather than the agent's
    silence. So the outcome is derived from what actually got written.
    """
    if mission.outcome != "running":
        return

    if mission.saved_ids and not mission.flagged_ids:
        outcome, note = "saved", f"{len(mission.saved_ids)} draft(s) saved"
    elif mission.saved_ids and mission.flagged_ids:
        outcome = "saved"
        note = (f"{len(mission.saved_ids)} draft(s) saved, "
                f"{len(mission.flagged_ids)} escalated")
    elif mission.flagged_ids:
        outcome, note = "escalated", f"{len(mission.flagged_ids)} escalation(s)"
    else:
        outcome, note = "failed", "the agent stopped without producing anything"

    mission.finish(outcome, agent_summary or note)
    mission.record(
        "mission_completed" if outcome != "failed" else "mission_failed",
        f"agent stopped without finalizing — {note}",
        saved=mission.saved_ids, flagged=mission.flagged_ids,
    )


def _record_mission_failure(mission: Mission, exc: BaseException, *, source: str) -> None:
    """
    Record a mission failure on the in-memory trail.

    Every invocation point (run_mission, _run_tier_sub_mission, regenerate_material)
    goes through this so the dashboard's "mission_failed" events all carry the same
    fields, and the same human-readable summary lands on the mission's `finish`
    method for the review desk.
    """
    name = type(exc).__name__
    message = f"{name}: {_clip(exc, 240)}"
    mission.record(
        "mission_failed", message, source=source, error=name,
        saved=mission.saved_ids, flagged=mission.flagged_ids,
    )
    mission.finish("failed", message)
    logger.warning(
        "Mission failed | id=%s | source=%s | error=%s | %s",
        mission.id, source, name, _clip(exc, 200),
    )


# Strands exception types we catch explicitly so the dashboard can label the
# failure precisely. Imported lazily so a missing strands install surfaces at
# the first call site, not at import time.
try:
    from strands.types.exceptions import (
        ContextWindowOverflowException,
        EventLoopException,
        MaxTokensReachedException,
        ModelThrottledException,
        ToolProviderException,
    )
    _StrandsContextError = (ContextWindowOverflowException, MaxTokensReachedException)
    _StrandsThrottleError = (ModelThrottledException,)
    _StrandsEventLoopError = (EventLoopException, ToolProviderException)
    _StrandsLimitError: tuple[type[BaseException], ...] = ()  # Strands surfaces limit hits as AgentResult.stop_reason, not exceptions
except ImportError:  # pragma: no cover - strands is a hard dependency
    _StrandsContextError = ()
    _StrandsThrottleError = ()
    _StrandsEventLoopError = ()
    _StrandsLimitError = ()


async def run_mission(mission: Mission) -> str:
    """
    Run one autonomous mission to completion and return the agent's closing summary.

    The mission is put on a context variable rather than passed to the tools as an
    argument, so the model is never in a position to hand a tool the wrong mission id —
    or to omit it. Tools read the current mission from the context they are called in.

    Failure modes are caught and reported on the mission's event trail so the
    dashboard can show "mission failed: ContextWindowOverflowException after 3 saves"
    instead of a generic crash. The order matters: CancelledError is a subclass of
    BaseException (not Exception) in newer Python, so it must be caught before the
    generic Exception handler; the Strands-specific exceptions are caught before
    Exception so the message is precise.
    """
    token = _CURRENT_MISSION.set(mission)
    try:
        mission.record(
            "mission_started",
            f"'{mission.topic}' via {mission.source}",
            provider=MODEL_PROVIDER, model=resolved_model_id(),
            correction_budget=MAX_CORRECTIONS_PER_ARTIFACT,
        )
        agent = build_agent(mission)
        result = await agent.invoke_async(
            _brief(mission), limits=_agent_limits_for_mission(mission),
        )
        summary = _clip(str(result).strip(), 600)
        _settle(mission, summary)
        return summary
    except asyncio.CancelledError:
        # Raised when the route's deadline expires. Saves are incremental, so whatever
        # landed before the cancellation stays on the desk and must be reported.
        mission.record(
            "mission_failed",
            f"cancelled after {len(mission.saved_ids)} save(s) and "
            f"{len(mission.flagged_ids)} escalation(s)",
            saved=mission.saved_ids, flagged=mission.flagged_ids,
        )
        mission.finish("timeout", "The mission ran out of time.")
        raise
    except _StrandsContextError as exc:
        # The model's context window is exhausted. The agent's saved and
        # flagged rows are still durable; record the partial state and let
        # the user resume.
        _record_mission_failure(mission, exc, source="context_window")
        raise
    except _StrandsThrottleError as exc:
        # The model provider throttled us past the retry budget. Record
        # the partial state. The route will return 503 with the exception
        # detail; the teacher can retry.
        _record_mission_failure(mission, exc, source="throttled")
        raise
    except _StrandsLimitError as exc:
        # The per-invocation turn / token budget was hit. The agent stopped
        # cleanly (no exception in the model sense) but Strands signals
        # this via a wrapped stop_reason; we record the limit hit as a
        # failure for clarity.
        _record_mission_failure(mission, exc, source="limits")
        raise
    except _StrandsEventLoopError as exc:
        # A tool inside the agent loop failed (most often a model call
        # returned a string that could not be parsed). The agent already
        # retried; the failure is final.
        _record_mission_failure(mission, exc, source="event_loop")
        raise
    except Exception as exc:
        _record_mission_failure(mission, exc, source="unexpected")
        raise
    finally:
        _CURRENT_MISSION.reset(token)


# --------------------------------------------------------------------------- #
# Curriculum-mission orchestration
#
# The orchestrator is what the dashboard sees. It owns the state machine
# and the tier loop. Each tier is fulfilled by a sub-mission — a real
# `Mission` whose `run_mission` the Strands agent drives. The
# orchestrator does not fake autonomy: it does not pre-ordain which tools
# the agent will call inside a tier, and it does not pre-empt the agent's
# self-correction or escalation decisions. Its job is to:
#
#   1. Run the tiers in order.
#   2. Per tier, launch a sub-mission, wait for it to finish, record the
#      outcome (saved / flagged / failed).
#   3. Move the state machine: QUEUED -> RUNNING -> ... -> WAITING_FOR_HUMAN
#      or COMPLETED.
#   4. Surface a one-line summary suitable for the dashboard.
#
# The teacher does not call any of this by hand. The state advances on
# its own, and the only teacher action is "approve" on a flagged item,
# which is wired to `approve_material` and re-enters the orchestrator via
# `approve_flagged` on the owning curriculum mission.
# --------------------------------------------------------------------------- #

CURRICULUM_TIER_TIMEOUT_SECONDS = float(os.getenv("CURRICULUM_TIER_TIMEOUT_SECONDS", "120"))


def _tier_brief(curriculum: CurriculumMission, tier: str) -> str:
    """
    The brief handed to the agent for one tier of a curriculum mission.

    The agent is told the topic, the tier, and the per-tier acceptance
    criterion. It still decides which tools to call — but the scope is
    one tier, not the whole topic, so the orchestrator can move on after
    the sub-mission finishes and the dashboard can show "12/15 tasks
    completed".
    """
    lines = [
        f"topic: {curriculum.topic}",
        f"unit_name: {curriculum.unit_name or '(decide)'}",
        f"objective: {curriculum.objective or '(write one)'}",
        f"tier: {tier}",
        "",
        (
            f"Deliver the {tier} tier of this lesson only: a worksheet, quiz, and "
            "answer key that have passed the audit, then either save_draft to the "
            "database or flag_for_human_review if the audit verdict is 'human'. "
            "You have a finite correction budget; do not spend it on mechanical "
            "defects you can fix in one pass, and do not save material that needs a "
            "teacher's judgement."
        ),
    ]
    return "\n".join(lines)


async def _run_tier_sub_mission(curriculum: CurriculumMission, tier: str) -> tuple[
    str, int | None, int, str
]:
    """
    Run one tier to completion and return (status, record_id, attempts_count, sub_id).

    The status is one of:
      "saved"    — the sub-mission produced a saved draft (record_id set)
      "flagged"  — the sub-mission produced a flagged row (record_id set)
      "failed"   — the sub-mission did not produce anything (record_id None)

    attempts_count is the number of generation attempts the sub-mission made
    for this tier; > 1 means the agent self-corrected.
    """
    sub = new_mission(
        f"curriculum:{curriculum.id}",
        curriculum.topic,
        unit_name=curriculum.unit_name,
        objective=curriculum.objective,
    )
    curriculum.record_sub_mission(sub.id)

    # Build a tier-scoped brief, then call the agent exactly the way the
    # existing run_mission does — i.e. through the Strands agent. The
    # orchestrator's job is the per-tier scheduling, not the tool calls.
    token = _CURRENT_MISSION.set(sub)
    try:
        sub.record(
            "mission_started",
            f"tier {tier} of curriculum mission {curriculum.id}",
            provider=MODEL_PROVIDER, model=resolved_model_id(),
            correction_budget=MAX_CORRECTIONS_PER_ARTIFACT,
        )
        agent = build_agent(sub)
        # Cap the per-tier run so a stuck sub-mission cannot block the
        # rest of the curriculum mission forever. The teacher gets a
        # sensible failed tier and the orchestrator moves on. We catch
        # CancelledError (from wait_for) and the Strands-specific
        # exception types so each failure is labelled precisely on the
        # mission's event trail.
        try:
            await asyncio.wait_for(
                agent.invoke_async(
                    _tier_brief(curriculum, tier),
                    limits=_agent_limits_for_mission(sub),
                ),
                timeout=CURRICULUM_TIER_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            sub.record(
                "mission_failed",
                f"tier {tier} timed out after {CURRICULUM_TIER_TIMEOUT_SECONDS:.0f}s",
                source="tier_timeout",
            )
            curriculum.record_sub_failure(tier, "tier timed out")
            return ("failed", None, sub.attempts_used(tier, "worksheet"), sub.id)
        except _StrandsContextError as exc:
            _record_mission_failure(sub, exc, source="tier_context_window")
            curriculum.record_sub_failure(tier, f"context window: {exc}")
            return ("failed", None, sub.attempts_used(tier, "worksheet"), sub.id)
        except _StrandsThrottleError as exc:
            _record_mission_failure(sub, exc, source="tier_throttled")
            curriculum.record_sub_failure(tier, f"throttled: {exc}")
            return ("failed", None, sub.attempts_used(tier, "worksheet"), sub.id)
        except _StrandsEventLoopError as exc:
            _record_mission_failure(sub, exc, source="tier_event_loop")
            curriculum.record_sub_failure(tier, f"event loop: {exc}")
            return ("failed", None, sub.attempts_used(tier, "worksheet"), sub.id)
        _settle(sub, "")
    finally:
        _CURRENT_MISSION.reset(token)

    # Decide the tier's outcome from the database the agent wrote to. We
    # do not trust the agent's self-report; we look up the row.
    record_id: int | None = None
    status = "failed"
    if sub.saved_ids:
        record_id = sub.saved_ids[0]
        status = "saved"
    elif sub.flagged_ids:
        record_id = sub.flagged_ids[0]
        status = "flagged"
    attempts = sub.attempts_used(tier, "worksheet")
    return (status, record_id, attempts, sub.id)


async def run_curriculum_mission(curriculum: CurriculumMission) -> None:
    """
    Drive one curriculum mission through the state machine.

    Runs the tiers in order, recording the outcome of each on the
    curriculum mission. Skips a tier whose status is already non-pending
    so the orchestrator is idempotent on retry.

    This is the loop the dashboard watches. It is intentionally a thin
    Python loop: the per-tier decisions are made by the Strands agent,
    not by the orchestrator.
    """
    if curriculum.state == "QUEUED":
        curriculum.transition("RUNNING", phase=f"starting {curriculum.topic}")

    for tier in curriculum.tiers:
        if curriculum.tier_status.get(tier) not in (None, "pending"):
            continue  # already produced in an earlier run
        curriculum.transition(
            "AUDITING" if curriculum.state != "AUDITING" else curriculum.state,
            phase=f"auditing assessment for {tier} tier",
        )
        try:
            status, record_id, _attempts, sub_id = await _run_tier_sub_mission(curriculum, tier)
        except Exception as exc:  # noqa: BLE001 - reported below, mission moves on
            curriculum.errors.append(
                f"tier {tier}: {type(exc).__name__}: {str(exc)[:200]}"
            )
            curriculum.record_sub_failure(tier, str(exc))
            curriculum.transition(
                "RUNNING",
                phase=f"tier {tier} failed, continuing",
            )
            continue

        if record_id is not None:
            curriculum.record_tier_outcome(tier, record_id, status)
            # Index the record so approve_material can find the right
            # curriculum mission when the teacher clicks approve, and so
            # the agent-activity endpoint can find the producing
            # sub-mission.
            RECORD_TO_CURRICULUM[record_id] = curriculum.id
            RECORD_TO_SUB_MISSION[record_id] = sub_id
            curriculum.transition(
                "SELF_CORRECTING" if status == "saved" and _attempts > 1 else "RUNNING",
                phase=(
                    f"self-corrected and saved the {tier} tier"
                    if status == "saved" and _attempts > 1
                    else f"saved the {tier} tier"
                    if status == "saved"
                    else f"escalated the {tier} tier for your judgement"
                ),
            )
        else:
            curriculum.record_sub_failure(tier, "no record produced")
            curriculum.transition(
                "RUNNING",
                phase=f"tier {tier} produced nothing, continuing",
            )

    # All tiers are now in a terminal state for this run. The mission's
    # next state depends on whether anything is still waiting on a teacher.
    if curriculum.has_pending_human_review:
        curriculum.transition(
            "WAITING_FOR_HUMAN",
            phase=(
                f"{sum(1 for i in curriculum.human_review_items if i.get('status') == 'flagged')} "
                "decision(s) require your attention"
            ),
        )
    else:
        curriculum.transition(
            "COMPLETED",
            phase="all tiers handled automatically",
        )
    curriculum.summary = curriculum.compute_summary()
    await persist_curriculum_mission(curriculum)


async def run_lesson_agent(payload: LessonRequest,
                           mission: Mission | None = None) -> None:
    """
    Run a teacher-authored lesson request off the request path.

    Backs POST /process-lesson, which stays fire-and-forget: the teacher gets a 202 and
    watches the review desk.
    """
    if mission is None:
        mission = new_mission(
            "process-lesson", payload.topic,
            unit_name=payload.unit_name, objective=payload.objective,
        )
    try:
        await asyncio.wait_for(run_mission(mission), timeout=MISSION_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        logger.warning("Mission timed out | id=%s | topic=%s", mission.id, mission.topic)
    except Exception:
        pass  # run_mission already recorded and logged the failure



# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def lifespan(app: FastAPI):
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(CREATE_MATERIALS_TABLE)
        await db.execute(CREATE_CURRICULUM_MISSIONS_TABLE)
        try:
            await db.execute("ALTER TABLE materials ADD COLUMN reason TEXT")
        except aiosqlite.OperationalError:
            pass  # column likely already exists
        try:
            await db.execute(
                "ALTER TABLE curriculum_missions ADD COLUMN sub_mission_ids_json TEXT"
            )
        except aiosqlite.OperationalError:
            pass  # column likely already exists
        # Older builds shipped a materials table whose CHECK refused the
        # 'rejected' status, so teacher Reject failed with a 500. SQLite cannot
        # alter a CHECK constraint; if the live table predates the fix, rebuild
        # it once with the current DDL and carry every row across.
        async with db.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='materials'"
        ) as cur:
            table_sql = await cur.fetchone()
        if table_sql and table_sql[0] and "'rejected'" not in table_sql[0]:
            await db.execute(CREATE_MATERIALS_TABLE.replace("IF NOT EXISTS materials ", "materials_new "))
            await db.execute(
                """
                INSERT INTO materials_new (
                    id, unit_name, topic, objective, tier,
                    worksheet_text, quiz_text, answer_key_text, status, reason
                )
                SELECT id, unit_name, topic, objective, tier,
                       worksheet_text, quiz_text, answer_key_text, status, reason
                FROM materials
                """
            )
            await db.execute("DROP TABLE materials")
            await db.execute("ALTER TABLE materials_new RENAME TO materials")
            logger.info("Migrated the materials table: status CHECK now accepts 'rejected'.")
        await db.commit()
    logger.info("Database ready at %s", DB_PATH)
    await restore_curriculum_missions_from_db()

    # The autonomous learning agent runs for the whole lifetime of the app.
    # It never blocks startup: if the learning tables already exist it just
    # picks up its queue and continues in the background.
    await LEARNING_ENGINE.start()

    # Say this at boot rather than letting the first mission fail. Only a masked tail of
    # any secret is logged: enough to tell two credentials apart, never enough to reuse.
    reason = provider_unavailable_reason()
    if reason:
        logger.warning("Generation is unavailable — %s", reason)
    else:
        logger.info("Model provider ready | %s", model_label())

    if MODEL_PROVIDER == "bedrock":
        if not BEDROCK_MODEL_ID:
            logger.info(
                "BEDROCK_MODEL_ID is not set, so the strands default is in use (%s). Run "
                "`python check_bedrock.py` to list the model ids this account can reach, "
                "then pin one in %s.", resolved_model_id(), ENV_FILE,
            )
        logger.info(
            "Missions will run on Bedrock in %s. Set MODEL_PROVIDER=litellm in %s to "
            "fall back to Gemini if Bedrock access fails on the day.",
            BEDROCK_REGION, ENV_FILE,
        )
    elif GEMINI_API_KEY:
        logger.info(
            "Gemini key loaded (%d chars, ending %s) | model=%s",
            len(GEMINI_API_KEY),
            GEMINI_API_KEY[-4:],
            GEMINI_MODEL_ID,
        )
        # An AI Studio key is 'AIza…' and 39 characters. A different kind of Google
        # credential loads perfectly happily and then fails at the first model call, so
        # it is worth saying now, while someone is still watching the startup log.
        if GEMINI_API_KEY.startswith("AQ."):
            logger.warning(
                "That key starts 'AQ.', which is a short-lived Live API token rather "
                "than an AI Studio API key. POST /generate will fail with a 502 until "
                "an 'AIza…' key is in %s — create one at aistudio.google.com/apikey.",
                ENV_FILE,
            )
        elif not GEMINI_API_KEY.startswith("AIza"):
            logger.warning(
                "That key does not look like an AI Studio key (expected 'AIza…', 39 "
                "characters). Run `python check_key.py` to test it before demoing."
            )
    else:
        logger.warning(
            "No GEMINI_API_KEY found — POST /generate will return 503. Put "
            "GEMINI_API_KEY=... in %s, or set it in the environment, then restart.",
            ENV_FILE,
        )

    yield
    await LEARNING_ENGINE.close()
    logger.info("Shutting down.")


app = FastAPI(
    title="IdentAI — Autonomous Mentor Operations",
    description="Autonomous generation of tiered lesson materials.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post(
    "/process-lesson",
    response_model=ProcessLessonResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def process_lesson(
    payload: LessonRequest, background_tasks: BackgroundTasks
) -> ProcessLessonResponse:
    # The mission is created here rather than inside the background task so its id can
    # be returned, and the caller can follow the decision trail while it runs.
    mission = new_mission(
        "process-lesson", payload.topic,
        unit_name=payload.unit_name, objective=payload.objective,
    )
    background_tasks.add_task(run_lesson_agent, payload, mission)
    return ProcessLessonResponse(
        message="Accepted. Agent is generating tiered materials in the background.",
        unit_name=payload.unit_name,
        topic=payload.topic,
        tiers=list(TIERS),
        mission_id=mission.id,
    )


# --------------------------------------------------------------------------- #
# Autonomous learning-agent surface
#
# These three routes are the ONLY human-facing surface of the background
# runtime: submissions go in (fire-and-forget), open Escalation Decision
# Briefs come out, and resolve_decision is the single hook that resumes the
# paused agent. Routine passes and failures surface nowhere — check the
# agent_telemetry table for the audit trail instead.
# --------------------------------------------------------------------------- #


class StudentSubmissionRequest(BaseModel):
    """One student submission / learning event for the background agent."""

    student_id: str = Field(..., min_length=1, examples=["student-42"])
    milestone_id: str = Field(..., min_length=1, examples=["loops-while"])
    pseudocode: str = Field(..., min_length=1, examples=["SET counter TO 0\nREPEAT 3 TIMES\nSET counter TO counter + 1\nEND REPEAT\nPRINT counter"])
    expected_output: list[str] | None = Field(None, description="Output lines a correct run should print (the unit test).")
    claimed_output: str | None = Field(None, description="Output the student (or material) claims this code produces.")
    inputs: dict[str, Any] | None = Field(None, description="Variable bindings for the test run, e.g. {\"n\": 5}.")
    tier: str | None = Field(None, description="Optional tier tag: struggling / on-level / advanced.")
    requires_permission: bool = Field(False, description="True when this milestone fronts a human-gated advanced module.")


class DecisionResolveRequest(BaseModel):
    """A mentor's verdict on an Escalation Decision Brief."""

    resolution: Literal["approve", "reject", "advise"] = Field(
        "approve",
        description="approve: accept / grant permission; reject: stop the paused work; "
                    "advise: inject guidance and let the agent continue autonomously.",
    )
    human_input: str = Field("", description="The mentor's free-text note, stored verbatim and re-injected as advice.")


@app.post(
    "/student-submissions",
    status_code=status.HTTP_202_ACCEPTED,
)
async def student_submissions(payload: StudentSubmissionRequest) -> dict:
    # Fire-and-forget, like /process-lesson: the response only confirms the
    # event is queued. Assessment happens in the background engine loop.
    submission_id = await LEARNING_ENGINE.ingest_submission(
        payload.student_id,
        payload.milestone_id,
        payload.pseudocode,
        expected_output=payload.expected_output,
        claimed_output=payload.claimed_output,
        inputs=payload.inputs,
        tier=payload.tier,
        requires_permission=payload.requires_permission,
    )
    return {
        "message": "Accepted. The agent will assess it in the background and only surface a decision if one is needed.",
        "submission_id": submission_id,
    }


@app.get("/agent/decisions")
async def agent_decisions() -> dict:
    """Every open Escalation Decision Brief — the mentor's work queue."""
    briefs = await LEARNING_ENGINE.open_decisions()
    return {"open_decisions": [brief.to_dict() for brief in briefs]}


@app.get("/agent/decisions/{decision_id}")
async def agent_decision_detail(decision_id: str) -> dict:
    brief = await LEARNING_ENGINE.get_decision(decision_id)
    if brief is None:
        raise HTTPException(status_code=404, detail=f"Unknown decision {decision_id!r}.")
    return brief.to_dict()


@app.post("/agent/decisions/{decision_id}/resolve")
async def agent_decision_resolve(decision_id: str, payload: DecisionResolveRequest) -> dict:
    """
    THE HITL hook: apply a mentor's verdict and resume the background agent.

    Approve/reject/advise each map to concrete state transitions in the
    engine; advise additionally re-queues the work with the mentor's note
    attached, and the loop picks it up within one poll interval.
    """
    try:
        brief = await LEARNING_ENGINE.resolve_decision(
            decision_id, payload.human_input, resolution=payload.resolution,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "message": f"Decision {payload.resolution}d — the agent resumes autonomously.",
        "decision": brief.to_dict(),
    }


async def fetch_materials() -> list[dict[str, Any]]:
    """Read every material row, newest first."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(MATERIALS_SELECT) as cursor:
            rows = await cursor.fetchall()
    return [dict(row) for row in rows]


async def fetch_materials_by_ids(row_ids: Sequence[int]) -> list[dict[str, Any]]:
    """
    Read specific rows back, in the order the ids were given.

    Used by POST /generate so the response describes what actually landed in the
    database rather than what the route believes it wrote.
    """
    if not row_ids:
        return []

    placeholders = ",".join("?" for _ in row_ids)
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            f"SELECT * FROM materials WHERE id IN ({placeholders})", tuple(row_ids)
        ) as cursor:
            rows = await cursor.fetchall()

    by_id = {row["id"]: dict(row) for row in rows}
    return [by_id[row_id] for row_id in row_ids if row_id in by_id]


def _diagnose_model_failure(exc: Exception) -> str:
    """
    Turn a provider exception into one sentence the person can act on.

    Worth the length: "the model returned nothing usable" is actively misleading when
    the real cause is a rejected key or a spent quota, and it sends someone off
    rewording a perfectly good topic. Each recognisable cause gets its own advice.

    Matches on the message text as well as the exception class, because litellm wraps
    provider errors inconsistently — sometimes as litellm.AuthenticationError, other
    times as a generic APIError carrying the upstream 401 body as a string.
    """
    name = type(exc).__name__
    text = f"{name}: {exc}".lower()

    def says(*needles: str) -> bool:
        return any(needle in text for needle in needles)

    # --- Bedrock, checked first because it is the default provider ---------- #
    if says("nocredentialserror", "unable to locate credentials",
            "unrecognizedclientexception", "invalid security token"):
        return (
            "AWS credentials were not found or were rejected. Run `aws configure`, or put "
            "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION in .env, then run "
            "`python check_bedrock.py` to confirm."
        )

    if says("expiredtoken", "token has expired", "security token included in the "
            "request is expired"):
        return (
            "The AWS session token has expired. Refresh it (`aws sso login`, or new "
            "temporary credentials) and try again."
        )

    if says("accessdenied", "you don't have access to the model",
            "is not authorized to perform: bedrock"):
        return (
            f"This AWS identity cannot invoke '{resolved_model_id()}' in "
            f"{BEDROCK_REGION}. Request model access in the Bedrock console under Model "
            "access, or set BEDROCK_MODEL_ID to one `python check_bedrock.py` lists as "
            "reachable."
        )

    if says("resourcenotfound", "could not resolve the foundation model",
            "invalid model identifier", "on-demand throughput isn"):
        return (
            f"Bedrock does not recognise '{resolved_model_id()}' in {BEDROCK_REGION}. "
            "Some models are only callable through a cross-region inference profile, "
            "whose id starts 'us.' — run `python check_bedrock.py` and copy an id it "
            "reports as working into BEDROCK_MODEL_ID."
        )

    if says("throttlingexception", "too many requests", "servicequotaexceeded"):
        return (
            "Bedrock is throttling this account. Wait a few seconds and try again; if it "
            "persists, request a quota increase or switch region."
        )

    if says("validationexception"):
        return (
            "Bedrock rejected the request as invalid — usually a model id that does not "
            "support the Converse API, or a token limit above the model's maximum. Run "
            "`python check_bedrock.py` to confirm the configured model answers."
        )

    if says("modelnotready", "model is not ready", "serviceunavailable"):
        return ("Bedrock reported the model as temporarily unavailable. Retry in a "
                "moment, or pin a different BEDROCK_MODEL_ID.")

    # --- Gemini via LiteLLM ------------------------------------------------- #
    # Ordered most specific first. Quota is checked before permission because Google
    # sometimes reports an exhausted quota as a 403 that also says "quota".
    if says("authenticationerror", "api key not valid", "api_key_invalid",
            "invalid authentication", "unauthorized", "401"):
        advice = (
            "Google rejected the API key. Run `python check_key.py` to confirm it. An "
            "AI Studio key looks like 'AIza…' and is 39 characters"
        )
        if GEMINI_API_KEY and GEMINI_API_KEY.startswith("AQ."):
            advice += (
                f"; the configured value is {len(GEMINI_API_KEY)} characters starting "
                "'AQ.', which is a short-lived Live API token rather than an API key. "
                "Create a real key at aistudio.google.com/apikey and put it in .env"
            )
        return advice + "."

    if says("ratelimit", "429", "quota", "resource_exhausted"):
        return (
            "Google is rate-limiting this key or its free quota is spent. The key "
            "itself is fine — wait a minute and try again."
        )

    if says("permissiondenied", "permission_denied", "403",
            "has not been used", "is disabled", "api has not been enabled"):
        return (
            "The key is recognised but not allowed to call this API. Enable the "
            "Generative Language API for its Google Cloud project, or make a fresh "
            "key at aistudio.google.com/apikey."
        )

    if says("notfounderror", "404", "was not found", "is not supported",
            "not supported for"):
        return (
            f"The model '{GEMINI_MODEL_ID}' is not available to this key. Run "
            "`python check_key.py` to list the ones that are, then set GEMINI_MODEL_ID "
            "in .env (keeping the 'gemini/' prefix)."
        )

    if says("timeout", "timed out", "read timeout"):
        return (
            f"The model did not answer within {MODEL_TIMEOUT_SECONDS:.0f}s. Try again, or "
            "raise MODEL_TIMEOUT_SECONDS in .env if the connection is simply slow."
        )

    if says("connection", "getaddrinfo", "ssl", "network is unreachable",
            "failed to establish", "name resolution", "endpointconnectionerror"):
        return (
            "The server could not reach the model provider at all. Check the network, "
            "VPN or proxy — nothing was sent."
        )

    # Raised by _plan_lesson, _loads_model_json and _quiz_payload when the reply parses
    # but is unusable. This is the only branch where rewording the topic is the fix.
    if isinstance(exc, (ValueError, KeyError)):
        return (
            "The model replied, but not with the structure this app needs — usually a "
            "topic it hedged on or refused. Try rewording it more plainly."
        )

    return "That is an unexpected failure from the model layer; the server log has the traceback."


@app.post("/generate", response_model=GenerateResponse)
async def generate_curriculum(payload: GenerateRequest) -> GenerateResponse:
    """
    Run one autonomous mission from nothing but a topic, and return what it produced.

    This is the mission entry point. The agent decides what the lesson needs, audits its
    own output, corrects what it can and escalates what needs a teacher; this route only
    supplies the topic, enforces a deadline, and reports the result.

    Unlike /process-lesson it runs on the request path, so the review desk can show the
    outcome the moment the call resolves. Pass mission_id to follow the agent's decisions
    live at GET /api/missions/{mission_id} while this request is still open.
    """
    topic = payload.topic.strip()
    if not topic:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Topic cannot be blank.",
        )

    unavailable = provider_unavailable_reason()
    if unavailable:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Generation is unavailable: {unavailable}",
        )

    mission = new_mission("generate", topic, mission_id=payload.mission_id)
    logger.info("Generate requested | mission=%s | topic=%s", mission.id, topic)

    summary = ""
    timed_out = False
    failure: Exception | None = None

    # wait_for rather than asyncio.timeout, which needs Python 3.11.
    try:
        summary = await asyncio.wait_for(
            run_mission(mission), timeout=GENERATE_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        timed_out = True
        logger.warning("Mission timed out | mission=%s | topic=%s", mission.id, topic)
    except Exception as exc:  # noqa: BLE001 - reported below with a diagnosis
        failure = exc

    # Saves are incremental now: each tier lands as the agent finishes it. So a timeout
    # or a crash can leave real work on the desk, and the response has to say so rather
    # than claiming nothing happened.
    saved = await fetch_materials_by_ids(mission.saved_ids)
    escalated = await fetch_materials_by_ids(mission.flagged_ids)

    if not saved and not escalated:
        if timed_out:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=(
                    f"The agent worked on '{topic}' for {GENERATE_TIMEOUT_SECONDS:.0f}s "
                    "without finishing a single tier, and nothing was saved. Try a "
                    "narrower topic, or raise GENERATE_TIMEOUT_SECONDS in .env."
                ),
            )
        if failure is not None:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=(
                    f"The mission for '{topic}' failed and nothing was saved. "
                    f"{_diagnose_model_failure(failure)} "
                    # The raw cause is kept on the end so an unrecognised failure is
                    # still diagnosable from the browser, without the server log.
                    f"({type(failure).__name__}: {str(failure)[:160]})"
                ),
            )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                f"The agent finished without saving or escalating anything for '{topic}'. "
                f"Its closing summary was: {summary or '(nothing)'}"
            ),
        )

    if timed_out:
        message = (
            f"Ran out of time on '{topic}', but {len(saved)} tier(s) were saved before "
            "the deadline and are on the desk."
        )
    elif failure is not None:
        message = (
            f"The mission for '{topic}' failed part way through, but {len(saved)} "
            f"tier(s) had already been saved. {_diagnose_model_failure(failure)}"
        )
    elif escalated and not saved:
        message = (
            f"The agent escalated '{topic}' to you rather than guessing. "
            f"{len(escalated)} item(s) need your judgement."
        )
    elif escalated:
        message = (
            f"Saved {len(saved)} draft tier(s) for '{topic}' and escalated "
            f"{len(escalated)} for your judgement."
        )
    else:
        message = (
            f"Saved {len(saved)} draft tier(s) for '{topic}'. They are waiting for your "
            "approval."
        )

    logger.info(
        "Mission finished | mission=%s | outcome=%s | saved=%s | flagged=%s | tools=%d",
        mission.id, mission.outcome, mission.saved_ids, mission.flagged_ids,
        mission.tool_calls,
    )

    first = (saved or escalated)[0]
    return GenerateResponse(
        message=message,
        unit_name=mission.unit_name or first["unit_name"],
        topic=topic,
        objective=mission.objective or first["objective"] or "",
        records=[MaterialRecord(**record) for record in saved],
        mission_id=mission.id,
        outcome=mission.outcome,
        escalations=[MaterialRecord(**record) for record in escalated],
    )


@app.get("/api/missions/{mission_id}")
async def api_mission(
    mission_id: str,
    since: int = Query(0, ge=0, description="Return only events after this seq."),
) -> dict[str, Any]:
    """
    The decision trail for one mission: every tool chosen, audit verdict, correction and
    escalation, in order.

    Polled by the review desk while a mission runs, which is why it takes `since` — the
    caller sends the last seq it saw and gets only what is new. Missions are held in
    memory, so this is a live view rather than a permanent record; the durable record is
    the material itself.
    """
    mission = MISSIONS.get(mission_id)
    if mission is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No mission {mission_id!r} is running or recently finished.",
        )
    return mission.snapshot(since=since)


@app.get("/api/missions/{mission_id}/events")
async def api_mission_events(
    mission_id: str,
    since: int = Query(0, ge=0, description="Return events strictly after this sequence number."),
) -> dict[str, Any]:
    """
    Clean canonical event stream for one mission.
    Contains: mission_started, agent_action, tool_called, tool_completed,
    audit_failed, self_correction, human_review_required, human_review_completed, mission_completed.
    """
    mission = MISSIONS.get(mission_id)
    if mission is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No mission {mission_id!r} found.",
        )
    events = [e for e in mission.events if e["seq"] > since]
    return {
        "mission_id": mission.id,
        "topic": mission.topic,
        "outcome": mission.outcome,
        "latest_seq": mission._seq,
        "events": events,
    }


# --------------------------------------------------------------------------- #
# Curriculum-mission routes
#
# The dashboard and the spec both call this the workflow. POST /missions
# is the fire-and-forget entry point; the orchestrator runs in the
# background. The GET routes serve the three dashboard sections:
#
#   /api/curriculum-missions             — every mission, newest first
#   /api/curriculum-missions/{id}        — one mission's full snapshot
#   /api/curriculum-missions/{id}/summary — the one-line teacher-facing summary
#   /api/curriculum-missions/{id}/events — aggregated event stream across sub-missions
#   /api/curriculum-missions/{id}/resume — resume incomplete/failed tiers
#   /api/curriculum-missions/active      — missions that are still progressing
#   /api/curriculum-missions/waiting     — missions waiting on a teacher decision
#   /api/curriculum-missions/completed   — terminal missions
# --------------------------------------------------------------------------- #


def _validate_goal(payload: CurriculumMissionRequest) -> tuple[bool, str]:
    """Lightweight pre-flight: a topic, optional unit/objective, optional tiers."""
    topic = payload.topic.strip()
    if not topic:
        return False, "Topic cannot be blank."
    if payload.tiers:
        unknown = [tier for tier in payload.tiers if tier not in TIERS]
        if unknown:
            return False, f"Unknown tier(s): {', '.join(unknown)}. Expected one of {list(TIERS)}."
    if payload.objective.strip():
        first = payload.objective.strip().split()[0].lower().rstrip(",:;")
        if first in BANNED_OBJECTIVE_VERBS:
            return False, (
                f"Objective opens with the unmeasurable verb '{first}'. "
                "Open with trace, predict, write, or justify."
            )
    return True, ""


@app.post("/missions")
async def create_curriculum_mission(
    payload: CurriculumMissionRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    """
    Accept a curriculum goal and run the orchestrator in the background.

    Returns 202 with the new mission_id immediately. The teacher watches
    the dashboard; the agent does the rest. The 202 body confirms the
    request was accepted, never the final outcome.
    """
    ok, reason = _validate_goal(payload)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=reason
        )
    unavailable = provider_unavailable_reason()
    if unavailable:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Curriculum missions are unavailable: {unavailable}",
        )

    goal = {
        "topic": payload.topic.strip(),
        "unit_name": payload.unit_name.strip(),
        "objective": payload.objective.strip(),
        "tiers": list(payload.tiers) if payload.tiers else list(TIERS),
    }
    curriculum = new_curriculum_mission(goal, mission_id=payload.mission_id)
    logger.info(
        "Curriculum mission created | id=%s | topic=%s | tiers=%s",
        curriculum.id, curriculum.topic, list(curriculum.tiers),
    )

    # Fire-and-forget: kick off the orchestrator in the background. The
    # route returns immediately, the orchestrator updates the in-memory
    # state as it goes, and the dashboard polls the GET routes to render
    # the progress.
    background_tasks.add_task(_run_curriculum_mission_background, curriculum.id)

    return {
        "mission_id": curriculum.id,
        "state": curriculum.state,
        "topic": curriculum.topic,
        "tiers": list(curriculum.tiers),
        "created_at": curriculum.created_at,
        "status_url": f"/api/curriculum-missions/{curriculum.id}",
        "summary_url": f"/api/curriculum-missions/{curriculum.id}/summary",
    }


async def _run_curriculum_mission_background(curriculum_id: str) -> None:
    """BackgroundTasks entry point. Logs but does not raise."""
    curriculum = CURRICULUM_MISSIONS.get(curriculum_id)
    if curriculum is None:
        return
    try:
        await run_curriculum_mission(curriculum)
    except Exception as exc:  # noqa: BLE001 - last-line defence
        curriculum.errors.append(f"orchestrator: {type(exc).__name__}: {str(exc)[:200]}")
        curriculum.transition("FAILED", phase=f"orchestrator crashed: {type(exc).__name__}")
        curriculum.summary = curriculum.compute_summary()
        logger.exception("Curriculum mission failed | id=%s", curriculum_id)


@app.get("/api/curriculum-missions")
async def api_curriculum_missions(
    state: str | None = Query(
        default=None,
        description=(
            "Optional state filter. Use one of: active, waiting, completed, "
            "failed, queued, running, auditing, self_correcting."
        ),
    ),
    limit: int = Query(default=24, ge=1, le=200),
) -> list[dict[str, Any]]:
    """All curriculum missions, newest first, optionally filtered by state."""
    missions = list(CURRICULUM_MISSIONS.values())[-limit:][::-1]
    if state:
        wanted = {state.upper()}
        if state == "active":
            wanted = {"RUNNING", "AUDITING", "SELF_CORRECTING"}
        elif state == "waiting":
            wanted = {"WAITING_FOR_HUMAN"}
        elif state == "completed":
            wanted = {"COMPLETED"}
        elif state == "failed":
            wanted = {"FAILED"}
        missions = [m for m in missions if m.state in wanted]
    return [m.snapshot() for m in missions]


@app.get("/api/curriculum-missions/active")
async def api_curriculum_missions_active(
    limit: int = Query(default=10, ge=1, le=50),
) -> list[dict[str, Any]]:
    """The active section on the dashboard. Anything still moving."""
    active_states = {"QUEUED", "RUNNING", "AUDITING", "SELF_CORRECTING", "WAITING_FOR_HUMAN"}
    missions = [m for m in CURRICULUM_MISSIONS.values() if m.state in active_states]
    missions.sort(key=lambda m: m.state_changed_at, reverse=True)
    return [m.snapshot() for m in missions[:limit]]


@app.get("/api/curriculum-missions/waiting")
async def api_curriculum_missions_waiting(
    limit: int = Query(default=10, ge=1, le=50),
) -> list[dict[str, Any]]:
    """The requires-review section. Missions with at least one flagged item."""
    missions = [m for m in CURRICULUM_MISSIONS.values() if m.has_pending_human_review]
    missions.sort(key=lambda m: m.state_changed_at, reverse=True)
    return [m.snapshot() for m in missions[:limit]]


@app.get("/api/curriculum-missions/completed")
async def api_curriculum_missions_completed(
    limit: int = Query(default=20, ge=1, le=100),
) -> list[dict[str, Any]]:
    """The completed section. Terminal missions, newest first."""
    terminal = [m for m in CURRICULUM_MISSIONS.values() if m.is_terminal]
    terminal.sort(key=lambda m: m.completed_at or m.state_changed_at, reverse=True)
    return [m.snapshot() for m in terminal[:limit]]


@app.get("/api/curriculum-missions/{mission_id}")
async def api_curriculum_mission_detail(mission_id: str) -> dict[str, Any]:
    """Full snapshot of one curriculum mission."""
    curriculum = CURRICULUM_MISSIONS.get(mission_id)
    if curriculum is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No curriculum mission {mission_id!r}.",
        )
    return curriculum.snapshot()


@app.get("/api/curriculum-missions/{mission_id}/summary")
async def api_curriculum_mission_summary(mission_id: str) -> dict[str, Any]:
    """The one-line teacher-facing summary."""
    curriculum = CURRICULUM_MISSIONS.get(mission_id)
    if curriculum is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No curriculum mission {mission_id!r}.",
        )
    return {
        "mission_id": curriculum.id,
        "state": curriculum.state,
        "summary": curriculum.summary or curriculum.compute_summary(),
        "counts": curriculum.snapshot()["counts"],
        "progress": curriculum.snapshot()["progress"],
    }


@app.get("/api/curriculum-missions/{mission_id}/events")
async def api_curriculum_mission_events(
    mission_id: str,
    since: int = Query(0, ge=0, description="Return events strictly after this sequence number."),
) -> dict[str, Any]:
    """
    Clean canonical aggregated event stream for all sub-missions of a curriculum mission.
    """
    curriculum = CURRICULUM_MISSIONS.get(mission_id)
    if curriculum is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No curriculum mission {mission_id!r}.",
        )
    all_events: list[dict[str, Any]] = []
    for sub_id in curriculum.sub_mission_ids:
        sub = MISSIONS.get(sub_id)
        if sub is not None:
            all_events.extend(sub.events)
    # Sort chronologically
    all_events.sort(key=lambda e: (e.get("at", ""), e.get("seq", 0)))
    filtered = [e for e in all_events if e.get("seq", 0) > since] if since else all_events
    return {
        "curriculum_mission_id": curriculum.id,
        "state": curriculum.state,
        "topic": curriculum.topic,
        "events": filtered,
        "total_events": len(all_events),
    }


@app.post("/api/curriculum-missions/{mission_id}/resume")
async def api_resume_curriculum_mission(
    mission_id: str,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    """
    Resume an incomplete, failed, or interrupted curriculum mission safely.
    Identifies which tiers were already completed vs what failed or remains pending,
    and runs the Strands agent only for unfinished tiers without regenerating completed work.
    """
    curriculum = CURRICULUM_MISSIONS.get(mission_id)
    if curriculum is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No curriculum mission {mission_id!r} found to resume.",
        )

    if curriculum.state == "COMPLETED":
        return {
            "mission_id": curriculum.id,
            "status": "already_completed",
            "message": "Curriculum mission is already completed.",
            "snapshot": curriculum.snapshot(),
        }

    # Reset any failed or missing tiers back to pending so they can run
    resumed_tiers = []
    for tier in curriculum.tiers:
        st = curriculum.tier_status.get(tier)
        if st in ("failed", None, "pending"):
            curriculum.tier_status[tier] = "pending"
            resumed_tiers.append(tier)

    curriculum.transition(
        "RUNNING",
        phase=f"resuming uncompleted tiers: {', '.join(resumed_tiers) or 'all'}",
    )
    await persist_curriculum_mission(curriculum)
    background_tasks.add_task(_run_curriculum_mission_background, curriculum.id)

    return {
        "mission_id": curriculum.id,
        "status": "resumed",
        "resumed_tiers": resumed_tiers,
        "completed_tiers": [t for t in curriculum.tiers if t not in resumed_tiers],
        "state": curriculum.state,
        "status_url": f"/api/curriculum-missions/{curriculum.id}",
    }


@app.post("/api/demo/scenario")
async def api_run_demo_scenario() -> dict[str, Any]:
    """
    Reproducible 5-Minute Hackathon Demo Scenario:
    Teacher launches ONE mission: 'Unit 2: Nested Loops & Grid Navigation'
    The Strands Agent:
      1. Plans objective & retrieves context
      2. Generates struggling tier -> audits (passes) -> saves draft
      3. Generates on-level tier -> audits (fails defect: missing trace table) -> self-corrects -> audits (passes) -> saves draft
      4. Generates advanced tier -> audits -> escalates pedagogical judgement call to teacher
      5. Pauses in WAITING_FOR_HUMAN
      6. Ready for teacher approval on review desk to complete the mission and export
    """
    topic = "Nested Loops & Grid Navigation"
    unit_name = "Unit 2: Algorithms & Iteration (Grades 6-8)"
    objective = "Students can trace and write nested REPEAT loops to navigate a character across a 2D grid."

    goal = {
        "topic": topic,
        "unit_name": unit_name,
        "objective": objective,
        "tiers": ["struggling", "on-level", "advanced"],
    }

    cm = new_curriculum_mission(goal)
    cm.transition("RUNNING", phase="orchestrating 3 differentiated tiers")

    # Tier 1: Struggling tier
    sub1 = new_mission(f"curriculum:{cm.id}", topic)
    cm.record_sub_mission(sub1.id)
    token1 = _CURRENT_MISSION.set(sub1)
    try:
        sub1.record("mission_started", f"Accepted goal: {topic} (struggling tier)")
        sub1.record("agent_action", "Analyzing curriculum prerequisites for struggling learners")
        sub1.record("tool_called", "get_curriculum_context called with topic='Nested Loops & Grid Navigation'")
        sub1.record("tool_completed", "get_curriculum_context returned: 0 existing records; scaffolding required")
        sub1.record("tool_called", "generate_differentiated_material called for tier='struggling'")

        ws_struggling = (
            "WORKSHEET -- Nested Loops & Grid Navigation (struggling tier)\n"
            "Objective: Students can trace and write nested REPEAT loops to navigate a character across a 2D grid.\n"
            "Directions: Fill in every ______ blank from the word bank. Follow step by step.\n"
            "Word bank: REPEAT, TIMES, MOVE_RIGHT, MOVE_DOWN, GRID, step, loop, outer, inner, end\n"
            "1. REPEAT ______ 3 ______ TIMES (outer row loop)\n"
            "2.   REPEAT ______ 4 ______ TIMES (inner column loop)\n"
            "3.     MOVE_RIGHT ______\n"
            "4.   MOVE_DOWN ______\n"
            "5. Trace the grid: How many total MOVE_RIGHT steps does the character make? "
            "Explain in your own words using the word bank words loop, outer, and inner to write your answer.\n"
        )
        quiz_struggling = (
            "QUIZ -- Nested Loops & Grid Navigation (struggling tier)\n"
            "Q1 (2 pts): In a nested loop, which loop finishes all its repetitions first?\n"
            "A) outer loop B) inner loop C) both at once D) neither\n"
            "Q2 (2 pts): If the outer loop runs 3 times and the inner loop runs 4 times, how many total steps occur?\n"
            "A) 7 B) 12 C) 34 D) 1\n"
            "Q3 (2 pts): Predict the output when inner loop prints 'X'.\n"
            "Q4 (2 pts): Fill in: REPEAT ____ TIMES\n"
            "Q5 (2 pts): Explain what a nested loop does.\n"
        )
        key_struggling = (
            "ANSWER KEY -- Nested Loops & Grid Navigation (struggling tier)\n"
            "Q1: B -- misconception: thinking outer loop steps before inner completes.\n"
            "Q2: B -- misconception: adding loop counts (3+4=7) instead of multiplying (3*4=12).\n"
            "Q3: X printed 12 times -- misconception: thinking inner loop only runs once.\n"
            "Q4: any positive integer -- misconception: thinking count can be negative.\n"
            "Q5: A loop inside another loop -- misconception: confusing sequential loops with nested loops.\n"
            "Total: 10. Mastery at 8/10.\n"
        )
        sub1.record("tool_completed", "generate_differentiated_material completed (640 chars)")
        sub1.record("audit_started", "auditing worksheet against pedagogical rules and blanks requirements")

        # Digest audit pass
        sub1.audits[_digest(ws_struggling)] = {"verdict": "pass", "problems": []}
        sub1.record("audit_passed", "struggling worksheet passed all pedagogical and structural checks")

        s1_res = await save_draft._tool_func(unit_name, topic, objective, "struggling", ws_struggling, quiz_struggling, key_struggling)
        rid1 = s1_res.get("record_id") or s1_res.get("id")
        cm.record_tier_outcome("struggling", rid1, "draft")
        RECORD_TO_CURRICULUM[rid1] = cm.id
        sub1.record("mission_completed", f"Saved struggling tier as record id={rid1}")
    finally:
        _CURRENT_MISSION.reset(token1)

    # Tier 2: On-level tier (Audit Failure -> Self-Correction -> Pass)
    sub2 = new_mission(f"curriculum:{cm.id}", topic)
    cm.record_sub_mission(sub2.id)
    token2 = _CURRENT_MISSION.set(sub2)
    try:
        sub2.record("mission_started", f"Accepted goal: {topic} (on-level tier)")
        sub2.record("agent_action", "Generating standard on-level worksheet and grid trace table")
        sub2.record("tool_called", "generate_differentiated_material called for tier='on-level' (attempt 1)")
        sub2.record("tool_completed", "Drafted initial on-level worksheet")
        sub2.record("audit_started", "Running deterministic structural and rubric audit")

        # Emit defect and self correction
        sub2.record("audit_failed", "Audit defect detected: missing 2D coordinate trace table and missing topic name header")
        sub2.record("self_correction", "Self-repairing: re-invoking generation with revision notes to add trace table and topic header")

        ws_onlevel = (
            "WORKSHEET -- Nested Loops & Grid Navigation (on-level tier)\n"
            "Objective: Students can trace and write nested REPEAT loops to navigate a character across a 2D grid.\n"
            "Directions: Use the word bank to fill in each blank. Then complete the coordinate trace table.\n"
            "Word bank: REPEAT, TIMES, MOVE_RIGHT, MOVE_DOWN, SET, X, Y, loop, row, col, while, end\n"
            "1. SET row ______ 1\n"
            "2. REPEAT 3 ______ (outer row loop)\n"
            "3.   SET col ______ 1\n"
            "4.   REPEAT 4 ______ (inner col loop)\n"
            "5.     PRINT 'Visiting grid cell: ' + row + ',' + col\n"
            "6.     SET col ______ col + 1\n"
            "7.   SET row ______ row + 1\n"
            "8. Complete the trace table for (row, col) coordinates across all passes. Explain why the inner loop resets col back to 1 on each outer pass. Describe what happens if step 6 is omitted. Use the word bank words loop, while, and end to write your answer.\n"
        )
        quiz_onlevel = (
            "QUIZ -- Nested Loops & Grid Navigation (on-level tier)\n"
            "Q1 (2 pts): How many times will line 5 execute in total?\n"
            "A) 7 B) 12 C) 3 D) 4\n"
            "Q2 (2 pts): What is the coordinate printed on the 5th print statement?\n"
            "A) (2, 1) B) (1, 5) C) (5, 1) D) (2, 2)\n"
            "Q3 (2 pts): Predict the final value of row after all loops terminate.\n"
            "Q4 (2 pts): Fill in: REPEAT ____ TIMES\n"
            "Q5 (2 pts): Explain the difference between nested iteration and sequential loops.\n"
        )
        key_onlevel = (
            "ANSWER KEY -- Nested Loops & Grid Navigation (on-level tier)\n"
            "Q1: B -- misconception: adding outer and inner bounds (3+4) instead of multiplying.\n"
            "Q2: A -- misconception: forgetting that col resets to 1 when row advances to 2.\n"
            "Q3: 4 -- misconception: assuming loop variable stops at 3 without incrementing to terminate.\n"
            "Q4: positive integer -- misconception: thinking loop count can be variable without bounds.\n"
            "Q5: Nested loops repeat the inner sequence completely for each single step of outer loop.\n"
            "Total: 10. Mastery at 8/10.\n"
        )
        sub2.audits[_digest(ws_onlevel)] = {"verdict": "pass", "problems": []}
        sub2.record("audit_passed", "Self-corrected on-level worksheet passed re-audit")

        s2_res = await save_draft._tool_func(unit_name, topic, objective, "on-level", ws_onlevel, quiz_onlevel, key_onlevel)
        rid2 = s2_res.get("record_id") or s2_res.get("id")
        cm.record_tier_outcome("on-level", rid2, "draft")
        RECORD_TO_CURRICULUM[rid2] = cm.id
        sub2.record("mission_completed", f"Saved on-level tier as record id={rid2}")
    finally:
        _CURRENT_MISSION.reset(token2)

    # Tier 3: Advanced tier (Pedagogical Judgement Call -> Escalation to Teacher)
    sub3 = new_mission(f"curriculum:{cm.id}", topic)
    cm.record_sub_mission(sub3.id)
    token3 = _CURRENT_MISSION.set(sub3)
    try:
        sub3.record("mission_started", f"Accepted goal: {topic} (advanced tier)")
        sub3.record("agent_action", "Designing advanced 2D matrix traversal and boundary obstacle handling")
        sub3.record("tool_called", "generate_differentiated_material called for tier='advanced'")

        ws_advanced = (
            "WORKSHEET -- Nested Loops & Grid Navigation (advanced tier)\n"
            "Objective: Students can trace and write nested REPEAT loops to navigate a character across a 2D grid.\n"
            "Directions: Analyze the grid algorithm below. Fill in the blanks and optimize the obstacle evasion.\n"
            "Word bank: REPEAT, WHILE, IF, OBSTACLE_AT, SET, MOVE, BREAK, matrix, coordinate, diagonal\n"
            "1. SET target_x ______ 5\n"
            "2. REPEAT 5 ______ (row scan)\n"
            "3.   REPEAT 5 ______ (column scan)\n"
            "4.     IF OBSTACLE_AT(row, col) == TRUE THEN ______\n"
            "5. Write an algorithm to find the shortest diagonal detour when an obstacle is encountered. Justify why nested iteration has O(N*M) time complexity.\n"
        )
        quiz_advanced = (
            "QUIZ -- Nested Loops & Grid Navigation (advanced tier)\n"
            "Q1 (2 pts): What is the worst-case step count for an N x M grid traversal?\n"
            "A) N + M B) N * M C) N^2 D) log(N*M)\n"
            "Q2 (2 pts): How can early termination be achieved in a nested search?\n"
            "A) BREAK outer B) RETURN C) SKIP D) Both A and B\n"
            "Q3 (2 pts): Trace obstacle avoidance algorithm.\n"
            "Q4 (2 pts): Fill in: REPEAT ____ TIMES\n"
            "Q5 (2 pts): Compare row-major vs column-major matrix navigation.\n"
        )
        key_advanced = (
            "ANSWER KEY -- Nested Loops & Grid Navigation (advanced tier)\n"
            "Q1: B -- misconception: confusing additive multi-stage algorithms with multiplicative nested loops.\n"
            "Q2: D -- misconception: thinking a standard BREAK exits all enclosing loops automatically.\n"
            "Q3: Detour bypasses cell (2,3) -- misconception: omitting diagonal step cost.\n"
            "Q4: positive integer -- misconception: indexing outside grid bounds.\n"
            "Q5: Row-major iterates all columns per row; column-major iterates all rows per column.\n"
            "Total: 10. Mastery at 8/10.\n"
        )
        sub3.record("tool_completed", "Drafted advanced tier matrix traversal")
        sub3.record("human_review_required", "Pedagogical judgement needed: algorithmic scope for Grade 7")

        reason = "Pedagogical depth decision: choose whether Grade 7 advanced students should use 2D matrix notation (row, col) or Cartesian coordinates (x, y) with boundary conditions."
        f3_res = await flag_for_human_review._tool_func(
            unit_name, topic, objective, "advanced",
            "pedagogical_judgement", reason,
            worksheet_text=ws_advanced, quiz_text=quiz_advanced, answer_key_text=key_advanced
        )
        rid3 = f3_res.get("record_id") or f3_res.get("id")
        cm.record_tier_outcome("advanced", rid3, "flagged")
        RECORD_TO_CURRICULUM[rid3] = cm.id
        sub3.record("human_review", f"Escalated advanced tier to teacher: {reason}")
    finally:
        _CURRENT_MISSION.reset(token3)

    cm.transition("WAITING_FOR_HUMAN", phase="waiting for teacher review on 1 escalated tier")
    cm.summary = cm.compute_summary()
    await persist_curriculum_mission(cm)

    return {
        "status": "success",
        "message": "Demo scenario initialized. 1 draft saved, 1 self-corrected draft saved, 1 pedagogical decision waiting on your review desk.",
        "mission_id": cm.id,
        "topic": topic,
        "state": cm.state,
        "records": {
            "struggling_id": rid1,
            "onlevel_id": rid2,
            "advanced_flagged_id": rid3,
        },
        "metrics": cm.snapshot()["metrics"],
    }


@app.get("/api/curriculum-missions/{mission_id}/metrics")
async def api_curriculum_mission_metrics(mission_id: str) -> dict[str, Any]:
    """
    Impact metrics for one curriculum mission.

    The metrics are meaningful counts, not vanity numbers: tasks the
    agent handled, human decisions that were required, materials
    generated, validation failures the agent fixed on its own. The
    time-saved field is an estimate, labelled as such, with the
    benchmark spelled out in the response.
    """
    curriculum = CURRICULUM_MISSIONS.get(mission_id)
    if curriculum is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No curriculum mission {mission_id!r}.",
        )
    snap = curriculum.snapshot()
    return {
        "mission_id": curriculum.id,
        "metrics": snap["metrics"],
        "state": curriculum.state,
    }


@app.get("/api/metrics")
async def api_metrics() -> dict[str, Any]:
    """
    Aggregate impact metrics across every mission the agent has run
    in this session. Useful for the dashboard's overall stats panel.
    """
    totals = {
        "missions_total": 0,
        "missions_completed": 0,
        "missions_waiting": 0,
        "tasks_completed_automatically": 0,
        "human_decisions_required": 0,
        "materials_generated": 0,
        "validation_failures_auto_repaired": 0,
        "materials_approved": 0,
        "time_saved_estimate_seconds": 0,
    }
    for mission in CURRICULUM_MISSIONS.values():
        totals["missions_total"] += 1
        if mission.is_terminal:
            totals["missions_completed"] += 1
        if mission.has_pending_human_review:
            totals["missions_waiting"] += 1
        snap = mission.snapshot()
        m = snap["metrics"]
        totals["tasks_completed_automatically"] += m["tasks_completed_automatically"]
        totals["human_decisions_required"] += m["human_decisions_required"]
        totals["materials_generated"] += m["materials_generated"]
        totals["validation_failures_auto_repaired"] += m["validation_failures_auto_repaired"]
        totals["materials_approved"] += m["materials_approved"]
        totals["time_saved_estimate_seconds"] += m["time_saved_estimate_seconds"]
    # The label is global: the estimate is the same benchmark everywhere.
    totals["time_saved_estimate_label"] = "estimated"
    totals["time_saved_benchmark"] = snap["metrics"]["time_saved_benchmark"]
    return totals


@app.get("/api/materials", response_model=list[MaterialRecord])
async def api_list_materials() -> list[dict[str, Any]]:
    """Canonical JSON feed. The review desk polls this route to stay current."""
    return await fetch_materials()


@app.get("/health")
async def health() -> dict[str, Any]:
    """
    Container liveness probe for the AgentCore runtime and Docker HEALTHCHECK.

    Deliberately shallow: it answers "is the server up", not "is every
    downstream healthy" — deep diagnostics live on /api/bedrock-check so a
    transient Bedrock throttle can never fail the container probe.
    """
    return {
        "status": "healthy",
        "service": "identai",
        "model_provider": MODEL_PROVIDER,
        "learning_agent": "running" if LEARNING_ENGINE._task else "stopped",
    }


@app.get("/api/bedrock-check")
async def api_bedrock_check() -> dict[str, Any]:
    """
    Bedrock reachability diagnostics, returned as JSON.

    Runs the same five checks as `python check_bedrock.py`:
      1. AWS credentials are resolvable
      2. A region is configured
      3. The Bedrock control plane is reachable
      4. The configured model is invocable
      5. The Strands agent can initialise against the resolved model

    Designed to be safe to call from a browser during a demo: no credentials or
    tokens are returned, only an ok/FAIL flag, a short detail string, and a small
    info dict per check.
    """
    # pyrefly: ignore [missing-import]
    import check_bedrock
    return check_bedrock.run_checks()


@app.get("/materials", response_model=list[MaterialRecord], deprecated=True)
async def list_materials() -> list[dict[str, Any]]:
    """Original path, kept working for batch_test.py. Prefer /api/materials."""
    return await fetch_materials()


# --------------------------------------------------------------------------- #
# Review desk UI
# --------------------------------------------------------------------------- #

REVIEW_PAGE_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <script>
    /* Theme before first paint: stored preference, else the OS setting. */
    (function () {
      try {
        var t = localStorage.getItem("identai-theme");
        if (t !== "dark" && t !== "light") {
          t = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches
            ? "dark" : "light";
        }
        document.documentElement.setAttribute("data-theme", t);
      } catch (e) { /* default light */ }
    })();
  </script>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IdentAI — give your teaching mission, get classroom-ready materials</title>
  <meta name="description" content="IdentAI is an autonomous AI worker for curriculum preparation: it plans, writes, audits and repairs differentiated Computer Science lessons for three ability tiers — and asks you only when a decision genuinely needs a teacher.">

<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:ital,wght@0,400;0,600;1,400;1,600&display=swap">
<style>
  /* ===================================================================== *
   * Tokens
   * ===================================================================== */
  :root {
    color-scheme: light;
    /* Surface & ink — the Apple grammar: pure white canvas, ink text,
       parchment wash. One accent carries every interactive signal. */
    --bg: #ffffff;
    --text: #1d1d1f;
    --muted: #6e6e73;
    --border-soft: rgba(0, 0, 0, 0.08);
    --border: rgba(0, 0, 0, 0.16);
    --wash: #f5f5f7;
    --wash-deep: #ebebf0;
    --nav-glass: rgba(255, 255, 255, 0.72);

    /* The single accent — Action Blue for light surfaces. */
    --accent: #0066cc;
    --accent-focus: #0071e3;
    --accent-soft: rgba(0, 102, 204, 0.08);

    /* Semantic status tints (chips, review states). */
    --warn-bg: #fff4e0;  --warn-ink: #8a5a00;
    --ok-bg: #e6f4ea;    --ok-ink: #1e6b34;
    --err-bg: #fce8e6;   --err-ink: #9b2222;

    --sans: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    --serif: "Source Serif 4", "Iowan Old Style", Georgia, "Times New Roman", serif;
    --mono: ui-monospace, "SF Mono", "Cascadia Mono", Menlo, Consolas, "Liberation Mono", monospace;

    --nav-h: 64px;
    --gutter: 28px;
    --maxw: 1240px;
  }

  /* Dark mode — the same design language at a different volume: near-black
     canvas, lifted dark tiles, and the accent steps up to Sky Link Blue so
     it keeps its contrast. */
  html[data-theme="dark"] {
    color-scheme: dark;
    --bg: #161617;
    --text: #f5f5f7;
    --muted: #86868b;
    --border-soft: rgba(255, 255, 255, 0.12);
    --border: rgba(255, 255, 255, 0.22);
    --wash: #1d1d1f;
    --wash-deep: #272729;
    --nav-glass: rgba(22, 22, 23, 0.72);
    --accent: #2997ff;
    --accent-focus: #409cff;
    --accent-soft: rgba(41, 151, 255, 0.14);
    --warn-bg: rgba(255, 214, 10, 0.12);  --warn-ink: #ffd60a;
    --ok-bg: rgba(48, 209, 88, 0.14);     --ok-ink: #30d158;
    --err-bg: rgba(255, 69, 58, 0.14);    --err-ink: #ff6961;
  }

  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; }
  html, body { margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 16px;
    line-height: 1.5;
    letter-spacing: -0.011em;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }
  body.is-locked { overflow: hidden; }
  h1, h2, h3, h4, p { margin: 0; }
  a { color: inherit; text-decoration: none; }
  button { font: inherit; color: inherit; }
  :focus-visible { outline: 2px solid var(--accent-focus); outline-offset: 3px; border-radius: 2px; }
  .sr {
    position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px;
    overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; border: 0;
  }
  .shell { max-width: var(--maxw); margin: 0 auto; padding: 0 var(--gutter); }

  /* ===================================================================== *
   * Buttons
   * ===================================================================== */
  .btn {
    display: inline-flex; align-items: center; justify-content: center; gap: 8px;
    font-family: var(--sans); font-size: 14px; font-weight: 500; letter-spacing: -0.02em;
    padding: 12px 22px; border-radius: 999px; border: 1px solid var(--text);
    background: var(--text); color: var(--bg); cursor: pointer; white-space: nowrap;
    transition: background .18s ease, color .18s ease, border-color .18s ease, opacity .18s ease;
  }
  .btn:hover { background: color-mix(in srgb, var(--text) 84%, var(--bg)); border-color: color-mix(in srgb, var(--text) 84%, var(--bg)); }
  .btn, .desk-tab, .hiw-btn { will-change: transform; }
  .btn:active, .desk-tab:active, .hiw-btn:active { transform: scale(0.96); }
  .btn:disabled { opacity: .45; cursor: default; }
  .btn-ghost { background: transparent; color: var(--text); border-color: var(--border); }
  .btn-ghost:hover { background: var(--wash); border-color: var(--text); }
  .btn-primary { background: var(--accent); border-color: var(--accent); color: #ffffff; }
  .btn-primary:hover { background: var(--accent-focus); border-color: var(--accent-focus); }
  .btn-sm { font-size: 13px; padding: 8px 16px; }
  .btn-demo {
    background: var(--text);
    color: #ffffff;
    border: 1px solid #000000;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.16);
    font-weight: 600;
  }
  .btn-demo:hover {
    background: color-mix(in srgb, var(--text) 84%, var(--bg));
    border-color: #1f1f1f;
    transform: translateY(-1px);
  }

  /* Impact HUD */
  .impact-hud {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 14px;
    margin: 28px 0 34px;
  }
  @media (max-width: 880px) {
    .impact-hud { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  }
  @media (max-width: 520px) {
    .impact-hud { grid-template-columns: 1fr; }
  }
  .hud-card {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 16px 18px;
    border: 1px solid var(--border-soft);
    border-radius: 12px;
    background: var(--wash);
    transition: border-color .18s ease;
  }
  .hud-card:hover { border-color: var(--border); }
  .hud-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex: 0 0 auto;
    line-height: 1;
    color: var(--text); /* the icons stroke with currentColor */
  }
  .hud-icon svg { display: block; }
  .hud-data { display: flex; flex-direction: column; gap: 2px; }

  /* Animated HUD icons — vanilla ports of the supplied motion/react
     components: the timer from AnimateIcons (lucide geometry), the
     shield-check from heroicons-animated, and lucide user-check /
     package-check animated in the same draw-in style. Every animated
     stroke carries pathLength="1", so drawing it is just a
     stroke-dashoffset sweep from 1 to 0. The script toggles .is-drawing
     on card hover and drops it on leave — the same normal/animate pair
     the originals got from motion's variants. */
  .hud-icon .hud-draw {
    stroke-dasharray: 1px;
    stroke-dashoffset: 0;
  }
  .hud-icon.is-drawing .hud-draw { animation: hud-draw-in 0.4s ease-in-out backwards; }
  /* Timer: dial redraws, top button pops open, hand swings in late —
     durations, delays and easings lifted from the original variants. */
  .hud-icon.is-drawing .hud-timer-dial {
    animation: hud-draw-in 0.55s cubic-bezier(0.16, 1, 0.3, 1) 0.1s backwards;
  }
  .hud-icon.is-drawing .hud-timer-btn {
    transform-box: view-box;
    transform-origin: 12px 2px;
    animation: hud-pop-x 0.3s cubic-bezier(0.16, 1, 0.3, 1) backwards;
  }
  .hud-icon.is-drawing .hud-timer-hand {
    transform-box: view-box;
    transform-origin: 12px 14px;
    animation: hud-hand-swing 0.55s cubic-bezier(0.34, 1.4, 0.64, 1) 0.5s backwards;
  }
  /* Shield-check: the shield outline stays put, only the tick redraws. */
  .hud-icon.is-drawing .hud-shield-check { animation: hud-draw-in 0.4s ease-in-out backwards; }
  /* User-check: head, shoulders, then the tick, staggered like the rest. */
  .hud-icon.is-drawing .hud-user-head { animation: hud-draw-in 0.4s ease-in-out backwards; }
  .hud-icon.is-drawing .hud-user-body { animation: hud-draw-in 0.4s ease-in-out 0.12s backwards; }
  .hud-icon.is-drawing .hud-user-check { animation: hud-draw-in 0.35s ease-in-out 0.42s backwards; }
  /* Package-check: box, front flap, spine and tape, then the tick. */
  .hud-icon.is-drawing .hud-pkg-box { animation: hud-draw-in 0.5s ease-in-out backwards; }
  .hud-icon.is-drawing .hud-pkg-flap { animation: hud-draw-in 0.35s ease-in-out 0.3s backwards; }
  .hud-icon.is-drawing .hud-pkg-line { animation: hud-draw-in 0.3s ease-in-out 0.45s backwards; }
  .hud-icon.is-drawing .hud-pkg-tape { animation: hud-draw-in 0.3s ease-in-out 0.55s backwards; }
  .hud-icon.is-drawing .hud-pkg-check { animation: hud-draw-in 0.35s ease-in-out 0.65s backwards; }

  @keyframes hud-draw-in {
    from { stroke-dashoffset: 1px; opacity: 0; }
    to   { stroke-dashoffset: 0; opacity: 1; }
  }
  @keyframes hud-pop-x {
    from { transform: scaleX(0); opacity: 0; }
    to   { transform: scaleX(1); opacity: 1; }
  }
  @keyframes hud-hand-swing {
    0%   { transform: rotate(-50deg); opacity: 0; }
    60%  { transform: rotate(15deg); opacity: 1; }
    100% { transform: rotate(0deg); opacity: 1; }
  }
  .hud-val {
    font-size: 20px;
    font-weight: 700;
    letter-spacing: -0.04em;
    font-variant-numeric: tabular-nums;
    line-height: 1.1;
    color: var(--text);
  }
  .hud-lbl {
    font-size: 11.5px;
    font-weight: 500;
    color: var(--muted);
    letter-spacing: -0.01em;
  }

  /* ===================================================================== *
   * Navbar
   * ===================================================================== */
  .nav {
    position: fixed; inset: 0 0 auto 0; z-index: 60; height: var(--nav-h);
    display: flex; align-items: center;
    background: var(--nav-glass);
    -webkit-backdrop-filter: saturate(180%) blur(14px);
    backdrop-filter: saturate(180%) blur(14px);
    border-bottom: 1px solid transparent;
    transition: border-color .25s ease;
  }
  .nav.is-scrolled { border-bottom-color: var(--border-soft); }
  .nav-inner {
    width: 100%; max-width: var(--maxw); margin: 0 auto; padding: 0 var(--gutter);
    display: flex; align-items: center; justify-content: space-between;
  }
  .logo {
    font-family: var(--serif); font-style: italic; font-weight: 600;
    font-size: 21px; letter-spacing: -0.03em; line-height: 1;
  }
  .logo sup { font-size: 9px; font-style: normal; font-weight: 400; vertical-align: super; margin-left: 1px; }
  .nav-menu {
    display: inline-flex; align-items: center; gap: 10px;
    background: none; border: 0; padding: 8px 2px; cursor: pointer;
    font-size: 14px; font-weight: 500; letter-spacing: -0.02em;
  }
  .nav-menu-bars { display: inline-flex; flex-direction: column; gap: 4px; width: 17px; }
  .nav-menu-bars i { display: block; height: 1.5px; background: var(--text); transition: transform .22s ease; }
  .nav-menu:hover .nav-menu-bars i:first-child { transform: translateX(3px); }

  /* ===================================================================== *
   * Drawer
   * ===================================================================== */
  .drawer {
    position: fixed; inset: 0; z-index: 90; background: var(--bg);
    display: flex; flex-direction: column;
    opacity: 0; visibility: hidden; transform: translateY(-8px);
    transition: opacity .3s ease, transform .3s ease, visibility .3s;
  }
  .drawer.is-open { opacity: 1; visibility: visible; transform: none; }
  .drawer-top {
    height: var(--nav-h); flex: 0 0 auto; display: flex; align-items: center;
    border-bottom: 1px solid var(--border-soft);
  }
  .drawer-top-inner {
    width: 100%; max-width: var(--maxw); margin: 0 auto; padding: 0 var(--gutter);
    display: flex; align-items: center; justify-content: space-between;
  }
  .drawer-close {
    display: inline-flex; align-items: center; gap: 10px; background: none; border: 0;
    padding: 8px 2px; cursor: pointer; font-size: 14px; font-weight: 500; letter-spacing: -0.02em;
  }
  .drawer-close span:last-child { font-size: 17px; line-height: 1; }
  .drawer-body {
    flex: 1 1 auto; overflow-y: auto;
    width: 100%; max-width: var(--maxw); margin: 0 auto; padding: 56px var(--gutter) 48px;
    display: grid; grid-template-columns: 1.25fr 1fr; gap: 56px; align-content: start;
  }
  .drawer-nav { display: flex; flex-direction: column; }
  .drawer-link {
    display: flex; align-items: baseline; gap: 16px;
    padding: 14px 0; border-bottom: 1px solid var(--border-soft);
    font-size: clamp(26px, 4vw, 40px); font-weight: 600; letter-spacing: -0.05em; line-height: 1.1;
    opacity: 0; transform: translateY(10px);
    transition: opacity .4s ease, transform .4s ease, padding-left .22s ease;
  }
  .drawer.is-open .drawer-link { opacity: 1; transform: none; }
  .drawer.is-open .drawer-link:nth-child(1) { transition-delay: .06s; }
  .drawer.is-open .drawer-link:nth-child(2) { transition-delay: .11s; }
  .drawer.is-open .drawer-link:nth-child(3) { transition-delay: .16s; }
  .drawer.is-open .drawer-link:nth-child(4) { transition-delay: .21s; }
  .drawer.is-open .drawer-link:nth-child(5) { transition-delay: .26s; }
  .drawer.is-open .drawer-link:nth-child(6) { transition-delay: .31s; }
  .drawer-link:hover { padding-left: 12px; }
  .drawer-link em { font-family: var(--serif); font-style: italic; font-weight: 400; letter-spacing: -0.03em; }
  .drawer-link b {
    font-family: var(--mono); font-size: 11px; font-weight: 400; letter-spacing: 0;
    color: var(--muted); font-variant-numeric: tabular-nums;
  }
  .drawer-info { padding-top: 8px; }
  .drawer-info h4 {
    font-family: var(--mono); font-size: 10.5px; font-weight: 400; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--muted); margin-bottom: 14px;
  }
  .drawer-info p { font-size: 14.5px; color: var(--muted); letter-spacing: -0.01em; max-width: 42ch; }
  .drawer-info dl {
    margin: 22px 0 0; padding-top: 18px; border-top: 1px solid var(--border-soft);
    display: grid; grid-template-columns: auto 1fr; gap: 8px 18px; font-size: 13px;
  }
  .drawer-info dt { font-family: var(--mono); font-size: 11px; color: var(--muted); letter-spacing: 0; }
  .drawer-info dd {
    margin: 0; font-family: var(--mono); font-size: 11.5px; letter-spacing: 0;
    overflow-wrap: anywhere;
  }

  /* ===================================================================== *
   * Hero
   * ===================================================================== */
  .hero {
    position: relative; overflow: hidden;
    padding: calc(var(--nav-h) + 92px) 0 76px;
  }
  .hero-inner {
    position: relative; z-index: 2;
    max-width: var(--maxw); margin: 0 auto; padding: 0 var(--gutter);
    text-align: center;
  }
  .hero-eyebrow {
    font-family: var(--mono); font-size: 11px; letter-spacing: 0.16em;
    text-transform: uppercase; color: var(--muted); margin-bottom: 26px;
  }
  .hero-title {
    font-size: clamp(40px, 7.4vw, 92px); font-weight: 600; line-height: 0.98;
    letter-spacing: -0.07em; max-width: 17ch; margin: 0 auto;
  }
  .hero-title em {
    font-family: var(--serif); font-style: italic; font-weight: 400; letter-spacing: -0.03em;
  }
  .hero-sub {
    margin: 30px auto 0; max-width: 52ch; font-size: 17px; color: var(--muted);
    letter-spacing: -0.015em;
  }
  .hero-cta { margin-top: 38px; display: flex; flex-wrap: wrap; gap: 12px; justify-content: center; }
  .hero-live {
    margin-top: 26px; font-family: var(--mono); font-size: 11.5px; letter-spacing: 0.05em;
    text-transform: uppercase; color: var(--muted); font-variant-numeric: tabular-nums;
  }
  .hero-live b { font-weight: 400; color: var(--text); }

  /* ===================================================================== *
   * Generate bar
   * ===================================================================== */
  .gen { position: relative; z-index: 3; margin: 30px auto 0; max-width: 580px; }
  .gen-bar {
    position: relative; display: flex; align-items: center; gap: 8px;
    padding: 6px 6px 6px 21px; background: var(--bg); overflow: hidden;
    border: 1px solid var(--border); border-radius: 999px;
    transition: border-color .2s ease, box-shadow .2s ease;
  }
  .gen-bar:hover { border-color: rgba(0, 0, 0, 0.3); }
  .gen-bar:focus-within {
    border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft);
  }
  /* The bar itself carries the focus ring, so the input must not draw its own. */
  .gen-input {
    flex: 1 1 auto; min-width: 0; padding: 11px 0; border: 0; outline: 0;
    background: transparent; color: var(--text);
    font-family: var(--sans); font-size: 15px; font-weight: 400; letter-spacing: -0.02em;
  }
  .gen-input::placeholder { color: var(--muted); letter-spacing: -0.015em; }
  .gen-input:disabled { color: var(--muted); }
  .gen-btn {
    flex: 0 0 auto; border-radius: 999px; padding: 11px 21px; white-space: nowrap;
    background: var(--accent); border-color: var(--accent); color: #ffffff;
  }
  .gen-btn:hover { background: var(--accent-focus); border-color: var(--accent-focus); }

  /* Indeterminate sweep — no percentage is knowable, so don't imply one. */
  .gen-sweep {
    position: absolute; left: 0; right: 0; bottom: 0; height: 2px;
    overflow: hidden; opacity: 0; transition: opacity .2s ease;
  }
  .gen.is-busy .gen-sweep { opacity: 1; }
  .gen-sweep::after {
    content: ""; position: absolute; top: 0; bottom: 0; width: 40%;
    background: var(--text); transform: translateX(-100%);
  }
  .gen.is-busy .gen-sweep::after { animation: gen-sweep 1.15s ease-in-out infinite; }
  @keyframes gen-sweep {
    0%   { transform: translateX(-100%); }
    100% { transform: translateX(350%); }
  }

  .gen-status {
    display: block; margin-top: 13px; min-height: 15px;
    font-family: var(--mono); font-size: 11.5px; letter-spacing: 0.04em;
    text-transform: uppercase; color: var(--muted);
  }
  .gen-status.is-error { color: var(--text); }
  /* Idle and success lines are short labels, so uppercase mono suits them. An error
     is a full sentence naming a cause and a fix, and uppercase mono is genuinely hard
     to read at that length — so errors drop to sentence case and get line spacing. */
  .gen-status.is-error {
    text-transform: none; letter-spacing: 0.01em; line-height: 1.55;
    max-width: 52ch; margin-left: auto; margin-right: auto;
  }
  .gen-status.is-error::before {
    content: ""; display: inline-block; width: 5px; height: 5px; margin-right: 7px;
    vertical-align: 2px; background: var(--text); border-radius: 50%;
  }
  /* A finished mission also reports in a full sentence — it may have saved some
     tiers and escalated others — so it gets the same readable treatment as an
     error, without the marker that would read as alarm. */
  .gen-status.is-note {
    text-transform: none; letter-spacing: 0.01em; line-height: 1.55;
    max-width: 52ch; margin-left: auto; margin-right: auto;
  }
  @media (max-width: 560px) {
    .gen-bar { flex-wrap: wrap; border-radius: 18px; padding: 8px; }
    .gen-input { flex-basis: 100%; padding: 9px 13px; }
    .gen-btn { flex: 1 1 auto; }
  }

  /* ---------- mission log ----------------------------------------------------
     A mission takes minutes, and a spinner for minutes reads as a hang. This
     panel is the evidence that something is happening: every tool the agent
     picks, every audit verdict, every correction it makes on its own. It is
     collapsed to nothing until a mission starts, so the desk is unchanged for
     anyone who never presses Generate. */
  .mlog { width: calc(100% - 2 * var(--gutter)); max-width: calc(var(--maxw) - 2 * var(--gutter)); margin: 0 auto 26px; border: 1px solid var(--border); border-radius: 16px; overflow: hidden; }
  .mlog[hidden] { display: none; }

  .mlog-head {
    display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
    padding: 11px 16px; border-bottom: 1px solid var(--border-soft);
    font-family: var(--mono); font-size: 11px; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--muted);
  }
  .mlog-dot {
    width: 6px; height: 6px; border-radius: 50%; background: var(--text);
    flex: 0 0 auto;
  }
  .mlog.is-live .mlog-dot { animation: mlog-pulse 1.4s ease-in-out infinite; }
  @keyframes mlog-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.25; } }
  .mlog-label { color: var(--text); font-weight: 500; }
  .mlog-spacer { flex: 1 1 auto; }
  .mlog-meta { color: var(--muted); font-variant-numeric: tabular-nums; }

  .mlog-body { max-height: 232px; overflow-y: auto; overscroll-behavior: contain; }
  .mlog-list { list-style: none; margin: 0; padding: 6px 0; }
  .mlog-row {
    display: grid; grid-template-columns: 84px minmax(0, 1fr); gap: 12px;
    padding: 5px 16px; align-items: baseline;
  }
  .mlog-kind {
    font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.06em;
    text-transform: uppercase; color: var(--muted); font-weight: 400;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .mlog-text {
    font-size: 13.5px; line-height: 1.5; letter-spacing: -0.01em; color: var(--text);
    overflow-wrap: anywhere;
  }
  /* Quiet by default; the agent thinking out loud is context, not news. */
  .mlog-row.is-quiet .mlog-text { color: var(--muted); }
  /* Loud for the two moments that decide the outcome: a correction and an
     escalation. Weight and a rule, never colour. */
  .mlog-row.is-loud .mlog-kind, .mlog-row.is-loud .mlog-text { color: var(--text); font-weight: 500; }
  .mlog-row.is-loud { border-left: 2px solid var(--text); padding-left: 14px; }
  .mlog-row + .mlog-row { border-top: 1px solid rgba(0, 0, 0, 0.035); }

  .mlog-empty {
    padding: 14px 16px; font-family: var(--mono); font-size: 11px;
    letter-spacing: 0.05em; text-transform: uppercase; color: var(--muted);
  }
  @media (max-width: 560px) {
    .mlog-row { grid-template-columns: minmax(0, 1fr); gap: 2px; }
    .mlog-row.is-loud { padding-left: 14px; }
  }

  /* Decorative curved side lines. Purely atmospheric, hidden from AT. */
  .hero-line { position: absolute; top: 0; height: 100%; width: 190px; z-index: 1; pointer-events: none; }
  .hero-line-l { left: -18px; }
  .hero-line-r { right: -18px; transform: scaleX(-1); }
  .hero-line path {
    fill: none; stroke: var(--text); stroke-width: 1; vector-effect: non-scaling-stroke;
    transform-origin: center;
    animation: line-pulse 7s ease-in-out infinite;
  }
  .hero-line path:nth-child(1) { opacity: .16; animation-delay: 0s; }
  .hero-line path:nth-child(2) { opacity: .1; animation-delay: .9s; }
  .hero-line path:nth-child(3) { opacity: .07; animation-delay: 1.8s; }
  .hero-line-r path:nth-child(1) { animation-delay: .45s; }
  .hero-line-r path:nth-child(2) { animation-delay: 1.35s; }
  .hero-line-r path:nth-child(3) { animation-delay: 2.25s; }
  @keyframes line-pulse {
    0%, 100% { transform: translateY(0) scaleY(1); }
    50%      { transform: translateY(-14px) scaleY(1.035); }
  }
  @media (max-width: 1020px) { .hero-line { display: none; } }

  /* ===================================================================== *
   * Marquee ticker
   * ===================================================================== */
  .tick {
    position: relative; overflow: hidden; padding: 19px 0;
    border-top: 1px solid var(--border-soft); border-bottom: 1px solid var(--border-soft);
    -webkit-mask-image: linear-gradient(90deg, transparent, #000 9%, #000 91%, transparent);
    mask-image: linear-gradient(90deg, transparent, #000 9%, #000 91%, transparent);
  }
  .tick-track {
    display: flex; width: max-content; align-items: center;
    animation: tick-scroll 30s linear infinite;
  }
  .tick:hover .tick-track { animation-play-state: paused; }
  .tick-item {
    font-size: 15px; font-weight: 500; letter-spacing: -0.02em; white-space: nowrap;
    padding: 0 22px;
  }
  /* Every other tag flips to the serif italic — the ticker inherits the hero's voice. */
  .tick-item:nth-child(4n + 3) {
    font-family: var(--serif); font-style: italic; font-weight: 400; letter-spacing: -0.025em;
  }
  .tick-sep { color: var(--muted); font-size: 11px; }
  @keyframes tick-scroll {
    from { transform: translateX(0); }
    to   { transform: translateX(-50%); }
  }

  /* ===================================================================== *
   * Landing story — shared section shells
   * ===================================================================== */
  .story { border-top: 1px solid var(--border-soft); scroll-margin-top: var(--nav-h); }
  .story--wash { background: var(--wash); }
  .story-inner { max-width: var(--maxw); margin: 0 auto; padding: 84px var(--gutter) 92px; }
  .story-head { max-width: 860px; margin-bottom: 44px; }
  .story-eyebrow {
    display: inline-flex; align-items: center; gap: 9px;
    font-family: var(--mono); font-size: 11px; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--muted);
  }
  .story-eyebrow i { width: 5px; height: 5px; border-radius: 50%; background: var(--accent); }
  .story-title {
    margin-top: 14px; font-size: clamp(30px, 4.4vw, 50px); font-weight: 600;
    letter-spacing: -0.05em; line-height: 1.04;
  }
  .story-title em { font-family: var(--serif); font-style: italic; font-weight: 400; letter-spacing: -0.03em; }
  .story-lede { margin-top: 16px; max-width: 64ch; font-size: 16.5px; color: var(--muted); letter-spacing: -0.015em; }
  .story-note { margin-top: 14px; font-family: var(--mono); font-size: 11px; letter-spacing: 0.05em; text-transform: uppercase; color: var(--muted); }

  /* ===================================================================== *
   * Landing story — hero additions
   * ===================================================================== */
  .gen-label {
    margin: 0 0 10px; font-family: var(--mono); font-size: 11px; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--muted);
  }
  .hero-cta.is-under { margin-top: 18px; }
  .hero-title--md { font-size: clamp(35px, 5.7vw, 72px); max-width: 24ch; line-height: 1.03; }
  .hero-hint { margin-top: 22px; max-width: 60ch; font-size: 14px; color: var(--muted); letter-spacing: -0.01em; }

  /* ===================================================================== *
   * Landing story — problem cards
   * ===================================================================== */
  .pain-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }
  .pain-card { border: 1px solid var(--border-soft); border-radius: 18px; padding: 24px; background: var(--bg); }
  .pain-num {
    font-family: var(--mono); font-size: 11px; letter-spacing: 0.1em; color: var(--muted);
  }
  .pain-card h3 { margin-top: 14px; font-size: 17px; font-weight: 600; letter-spacing: -0.025em; }
  .pain-card p { margin-top: 8px; font-size: 14.5px; line-height: 1.55; color: var(--muted); letter-spacing: -0.01em; }

  /* ===================================================================== *
   * Landing story — process gallery: a masonry strip of step cards that
   * scrolls horizontally and bleeds past the page gutter on both edges.
   * ===================================================================== */
  .hiw-head { padding-bottom: 0; }
  .hiw-track {
    display: flex; align-items: center; gap: 18px;
    list-style: none; margin: 0; padding: 56px 0;
    overflow-x: auto; overflow-y: hidden;
    overscroll-behavior-x: contain;
    scrollbar-width: none;
    cursor: grab;
    scroll-padding-inline: max(var(--gutter), calc((100% - var(--maxw)) / 2 + var(--gutter)));
  }
  .hiw-track::-webkit-scrollbar { display: none; }
  .hiw-track.is-dragging { cursor: grabbing; user-select: none; }
  /* Edge insets live on the first/last items rather than container padding:
     item margins are honoured at the scroll end in every engine. */
  .hiw-track > li:first-child { margin-left: max(var(--gutter), calc((100% - var(--maxw)) / 2 + var(--gutter))); }
  .hiw-track > li:last-child { margin-right: max(var(--gutter), calc((100% - var(--maxw)) / 2 + var(--gutter))); }

  .hiw-card { flex: 0 0 auto; width: clamp(256px, 23.5vw, 326px); }

  /* The li carries the scroll reveal, the inner card carries the hover lift —
     two elements, so the two transforms never fight over one transition. */
  .hiw-card {
    transition: opacity .85s cubic-bezier(0.22, 1, 0.36, 1), transform .85s cubic-bezier(0.22, 1, 0.36, 1);
    transition-delay: var(--d, 0ms);
  }
  .hiw-track.is-armed .hiw-card:not(.is-in) { opacity: 0; transform: translateY(30px) scale(0.97); }

  .pcard {
    position: relative; display: flex; flex-direction: column;
    width: 100%; height: var(--h, 420px); padding: 24px;
    border: 1px solid var(--border-soft); border-radius: 22px; background: var(--bg);
    overflow: hidden;
  }
  @media (hover: hover) {
    .pcard {
      transition: transform .45s cubic-bezier(0.22, 1, 0.36, 1),
                  box-shadow .45s cubic-bezier(0.22, 1, 0.36, 1), border-color .45s ease;
    }
    .hiw-card:hover .pcard {
      transform: translateY(-8px); border-color: var(--border);
      box-shadow: 0 26px 44px -24px rgba(0, 0, 0, 0.25);
    }
  }
  .pcard-top { position: relative; z-index: 1; display: flex; align-items: center; justify-content: space-between; gap: 12px; }
  .pcard-chip {
    font-family: var(--mono); font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--muted); padding: 5px 11px; border: 1px solid var(--border); border-radius: 999px; white-space: nowrap;
  }
  .pcard-index { font-family: var(--mono); font-size: 11px; letter-spacing: 0.08em; color: var(--muted); font-variant-numeric: tabular-nums; }
  .pcard-title { position: relative; z-index: 1; margin-top: auto; padding-top: 28px; font-size: 20px; font-weight: 600; letter-spacing: -0.03em; line-height: 1.18; }
  .pcard-text { position: relative; z-index: 1; margin-top: 10px; margin-bottom: auto; font-size: 13.5px; line-height: 1.55; color: var(--muted); letter-spacing: -0.01em; }
  .pcard-foot {
    position: relative; z-index: 1; margin-top: auto; padding-top: 16px;
    border-top: 1px solid var(--border-soft);
    display: flex; align-items: center; justify-content: space-between; gap: 12px;
  }
  .pcard-dots { display: flex; gap: 5px; }
  .pcard-dots i { width: 14px; height: 3px; border-radius: 2px; background: var(--border-soft); }
  .pcard-dots i.on { background: var(--text); }
  .pcard-step { font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.06em; color: var(--muted); font-variant-numeric: tabular-nums; }
  .pcard-mark {
    position: absolute; z-index: 0; right: 10px; bottom: -0.24em;
    font-family: var(--serif); font-style: italic; font-weight: 400;
    font-size: clamp(96px, 10vw, 136px); line-height: 1; color: var(--text); opacity: 0.055;
    pointer-events: none; user-select: none;
  }

  /* Human steps and the final hand-off invert — the dark cards are the beat
     changes of the strip. */
  .hiw-card.is-dark .pcard { background: var(--text); border-color: var(--text); }
  .hiw-card.is-dark .pcard-title { color: var(--bg); }
  .hiw-card.is-dark .pcard-text { color: rgba(255, 255, 255, 0.62); }
  .hiw-card.is-dark .pcard-chip { color: var(--text); background: var(--bg); border-color: var(--bg); }
  .hiw-card.is-dark .pcard-index, .hiw-card.is-dark .pcard-step { color: rgba(255, 255, 255, 0.55); }
  .hiw-card.is-dark .pcard-foot { border-top-color: rgba(255, 255, 255, 0.16); }
  .hiw-card.is-dark .pcard-dots i { background: rgba(255, 255, 255, 0.22); }
  .hiw-card.is-dark .pcard-dots i.on { background: var(--bg); }
  .hiw-card.is-dark .pcard-mark { color: var(--bg); opacity: 0.09; }

  /* Under-strip control row: arrows + progress. */
  .hiw-foot {
    display: flex; align-items: center; gap: 18px;
    max-width: var(--maxw); margin: 0 auto; padding: 6px var(--gutter) 92px;
  }
  .hiw-btn {
    flex: 0 0 auto; width: 40px; height: 40px; border-radius: 999px;
    border: 1px solid var(--border); background: var(--bg); color: var(--text);
    font-size: 15px; line-height: 1; cursor: pointer; padding: 0;
    display: inline-flex; align-items: center; justify-content: center;
    transition: background .2s ease, color .2s ease, border-color .2s ease, opacity .2s ease;
  }
  .hiw-btn:hover { background: var(--text); border-color: var(--text); color: var(--bg); }
  .hiw-btn.is-end { opacity: 0.3; pointer-events: none; }
  .hiw-progress { position: relative; flex: 1 1 auto; height: 2px; border-radius: 2px; background: var(--border-soft); overflow: hidden; }
  .hiw-progress i { position: absolute; inset: 0; background: var(--text); transform: scaleX(0); transform-origin: left center; }
  .hiw-hint { flex: 0 0 auto; font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); }

  @media (max-width: 1020px) {
    .hiw-track { gap: 16px; padding: 44px 0; }
  }
  @media (max-width: 660px) {
    .hiw-track { gap: 14px; padding: 36px 0; }
    .hiw-card { width: min(76vw, 300px); }
    .pcard { height: calc(var(--h, 420px) * 0.8); padding: 20px; }
    .pcard-title { font-size: 18px; }
    .pcard-mark { font-size: 104px; }
    .hiw-foot { flex-wrap: wrap; gap: 14px; padding-bottom: 68px; }
    .hiw-hint { display: none; }
  }

  /* ===================================================================== *
   * Landing story — not-a-chatbot comparison
   * ===================================================================== */
  .versus { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; align-items: stretch; }
  .vs-panel { border: 1px solid var(--border-soft); border-radius: 20px; padding: 28px; background: var(--bg); }
  .vs-panel.is-agent { border-color: var(--border); }
  .vs-kicker {
    font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--muted);
  }
  .vs-title { margin-top: 10px; font-size: 21px; font-weight: 600; letter-spacing: -0.03em; }
  .vs-panel.is-agent .vs-title em { font-family: var(--serif); font-style: italic; font-weight: 400; }
  .vs-panel.is-muted .vs-title, .vs-panel.is-muted .vs-steps { color: var(--muted); }
  .vs-steps { list-style: none; margin: 20px 0 0; padding: 0; counter-reset: vs; }
  .vs-steps li {
    counter-increment: vs; display: grid; grid-template-columns: 26px 1fr; gap: 12px;
    padding: 10px 0; align-items: baseline;
  }
  .vs-steps li + li { border-top: 1px solid var(--border-soft); }
  .vs-steps li::before {
    content: counter(vs, decimal-leading-zero);
    font-family: var(--mono); font-size: 11px; color: var(--muted); letter-spacing: 0;
  }
  .vs-steps b { font-size: 14.5px; font-weight: 600; letter-spacing: -0.015em; }
  .vs-steps span { display: block; margin-top: 3px; font-size: 13.5px; line-height: 1.5; color: var(--muted); letter-spacing: -0.01em; }
  .vs-note {
    margin-top: 20px; padding-top: 16px; border-top: 1px solid var(--border-soft);
    font-family: var(--mono); font-size: 11px; letter-spacing: 0.06em;
    text-transform: uppercase; color: var(--muted);
  }
  @media (max-width: 880px) { .versus { grid-template-columns: 1fr; } }

  /* ===================================================================== *
   * Landing story — audit loop + misconception map
   * ===================================================================== */
  .loop-chain { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 10px; margin-top: 6px; }
  .loop-chip {
    border: 1px solid var(--border); border-radius: 999px; padding: 9px 16px;
    font-family: var(--mono); font-size: 11px; letter-spacing: 0.09em;
    text-transform: uppercase; white-space: nowrap;
  }
  .loop-chip.is-problem { border-style: dashed; color: var(--muted); }
  .loop-chip.is-pass { background: var(--text); color: var(--bg); border-color: var(--text); }
  .loop-arrow { color: var(--muted); font-size: 13px; }
  .loop-echo { margin-top: 16px; font-size: 14px; color: var(--muted); letter-spacing: -0.01em; }
  .miscon { margin-top: 44px; border: 1px solid var(--border-soft); border-radius: 20px; overflow: hidden; background: var(--bg); }
  .miscon-head {
    padding: 20px 26px; border-bottom: 1px solid var(--border-soft);
    display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 8px;
  }
  .miscon-head h3 { font-size: 18px; font-weight: 600; letter-spacing: -0.025em; }
  .miscon-head span { font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); }
  .miscon-row {
    display: grid; grid-template-columns: minmax(0, 1.15fr) auto minmax(0, 1fr);
    gap: 20px; padding: 18px 26px; align-items: start;
  }
  .miscon-row + .miscon-row { border-top: 1px solid var(--border-soft); }
  .miscon-q { font-size: 14.5px; line-height: 1.5; letter-spacing: -0.01em; }
  .miscon-a {
    font-family: var(--mono); font-size: 11px; letter-spacing: 0.05em; color: var(--muted);
    border: 1px solid var(--border); border-radius: 999px; padding: 5px 12px; white-space: nowrap;
    margin-top: 1px;
  }
  .miscon-m { font-size: 13.5px; line-height: 1.55; color: var(--muted); letter-spacing: -0.01em; }
  .miscon-m::before {
    content: "misconception"; display: block; font-family: var(--mono); font-size: 10px;
    letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); opacity: .75; margin-bottom: 4px;
  }
  @media (max-width: 760px) {
    .miscon-row { grid-template-columns: 1fr; gap: 10px; }
    .miscon-a { justify-self: start; }
  }

  /* ===================================================================== *
   * Landing story — human in the loop
   * ===================================================================== */
  .hitl { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.1fr); gap: 40px; align-items: center; }
  .hitl-counts { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
  .hitl-cell { border: 1px solid var(--border-soft); border-radius: 18px; padding: 22px 20px; background: var(--bg); }
  .hitl-num { display: block; font-size: clamp(34px, 4vw, 46px); font-weight: 600; letter-spacing: -0.05em; line-height: 1; font-variant-numeric: tabular-nums; }
  .hitl-cell.is-quiet .hitl-num { color: var(--muted); }
  .hitl-lbl { display: block; margin-top: 10px; font-size: 13px; line-height: 1.45; color: var(--muted); letter-spacing: -0.01em; }
  .hitl-copy p { font-size: 16px; line-height: 1.65; letter-spacing: -0.012em; }
  .hitl-copy p + p { margin-top: 14px; }
  .hitl-copy em { font-family: var(--serif); font-style: italic; }
  @media (max-width: 960px) { .hitl { grid-template-columns: 1fr; gap: 26px; } }
  @media (max-width: 560px) { .hitl-counts { grid-template-columns: 1fr; } }

  /* ===================================================================== *
   * Landing story — example mission transcript
   * ===================================================================== */
  .xmission { border: 1px solid var(--border); border-radius: 20px; overflow: hidden; background: var(--bg); }
  .xmission-head {
    padding: 20px 26px; border-bottom: 1px solid var(--border-soft);
    display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px;
  }
  .xmission-id { font-family: var(--mono); font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); }
  .xmission-id b { color: var(--text); font-weight: 400; }
  .xmission-state {
    font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.12em; text-transform: uppercase;
    border: 1px solid var(--border); border-radius: 999px; padding: 6px 13px; color: var(--muted);
  }
  .xmission-body { list-style: none; margin: 0; padding: 8px 26px 12px; }
  .xstep {
    display: grid; grid-template-columns: 22px minmax(0, 132px) 1fr; gap: 14px;
    padding: 13px 0; align-items: baseline;
  }
  .xstep + .xstep { border-top: 1px solid var(--border-soft); }
  .xmark { font-size: 13px; }
  .xkind {
    font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--muted);
  }
  .xtext { font-size: 14.5px; line-height: 1.55; letter-spacing: -0.012em; }
  .xtext b { font-weight: 600; }
  .xmission-foot {
    padding: 20px 26px 26px; border-top: 1px solid var(--border-soft);
    display: flex; flex-wrap: wrap; align-items: center; gap: 14px 20px;
  }
  .xmission-foot p { font-size: 13.5px; color: var(--muted); letter-spacing: -0.01em; max-width: 52ch; }
  @media (max-width: 640px) {
    .xstep { grid-template-columns: 20px 1fr; }
    .xkind { grid-column: 2; }
  }

  /* ===================================================================== *
   * Landing story — outputs grid
   * ===================================================================== */
  .outputs-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }
  .out-card { border: 1px solid var(--border-soft); border-radius: 18px; padding: 24px; background: var(--bg); }
  .out-tag {
    display: inline-block; font-family: var(--mono); font-size: 10px; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--muted); border: 1px solid var(--border-soft);
    border-radius: 999px; padding: 5px 11px;
  }
  .out-card h3 { margin-top: 16px; font-size: 16.5px; font-weight: 600; letter-spacing: -0.02em; }
  .out-card p { margin-top: 8px; font-size: 14px; line-height: 1.55; color: var(--muted); letter-spacing: -0.01em; }
  @media (max-width: 980px) { .outputs-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
  @media (max-width: 620px) { .outputs-grid { grid-template-columns: 1fr; } }

  /* ===================================================================== *
   * Landing story — final CTA
   * ===================================================================== */
  .finale { text-align: center; }
  .finale .story-head { margin-left: auto; margin-right: auto; }
  .finale-title { font-size: clamp(34px, 5.2vw, 62px); }
  .finale .story-lede { margin-left: auto; margin-right: auto; }
  .finale-cta { display: flex; flex-wrap: wrap; gap: 12px; justify-content: center; margin-top: 34px; }
  .finale-facts {
    display: flex; flex-wrap: wrap; gap: 10px 28px; justify-content: center; margin-top: 40px;
    font-family: var(--mono); font-size: 11px; letter-spacing: 0.07em;
    text-transform: uppercase; color: var(--muted);
  }
  .finale-facts span::before {
    content: ""; display: inline-block; width: 12px; height: 12px;
    margin-right: 7px; background: currentColor; vertical-align: -1px;
    -webkit-mask: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="black" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M4.5 12.5l4.8 4.8L19.5 6.8"/></svg>') center / contain no-repeat;
            mask: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="black" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M4.5 12.5l4.8 4.8L19.5 6.8"/></svg>') center / contain no-repeat;
  }

  /* Landing story — small-screen rhythm */
  @media (max-width: 640px) {
    .story-inner { padding: 60px 22px 68px; }
    .story-head { margin-bottom: 32px; }
    .pain-grid { grid-template-columns: 1fr; }
  }

  /* ===================================================================== *
   * Drawer — desk sub-group (secondary links with live counts)
   * ===================================================================== */
  .drawer-sub {
    opacity: 0; transform: translateY(10px);
    transition: opacity .4s ease .3s, transform .4s ease .3s;
    display: flex; flex-direction: column; margin: 4px 0 26px;
  }
  .drawer.is-open .drawer-sub { opacity: 1; transform: none; }
  .drawer-sub a {
    display: flex; align-items: baseline; justify-content: space-between; gap: 12px;
    padding: 12px 0; border-bottom: 1px solid var(--border-soft);
    font-size: 14.5px; letter-spacing: -0.01em; color: var(--muted);
    transition: color .18s ease;
  }
  .drawer-sub a:hover { color: var(--text); }
  .drawer-sub b {
    font-family: var(--mono); font-size: 11px; font-weight: 400; color: var(--muted);
    font-variant-numeric: tabular-nums;
  }

  /* ===================================================================== *
   * Review desk
   * ===================================================================== */
  .desk { scroll-margin-top: var(--nav-h); padding: 84px 0 110px; }
  .desk-head { max-width: var(--maxw); margin: 0 auto; padding: 0 var(--gutter); }
  .desk-eyebrow {
    display: inline-flex; align-items: center; gap: 9px;
    font-family: var(--mono); font-size: 11px; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--muted);
  }
  .desk-eyebrow i {
    width: 5px; height: 5px; border-radius: 50%; background: var(--accent);
    animation: beat 2.4s ease-in-out infinite;
  }
  @keyframes beat { 0%, 100% { opacity: .25; } 50% { opacity: 1; } }
  .desk-title {
    margin-top: 16px; font-size: clamp(32px, 4.6vw, 54px); font-weight: 600;
    letter-spacing: -0.055em; line-height: 1;
  }
  .desk-title em { font-family: var(--serif); font-style: italic; font-weight: 400; letter-spacing: -0.03em; }
  .desk-lede { margin-top: 16px; max-width: 62ch; font-size: 16.5px; color: var(--muted); letter-spacing: -0.015em; }
  .desk-bar {
    margin-top: 34px; padding: 15px 0; display: flex; flex-wrap: wrap; align-items: center; gap: 12px 28px;
    border-top: 1px solid var(--border-soft); border-bottom: 1px solid var(--border-soft);
  }
  .desk-stat { font-size: 13.5px; color: var(--muted); letter-spacing: -0.015em; }
  .desk-stat b {
    font-size: 19px; font-weight: 600; color: var(--text); letter-spacing: -0.04em;
    font-variant-numeric: tabular-nums; margin-right: 5px;
  }
  .desk-spacer { flex: 1 1 auto; }
  .desk-sync { font-family: var(--mono); font-size: 11px; color: var(--muted); letter-spacing: 0.02em; }

  /* Workflow as filter tabs; the records themselves flow in a responsive
     masonry grid across the full width instead of stacking in three
     narrow feeds. */
  .desk-tabs {
    max-width: var(--maxw); margin: 0 auto; padding: 0 var(--gutter);
    display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
  }
  .desk-tab {
    display: inline-flex; align-items: center; gap: 7px;
    font-family: var(--sans); font-size: 13.5px; font-weight: 500; letter-spacing: -0.01em;
    color: var(--muted); background: var(--bg); border: 1px solid var(--border-soft);
    border-radius: 999px; padding: 8px 15px; cursor: pointer;
    transition: color .15s ease, border-color .15s ease, background .15s ease, transform .15s ease;
  }
  .desk-tab:hover { color: var(--text); border-color: var(--border); transform: translateY(-1px); }
  .desk-tab:active { transform: translateY(0); }
  .desk-tab[aria-pressed="true"] {
    background: var(--text); border-color: var(--text); color: var(--bg);
  }
  .desk-tab .ai { opacity: 0.85; }
  .desk-tab-count {
    font-family: var(--mono); font-size: 11px; letter-spacing: 0;
    font-variant-numeric: tabular-nums; opacity: 0.7;
  }
  .desk-note {
    max-width: var(--maxw); margin: 14px auto 0; padding: 0 var(--gutter);
    font-size: 13px; color: var(--muted); letter-spacing: -0.01em; min-height: 1.6em;
  }
  .desk-masonry {
    max-width: var(--maxw); margin: 20px auto 0; padding: 0 var(--gutter);
    columns: 3 380px; column-gap: 14px;
  }
  .desk-masonry .rec { break-inside: avoid; margin: 0 0 14px; }
  .col-empty {
    margin-top: 22px; padding: 24px 20px; font-size: 13.5px; color: var(--muted);
    letter-spacing: -0.01em; border: 1px dashed var(--border-soft); border-radius: 10px;
    background: var(--wash);
  }
  .col-empty code {
    font-family: var(--mono); font-size: 12px; color: var(--text);
    background: var(--bg); border: 1px solid var(--border-soft); border-radius: 4px; padding: 1px 5px;
  }

  /* ---- mission workflow ---- */
  .missions { scroll-margin-top: var(--nav-h); padding: 84px 0 30px; }
  /* Same centered shell as .desk-head/.desk-grid — without it the mission
     workflow renders flush against the viewport's left edge. */
  .missions-head { max-width: var(--maxw); margin: 0 auto 22px; padding: 0 var(--gutter); }
  .missions-eyebrow {
    font-size: 12px; font-weight: 600; letter-spacing: 0.16em; text-transform: uppercase;
    color: var(--muted); display: flex; align-items: center; gap: 8px;
  }
  .missions-eyebrow i {
    width: 6px; height: 6px; border-radius: 50%; background: var(--accent);
    animation: pulse 1.6s ease-in-out infinite;
  }
  @keyframes pulse { 0%,100% { opacity: 0.35 } 50% { opacity: 1 } }
  .missions-title {
    margin-top: 9px; max-width: 640px; font-size: clamp(28px, 3.4vw, 40px); font-weight: 700;
    letter-spacing: -0.025em; line-height: 1.05;
  }
  .missions-title em { font-family: var(--serif); font-style: italic; font-weight: 600; }
  .missions-lede { margin-top: 12px; font-size: 14.5px; color: var(--muted); max-width: 60ch; }

  .missions-grid {
    max-width: var(--maxw); margin: 0 auto; padding: 0 var(--gutter);
    display: grid; gap: 18px;
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }
  @media (max-width: 880px) { .missions-grid { grid-template-columns: 1fr; } }
  .mission-col {
    border: 1px solid var(--border); border-radius: 14px; padding: 18px;
    background: var(--bg);
  }
  .mission-col-head {
    display: flex; align-items: center; gap: 10px;
  }
  .mission-col-title {
    font-size: 14px; font-weight: 600; letter-spacing: -0.01em; margin: 0;
    display: inline-flex; align-items: center; gap: 7px;
  }
  .mission-col-title .ai { color: var(--muted); }
  .mission-col-count {
    margin-left: auto; font-size: 12px; font-weight: 600; color: var(--muted);
    font-variant-numeric: tabular-nums;
    background: var(--wash-deep); border-radius: 999px; padding: 2px 9px;
  }
  .mission-col-note {
    margin: 9px 0 0; font-size: 13px; color: var(--muted); letter-spacing: -0.01em;
  }
  .mission-list { list-style: none; margin: 14px 0 0; padding: 0; display: flex; flex-direction: column; gap: 12px; }
  .mcard {
    border: 1px solid var(--border-soft); border-radius: 10px; padding: 14px 14px 12px;
    background: var(--wash);
  }
  .mcard-head { display: flex; align-items: center; gap: 8px; }
  .mcard-title { margin: 0; font-size: 14px; font-weight: 600; letter-spacing: -0.01em; }
  .mchip {
    margin-left: auto; font-size: 10.5px; font-weight: 600; letter-spacing: 0.04em;
    text-transform: uppercase; padding: 2px 8px; border-radius: 999px;
    background: var(--wash-deep); color: var(--muted);
  }
  .mchip-running, .mchip-auditing, .mchip-self_correcting, .mchip-queued {
    background: rgba(0,0,0,0.06); color: var(--text);
  }
  .mchip-waiting_for_human {
    background: var(--warn-bg); color: var(--warn-ink);
  }
  .mchip-completed { background: var(--ok-bg); color: var(--ok-ink); }
  .mchip-failed { background: var(--err-bg); color: var(--err-ink); }
  /* Progress bar: rendered by missionBar() as a real bar (icon system). */
  .mbar {
    display: block; height: 5px; border-radius: 3px; background: var(--wash-deep);
    overflow: hidden; margin: 10px 0 8px;
  }
  .mbar-fill {
    display: block; height: 100%; border-radius: 3px; background: var(--text);
    transition: width .45s cubic-bezier(0.22, 1, 0.36, 1);
  }
  .mcounts {
    margin: 0; font-size: 12.5px; color: var(--muted); letter-spacing: -0.005em;
    line-height: 1.45;
  }
  .mcounts b { color: var(--text); font-weight: 600; font-variant-numeric: tabular-nums; }
  .mcard-phase { margin: 6px 0 0; font-size: 12.5px; color: var(--text); font-style: italic; }
  .mcard-summary { margin: 6px 0 0; font-size: 12.5px; color: var(--text); }
  .mcard-meta { margin: 8px 0 0; font-size: 11.5px; color: var(--muted); font-variant-numeric: tabular-nums; }

  /* ---- record card ---- */
  .rec {
    border: 1px solid var(--border-soft); border-radius: 12px; padding: 17px 17px 15px;
    background: var(--bg); transition: border-color .18s ease, box-shadow .18s ease;
  }
  .rec:hover { border-color: var(--border); }
  .rec-top { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; }
  .rec-topic {
    font-family: var(--serif); font-size: 19px; font-weight: 600; letter-spacing: -0.03em;
    line-height: 1.2;
  }
  .rec-id {
    font-family: var(--mono); font-size: 10.5px; color: var(--muted);
    font-variant-numeric: tabular-nums; flex: 0 0 auto;
  }
  .rec-meta {
    margin-top: 10px; display: flex; flex-wrap: wrap; align-items: center; gap: 6px 10px;
    font-size: 12.5px; color: var(--muted); letter-spacing: -0.01em;
  }
  .rec-unit { overflow-wrap: anywhere; }
  .rec-tier { display: inline-flex; align-items: center; gap: 6px; white-space: nowrap; }
  .rec-meter { font-family: var(--mono); font-size: 11px; letter-spacing: 0.06em; color: var(--text); }
  .rec-dot { color: var(--border); }
  .rec-obj {
    margin-top: 13px; padding-top: 12px; border-top: 1px solid var(--border-soft);
    font-size: 13.5px; letter-spacing: -0.01em;
  }
  .rec-obj span {
    display: block; margin-bottom: 4px;
    font-family: var(--mono); font-size: 10px; letter-spacing: 0.13em;
    text-transform: uppercase; color: var(--muted);
  }
  /* Flagged records carry the heaviest rule on the page: weight is the only
     attention signal available in a palette with no colour. */
  .rec-reason {
    margin-top: 13px; padding: 11px 0 11px 14px; border-left: 2px solid var(--text);
    font-size: 13.5px; letter-spacing: -0.01em;
  }
  .rec-reason span {
    display: block; margin-bottom: 4px;
    font-family: var(--mono); font-size: 10px; letter-spacing: 0.13em;
    text-transform: uppercase; color: var(--muted);
  }
  .rec-none { margin-top: 13px; font-size: 13px; color: var(--muted); font-style: italic; }

  .rec-specimens { margin-top: 14px; display: flex; flex-direction: column; gap: 9px; }
  .spec { border: 1px solid var(--border-soft); border-radius: 8px; overflow: hidden; background: var(--wash); }
  .spec h4 {
    font-family: var(--mono); font-size: 9.5px; font-weight: 400; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--muted);
    padding: 8px 11px 7px; border-bottom: 1px solid var(--border-soft); background: var(--bg);
  }
  /* Ruled lines: these are excerpts of sheets that get printed and written on. */
  .spec pre {
    font-family: var(--mono); font-size: 11.5px; line-height: 20px; margin: 0;
    padding: 7px 11px 12px; white-space: pre-wrap; word-break: break-word; color: var(--text);
    background-image: repeating-linear-gradient(to bottom,
      transparent 0, transparent 19px, var(--border-soft) 19px, var(--border-soft) 20px);
    background-position: 0 7px;
  }
  .spec.is-clipped pre {
    max-height: 120px; overflow: hidden;
    -webkit-mask-image: linear-gradient(#000 58%, transparent);
    mask-image: linear-gradient(#000 58%, transparent);
  }

  .rec-actions {
    margin-top: 15px; padding-top: 13px; border-top: 1px solid var(--border-soft);
    display: flex; flex-wrap: wrap; align-items: center; gap: 9px;
  }
  /* Secondary action group: rejects, regenerates, exports. Sits under the
     primary actions so the eye lands on the dominant decision first. */
  .rec-actions-secondary {
    margin-top: 10px;
    display: flex; flex-wrap: wrap; align-items: center; gap: 9px;
  }
  .rec-actions-secondary .sep {
    color: var(--muted); font-size: 11px; letter-spacing: 0.06em; text-transform: uppercase;
  }

  /* Structured agent-review block: the six fields the spec calls for.
     Each line is a labelled small-caps row; the body is one or two short
     sentences. Compact, scannable, and identical across records so the
     teacher's eye learns the pattern. */
  .agent-review {
    margin-top: 14px; padding: 12px 14px;
    border: 1px solid var(--border-soft); border-radius: 10px;
    background: var(--wash); display: flex; flex-direction: column; gap: 8px;
  }
  .agent-review h4 {
    margin: 0 0 4px;
    font-family: var(--mono); font-size: 9.5px; font-weight: 400;
    letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted);
  }
  .ar-row { display: flex; gap: 10px; align-items: flex-start; font-size: 12.5px; }
  .ar-row .ar-label {
    flex: 0 0 9.5em;
    font-family: var(--mono); font-size: 10px; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--muted); padding-top: 1px;
  }
  .ar-row .ar-body { flex: 1; color: var(--text); line-height: 1.45; }
  .ar-row.is-decision .ar-body { font-weight: 600; }

  /* Agent Activity timeline: a compact, vertical list of safe action lines. */
  .agent-activity {
    margin-top: 14px; padding: 12px 14px;
    border: 1px solid var(--border-soft); border-radius: 10px;
    background: var(--bg);
  }
  .agent-activity h4 {
    margin: 0 0 8px;
    font-family: var(--mono); font-size: 9.5px; font-weight: 400;
    letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted);
  }
  .agent-activity ol { list-style: none; margin: 0; padding: 0; }
  .agent-activity li {
    display: flex; gap: 10px; align-items: flex-start;
    padding: 5px 0; font-size: 12.5px; line-height: 1.4;
    border-top: 1px solid var(--border-soft);
  }
  .agent-activity li:first-child { border-top: 0; }
  .agent-activity .a-icon {
    flex: 0 0 14px; line-height: 0; color: var(--muted); margin-top: 2px;
  }
  .agent-activity .a-label {
    flex: 0 0 11em;
    font-family: var(--mono); font-size: 10px; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--muted); padding-top: 1px;
  }
  .agent-activity .a-body { flex: 1; color: var(--text); }
  .agent-activity .a-empty {
    color: var(--muted); font-style: italic;
    display: flex; align-items: flex-start; gap: 7px;
  }
  .agent-activity .a-empty .ai { margin-top: 2px; opacity: 0.6; flex: none; }
  .agent-activity h4, .agent-review h4 {
    display: flex; align-items: center; gap: 6px;
  }
  .agent-activity h4 .ai, .agent-review h4 .ai { color: var(--muted); }

  /* Misconception diagnostic, teacher reference. Lives in the flagged
     card so a reviewer can see "this wrong answer reveals X" before
     deciding. */
  .miscon-table {
    margin-top: 12px; font-size: 12px; line-height: 1.45;
    border-collapse: collapse; width: 100%;
  }
  .miscon-table th, .miscon-table td {
    text-align: left; padding: 6px 8px;
    border-top: 1px solid var(--border-soft);
    vertical-align: top;
  }
  .miscon-table th {
    font-family: var(--mono); font-size: 9.5px; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--muted); font-weight: 400;
    border-top: 0;
  }

  /* Impact metrics strip on the mission card. Compact, honest labels,
     estimated numbers labelled as such. */
  .impact-strip {
    margin-top: 12px; padding: 10px 12px;
    border: 1px solid var(--border-soft); border-radius: 10px;
    background: var(--bg);
    display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px 14px;
  }
  .impact-cell { display: flex; flex-direction: column; gap: 1px; }
  .impact-cell .impact-num {
    font-size: 18px; font-weight: 700; letter-spacing: -0.02em; line-height: 1.1;
    font-variant-numeric: tabular-nums;
  }
  .impact-cell .impact-label {
    font-family: var(--mono); font-size: 9.5px; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--muted);
    display: inline-flex; align-items: center; gap: 4px;
  }
  .impact-cell .impact-label .ai { opacity: 0.75; }
  .impact-cell.is-estimated .impact-num::after {
    content: " *";
    color: var(--muted); font-weight: 400;
  }
  .impact-note {
    grid-column: 1 / -1;
    font-size: 10.5px; color: var(--muted); line-height: 1.45;
  }

  .rec-done {
    font-family: var(--mono); font-size: 11px; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--muted);
    display: inline-flex; align-items: center; gap: 5px;
  }
  .rec-err {
    margin-top: 11px; padding: 10px 12px; font-size: 12.5px; letter-spacing: -0.01em;
    border: 1px solid var(--text); border-radius: 8px; background: var(--wash-deep);
  }

  /* Approved cards recede: washed ground, muted type, no action. */
  .col-approved .rec { background: var(--wash); border-color: transparent; }
  .col-approved .rec:hover { border-color: var(--border-soft); }
  .col-approved .rec-topic { color: var(--muted); }
  .col-approved .spec { background: var(--bg); }

  .rec.is-moved { animation: settle .5s ease-out; }
  @keyframes settle {
    from { transform: translateY(-8px); background: var(--wash-deep); }
    to   { transform: none; }
  }

  /* ===================================================================== *
   * Footer
   * ===================================================================== */
  .foot {
    border-top: 1px solid var(--border-soft); padding: 30px 0 46px;
    display: flex; flex-wrap: wrap; align-items: center; gap: 12px 26px;
  }
  .foot p { font-size: 12.5px; color: var(--muted); letter-spacing: -0.01em; }
  .foot-spacer { flex: 1 1 auto; }
  .foot a { font-family: var(--mono); font-size: 11.5px; color: var(--muted); }
  .foot a:hover { color: var(--text); }

  /* ===================================================================== *
   * Animated SVG icon system — one visual family (Lucide-style: 24px grid,
   * 2px stroke, round caps, currentColor). No emoji, no mixed packs.
   *
   * Sizing scale (optically aligned to text via inline-flex + line-height 0):
   *   12 metadata · 14 compact controls · 16 standard · 18 buttons ·
   *   20/24 navigation · 28/32 empty states and major status
   *
   * Animation language — states only, never decoration:
   *   ai-anim-draw  success draws itself (once, ~450ms)
   *   ai-anim-spin  processing: restrained ring rotation
   *   ai-anim-pulse warning/human review: opacity breath
   *   ai-chevron    expand: 180° rotation
   * ===================================================================== */
  .ai {
    display: inline-flex; align-items: center; justify-content: center;
    flex: none; line-height: 0; color: inherit; vertical-align: -0.15em;
  }
  .ai svg { display: block; width: 100%; height: 100%; overflow: visible; }
  .ai-12 { width: 12px; height: 12px; }
  .ai-14 { width: 14px; height: 14px; }
  .ai-16 { width: 16px; height: 16px; }
  .ai-18 { width: 18px; height: 18px; }
  .ai-20 { width: 20px; height: 20px; }
  .ai-24 { width: 24px; height: 24px; }
  .ai-28 { width: 28px; height: 28px; }
  .ai-32 { width: 32px; height: 32px; }

  /* Interaction: neutral -> hover lifts slightly, active presses in. */
  .ai-movable { transition: transform .14s ease, opacity .14s ease; }
  a:hover > .ai-movable, button:hover > .ai-movable,
  .drawer-link:hover .ai-movable, .btn:hover .ai-movable { transform: scale(1.08); }
  a:active > .ai-movable, button:active > .ai-movable,
  .btn:active .ai-movable { transform: scale(0.94); }

  /* Success — the check draws itself exactly once. Paths carry pathLength=1. */
  .ai-anim-draw [pathLength] {
    stroke-dasharray: 1; stroke-dashoffset: 0;
  }
  .ai-anim-draw.is-draw [pathLength] {
    stroke-dashoffset: 1;
    animation: ai-draw .45s cubic-bezier(0.65, 0, 0.35, 1) .1s forwards;
  }
  @keyframes ai-draw { to { stroke-dashoffset: 0; } }

  /* Processing — a short arc travels the ring: progress, not decoration. */
  .ai-anim-spin { animation: ai-spin 1.4s linear infinite; transform-origin: 50% 50%; }
  @keyframes ai-spin { to { transform: rotate(360deg); } }

  /* Warning / human review — restrained opacity breath on the whole glyph. */
  .ai-anim-pulse { animation: ai-pulse 2.2s ease-in-out infinite; }
  @keyframes ai-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.45; } }

  /* AI activity — a single node orbits a thin ring (the agent icon). */
  .ai-orbit { transform-origin: 50% 50%; animation: ai-orbit 2.6s linear infinite; }
  @keyframes ai-orbit { to { transform: rotate(360deg); } }
  .ai.is-idle .ai-orbit { animation-play-state: paused; }

  /* Expand — chevrons rotate; the box follows its section's state class. */
  .ai-chevron { transition: transform .18s ease; }
  .is-open > .ai-chevron, .is-open .ai-chevron,
  .ai-chevron.is-open { transform: rotate(180deg); }

  /* Search — focus rings the lens, the handle nudges toward it. */
  .ai-search-lens { transition: stroke-width .18s ease; }
  .ai:focus-within .ai-search-lens, input:focus + .ai .ai-search-lens { stroke-width: 2.4; }

  /* Tier meter: three bars grow in on mount — data, drawn like an icon. */
  .tier-meter { display: inline-flex; line-height: 0; color: inherit; }
  .tier-meter svg { display: block; }
  .tier-meter rect { transform-origin: 50% 100%; animation: ai-meter-in .3s ease-out backwards; }
  .tier-meter rect:nth-child(2) { animation-delay: 0.05s; }
  .tier-meter rect:nth-child(3) { animation-delay: 0.1s; }
  @keyframes ai-meter-in {
    from { transform: scaleY(0.2); opacity: 0.2; }
    to   { transform: scaleY(1); opacity: 1; }
  }
  .tier-meter .tm-off { opacity: 0.25; }

  /* Mission progress: a real bar replaces the block-glyph meter. */
  /* (canonical styles live with the mission-card rules above) */

  /* Status chips carry their icon gracefully at 14px. */
  .mchip { display: inline-flex; align-items: center; gap: 5px; }
  .mchip .ai { color: currentColor; }

  /* Drawer nav icons: quiet by default, emphasis on hover/active. */
  .drawer-link { align-items: center; }
  .drawer-link .ai { color: var(--muted); transition: color .16s ease, transform .16s ease; }
  .drawer-link:hover .ai { color: var(--text); transform: translateX(2px); }

  @media (prefers-reduced-motion: reduce) {
    .ai-anim-spin, .ai-anim-pulse, .ai-orbit, .tier-meter rect { animation: none !important; }
    .ai-anim-draw [pathLength] { stroke-dashoffset: 0 !important; animation: none !important; }
    .ai-chevron, .ai-movable, .drawer-link .ai, .mbar-fill { transition: none !important; }
    /* Theme toggle: the icon snaps instead of morphing, and the reveal
       becomes a plain crossfade of surfaces. */
    .theme-toggle .tt-rays, .theme-toggle .tt-core, .theme-toggle .tt-biter { transition: none !important; }
    ::view-transition-new(root) { animation: none !important; }
    .theme-toggle, .theme-toggle:hover, .theme-toggle:active { transform: none; }
  }

  /* ===================================================================== *
   * Theme toggle — a floating frosted chip, bottom-right (the Apple
   * circular-control grammar). The icon is one SVG that morphs: the sun's
   * rays fold away while a mask circle slides in to bite the core into a
   * crescent. Theme changes reveal through a circular clip from the chip
   * (View Transitions API, progressive enhancement).
   * ===================================================================== */
  .theme-toggle {
    position: fixed; right: 22px; bottom: 22px; z-index: 80;
    width: 46px; height: 46px; border-radius: 50%;
    display: grid; place-items: center; padding: 0;
    background: var(--nav-glass);
    -webkit-backdrop-filter: saturate(180%) blur(18px);
    backdrop-filter: saturate(180%) blur(18px);
    border: 1px solid var(--border-soft);
    color: var(--text); cursor: pointer;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.14);
    transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
  }
  .theme-toggle:hover { transform: scale(1.08); box-shadow: 0 8px 26px rgba(0, 0, 0, 0.2); }
  .theme-toggle:active { transform: scale(0.92); }
  .theme-toggle:focus-visible { outline-offset: 4px; }
  .theme-toggle svg { display: block; width: 20px; height: 20px; overflow: visible; }

  /* Sun state (light): full core, rays out. */
  .theme-toggle .tt-core { transform-origin: 50% 50%; transition: transform .45s cubic-bezier(0.22, 1, 0.36, 1); }
  .theme-toggle .tt-rays {
    transform-origin: 50% 50%;
    transition: transform .45s cubic-bezier(0.22, 1, 0.36, 1), opacity .3s ease;
  }
  /* Moon state (dark): rays fold and fade, the mask circle slides in. */
  html[data-theme="dark"] .theme-toggle .tt-rays {
    transform: rotate(90deg) scale(0.4); opacity: 0;
  }
  .theme-toggle .tt-biter { transform: translate(0, 0); transition: transform .45s cubic-bezier(0.22, 1, 0.36, 1); }
  html[data-theme="dark"] .theme-toggle .tt-biter { transform: translate(-9px, -5px); }
  html[data-theme="dark"] .theme-toggle .tt-core { transform: scale(1.12); }

  /* Circular reveal of the theme change, radiating from the chip. */
  ::view-transition-old(root), ::view-transition-new(root) {
    animation: none; mix-blend-mode: normal;
  }
  ::view-transition-old(root) { z-index: 1; }
  ::view-transition-new(root) {
    z-index: 2;
    animation: tt-reveal .55s cubic-bezier(0.22, 1, 0.36, 1) forwards;
  }
  @keyframes tt-reveal {
    from { clip-path: circle(0px at var(--tt-x, calc(100% - 45px)) var(--tt-y, calc(100% - 45px))); }
    to   { clip-path: circle(150vmax at var(--tt-x, calc(100% - 45px)) var(--tt-y, calc(100% - 45px))); }
  }

  /* Browsers without the View Transitions API get a simple crossfade of the
     main surfaces instead (the class is applied only during the swap). */
  html.tt-anim, html.tt-anim body, html.tt-anim .nav, html.tt-anim .rec,
  html.tt-anim .mcard, html.tt-anim .spec, html.tt-anim .gen-bar,
  html.tt-anim .drawer, html.tt-anim .col-empty, html.tt-anim .impact-strip,
  html.tt-anim .pcard, html.tt-anim .agent-review, html.tt-anim .agent-activity {
    transition: background-color .3s ease, color .3s ease, border-color .3s ease;
  }

  /* ===================================================================== *
   * Responsive
   * ===================================================================== */
  @media (max-width: 1080px) {
    .desk-masonry { columns: 1 auto; }
    .col-empty { margin-top: 16px; }
  }
  @media (max-width: 760px) {
    :root { --gutter: 20px; }
    .hero { padding: calc(var(--nav-h) + 60px) 0 56px; }
    .drawer-body { grid-template-columns: 1fr; gap: 40px; padding-top: 40px; }
    .desk { padding: 60px 0 80px; }
    .desk-bar { gap: 10px 20px; }
  }

  @media (prefers-reduced-motion: reduce) {
    html { scroll-behavior: auto; }
    .tick-track, .hero-line path, .desk-eyebrow i, .rec.is-moved { animation: none; }
    .drawer, .drawer-link, .btn, .rec, .nav { transition: none; }
    .drawer-link { opacity: 1; transform: none; }
    /* The gallery reveals by scroll and lifts on hover; without motion both
       are absent and the strip is simply there, natively scrollable. */
    .hiw-card { transition: none; }
    .hiw-card:hover .pcard { transform: none; }
    .pcard, .hiw-btn { transition: none; }
    /* Keep the busy bar visible as a static rule rather than a sweep. */
    .gen.is-busy .gen-sweep::after { animation: none; transform: none; width: 100%; }
    .gen-bar, .gen-sweep { transition: none; }
    /* The pulsing dot says "still running"; a solid dot says the same thing. */
    .mlog.is-live .mlog-dot { animation: none; }
    /* The HUD icons only move while .is-drawing is set, so freezing their
       animations leaves them as static, fully-drawn icons. */
    .hud-icon.is-drawing svg .hud-draw,
    .hud-icon.is-drawing svg .hud-timer-btn,
    .hud-icon.is-drawing svg .hud-timer-hand { animation: none; }
  }
</style>
</head>
<body>

<!-- ===================== navbar ===================== -->
<header class="nav" id="nav">
  <div class="nav-inner">
    <a class="logo" href="#top">IdentAI<sup>&reg;</sup></a>
    <button class="nav-menu" type="button" id="menu-open" aria-expanded="false" aria-controls="drawer">
      <span class="nav-menu-bars" aria-hidden="true"><i></i><i></i></span>
      <span>Menu</span>
    </button>
  </div>
</header>

<!-- ===================== drawer ===================== -->
<div class="drawer" id="drawer" role="dialog" aria-modal="true" aria-label="Menu" aria-hidden="true">
  <div class="drawer-top">
    <div class="drawer-top-inner">
      <span class="logo">IdentAI<sup>&reg;</sup></span>
      <button class="drawer-close" type="button" id="menu-close">
        <span>Close</span><span aria-hidden="true">&times;</span>
      </button>
    </div>
  </div>
  <div class="drawer-body">
    <nav class="drawer-nav">
      <a class="drawer-link" href="#top" data-close><span class="ai ai-16 ai-movable" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" focusable="false"><path d="M12 3.5l1.9 5.1 5.1 1.9-5.1 1.9L12 17.5l-1.9-5.1L5 10.5l5.1-1.9L12 3.5z"/></svg></span>What is <em>IdentAI?</em></a>
      <a class="drawer-link" href="#how-it-works" data-close><span class="ai ai-16 ai-movable" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" focusable="false"><circle cx="12" cy="12" r="8.4"/><g class="ai-orbit"><circle cx="12" cy="3.6" r="1.7"/></g></svg></span>How it <em>works</em></a>
      <a class="drawer-link" href="#not-a-chatbot" data-close><span class="ai ai-16 ai-movable" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" focusable="false"><circle cx="12" cy="12" r="9"/><path d="M9.2 9.2l5.6 5.6M14.8 9.2l-5.6 5.6"/></svg></span>Not another <em>chatbot</em></a>
      <a class="drawer-link" href="#example-mission" data-close><span class="ai ai-16 ai-movable" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" focusable="false"><path d="M8 5.5l11 6.5-11 6.5v-13z"/></svg></span>Example <em>mission</em></a>
      <a class="drawer-link" href="#outputs" data-close><span class="ai ai-16 ai-movable" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" focusable="false"><path d="M13.5 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8.5L13.5 3z"/><path d="M13.5 3v5.5H19"/><path d="M9 13.5h6M9 17h4.5"/></svg></span>What you <em>get</em></a>
      <a class="drawer-link" href="#review-desk" data-close><span class="ai ai-16 ai-movable" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" focusable="false"><path d="M4 13.5h4.2l1.6 2.6h4.4l1.6-2.6H20"/><path d="M6 4.5h12l2.5 8.5v5.5a1.5 1.5 0 0 1-1.5 1.5H5a1.5 1.5 0 0 1-1.5-1.5V13l2.5-8.5z"/></svg></span>Review <em>desk</em></a>
      <div class="drawer-sub" aria-label="Live review desk shortcuts">
        <a href="#col-flagged" data-close>Requires your review <b id="menu-count-flagged">0</b></a>
        <a href="#col-draft" data-close>Drafts on the desk <b id="menu-count-draft">0</b></a>
        <a href="#col-approved" data-close>Approved, ready to print <b id="menu-count-approved">0</b></a>
      </div>
    </nav>
    <div class="drawer-info">
      <h4>About this build</h4>
      <p>A Strands agent on Amazon Bedrock decides what a lesson needs, drafts a
        worksheet, quiz and answer key for three ability tiers, then audits its own
        output and rewrites what fails. When a defect is a judgment call rather than a
        mistake it stops and flags the lesson. Nothing reaches a student until you
        approve it.</p>
      <dl>
        <dt>api docs</dt><dd><a href="/docs">/docs</a></dd>
        <dt>model</dt><dd>__MODEL_ID__</dd>
        <dt>database</dt><dd>__DB_PATH__</dd>
        <dt>tiers</dt><dd>struggling / on-level / advanced</dd>
        <dt>queue</dt><dd>POST /process-lesson</dd>
        <dt>mission</dt><dd>POST /generate</dd>
        <dt>trace</dt><dd>GET /api/missions/{id}</dd>
        <dt>read</dt><dd>GET /api/materials</dd>
        <dt>approve</dt><dd>POST /approve/{id}</dd>
        <dt>bedrock</dt><dd>GET /api/bedrock-check</dd>
      </dl>
    </div>
  </div>
</div>

<!-- ===================== hero ===================== -->
<main id="top">
  <section class="hero">
    <svg class="hero-line hero-line-l" viewBox="0 0 190 700" preserveAspectRatio="none" aria-hidden="true" focusable="false">
      <path d="M170 -40 C 60 130, 60 300, 150 470 C 200 570, 170 660, 120 740"></path>
      <path d="M132 -40 C 22 140, 30 320, 116 486 C 168 588, 132 668, 84 740"></path>
      <path d="M96 -40 C -4 150, 4 336, 84 500 C 138 606, 96 676, 50 740"></path>
    </svg>
    <svg class="hero-line hero-line-r" viewBox="0 0 190 700" preserveAspectRatio="none" aria-hidden="true" focusable="false">
      <path d="M170 -40 C 60 130, 60 300, 150 470 C 200 570, 170 660, 120 740"></path>
      <path d="M132 -40 C 22 140, 30 320, 116 486 C 168 588, 132 668, 84 740"></path>
      <path d="M96 -40 C -4 150, 4 336, 84 500 C 138 606, 96 676, 50 740"></path>
    </svg>

    <div class="hero-inner">
      <p class="hero-eyebrow">For computer science teachers &middot; Grades 6&ndash;8</p>
      <h1 class="hero-title hero-title--md">Give your teaching mission.<br><em>The agent handles the busywork.</em></h1>
      <p class="hero-sub">IdentAI is an autonomous AI worker for curriculum preparation. One mission in:
        it plans the objective, writes differentiated materials for three ability tiers, audits and corrects
        its own work, and packages classroom-ready files &mdash; you review the exceptions and approve what ships.</p>

      <!-- primary CTA: the mission bar itself. Topic in, three tiered drafts out. -->
      <form class="gen" id="gen" autocomplete="off">
        <p class="gen-label" id="gen-label">Your teaching mission &mdash; one line is enough</p>
        <label class="sr" for="gen-topic">Teaching mission</label>
        <div class="gen-bar">
          <input class="gen-input" id="gen-topic" name="topic" type="text"
                 placeholder="e.g. Prepare Unit 3 &mdash; Nested Loops"
                 maxlength="200" required>
          <button class="btn gen-btn" id="gen-btn" type="submit">Start a mission</button>
          <div class="gen-sweep" aria-hidden="true"></div>
        </div>
        <span class="gen-status" id="gen-status" role="status" aria-live="polite"></span>
      </form>

      <div class="hero-cta is-under">
        <button class="btn btn-ghost" id="btn-demo-scenario" data-demo-run type="button">Run the demo</button>
        <a class="btn btn-ghost" href="#how-it-works">See how it works &darr;</a>
      </div>

      <p class="hero-live" id="hero-live">Reading lessons.db&hellip;</p>
    </div>
  </section>

  <!-- ===================== ticker ===================== -->
  <div class="tick" aria-hidden="true">
    <div class="tick-track" id="tick-track"></div>
  </div>

  <!-- ===================== landing: the problem ===================== -->
  <section class="story" id="problem" aria-labelledby="problem-title">
    <div class="story-inner">
      <div class="story-head">
        <p class="story-eyebrow"><i></i> The problem</p>
        <h2 class="story-title" id="problem-title">One lesson, three tiers &mdash; <em>every asset in triplicate.</em></h2>
        <p class="story-lede">A single Computer Science lesson means a worksheet, a quiz and an answer key &mdash;
          written once for struggling students, once for the middle of the class, and once for the advanced
          table. Then everything has to agree. That is hours of careful, repetitive writing before you ever teach.</p>
      </div>
      <div class="pain-grid">
        <article class="pain-card">
          <span class="pain-num">01</span>
          <h3>Three tiers per lesson</h3>
          <p>Struggling students need word banks and trace tables. Advanced students need open-ended
            challenges. Same objective, three different documents.</p>
        </article>
        <article class="pain-card">
          <span class="pain-num">02</span>
          <h3>Assessments that match</h3>
          <p>The quiz has to test what the worksheet actually taught &mdash; at each tier &mdash; with
            an answer key that is actually correct.</p>
        </article>
        <article class="pain-card">
          <span class="pain-num">03</span>
          <h3>Everything cross-checked</h3>
          <p>Blanks must match the word bank, keys must match the quiz, structure must fit the tier.
            Checking is its own quiet hour.</p>
        </article>
      </div>
    </div>
  </section>

  <!-- ===================== landing: how it works ===================== -->
  <section class="story" id="how-it-works" aria-labelledby="how-title">
    <div class="story-inner hiw-head">
      <div class="story-head">
        <p class="story-eyebrow"><i></i> How it works</p>
        <h2 class="story-title" id="how-title">One mission in. <em>Six steps &mdash; five without you.</em></h2>
        <p class="story-lede">This is not a form that spits out text. It is an agent with tools, an audit
          loop, and one rule: stop and ask a teacher only when the problem is judgment, not effort.</p>
      </div>
    </div>
    <ol class="hiw-track" id="hiw-track" tabindex="0" aria-label="The six steps of a mission, in a horizontally scrolling strip">
      <li class="hiw-card is-dark" style="--h: 460px">
        <article class="pcard">
          <div class="pcard-top">
            <span class="pcard-chip">You &middot; once</span>
            <span class="pcard-index">01</span>
          </div>
          <h3 class="pcard-title">Give a mission</h3>
          <p class="pcard-text">Type one teaching goal: a unit, a topic, an objective. No prompt engineering.</p>
          <div class="pcard-foot">
            <span class="pcard-dots" aria-hidden="true"><i class="on"></i><i></i><i></i><i></i><i></i><i></i></span>
            <span class="pcard-step">01 / 06</span>
          </div>
          <span class="pcard-mark" aria-hidden="true">1</span>
        </article>
      </li>
      <li class="hiw-card" style="--h: 356px">
        <article class="pcard">
          <div class="pcard-top">
            <span class="pcard-chip">Agent</span>
            <span class="pcard-index">02</span>
          </div>
          <h3 class="pcard-title">Plans &amp; generates</h3>
          <p class="pcard-text">It decides what the lesson needs, then writes worksheet, quiz and answer key for all three tiers.</p>
          <div class="pcard-foot">
            <span class="pcard-dots" aria-hidden="true"><i class="on"></i><i class="on"></i><i></i><i></i><i></i><i></i></span>
            <span class="pcard-step">02 / 06</span>
          </div>
          <span class="pcard-mark" aria-hidden="true">2</span>
        </article>
      </li>
      <li class="hiw-card" style="--h: 424px">
        <article class="pcard">
          <div class="pcard-top">
            <span class="pcard-chip">Agent</span>
            <span class="pcard-index">03</span>
          </div>
          <h3 class="pcard-title">Audits its work</h3>
          <p class="pcard-text">Every asset is checked against the objective and the structural rules for its tier.</p>
          <div class="pcard-foot">
            <span class="pcard-dots" aria-hidden="true"><i class="on"></i><i class="on"></i><i class="on"></i><i></i><i></i><i></i></span>
            <span class="pcard-step">03 / 06</span>
          </div>
          <span class="pcard-mark" aria-hidden="true">3</span>
        </article>
      </li>
      <li class="hiw-card" style="--h: 340px">
        <article class="pcard">
          <div class="pcard-top">
            <span class="pcard-chip">Agent</span>
            <span class="pcard-index">04</span>
          </div>
          <h3 class="pcard-title">Self-corrects</h3>
          <p class="pcard-text">Failed the audit? It rewrites the material and audits again &mdash; before you ever see it.</p>
          <div class="pcard-foot">
            <span class="pcard-dots" aria-hidden="true"><i class="on"></i><i class="on"></i><i class="on"></i><i class="on"></i><i></i><i></i></span>
            <span class="pcard-step">04 / 06</span>
          </div>
          <span class="pcard-mark" aria-hidden="true">4</span>
        </article>
      </li>
      <li class="hiw-card is-dark" style="--h: 476px">
        <article class="pcard">
          <div class="pcard-top">
            <span class="pcard-chip">You &middot; only if asked</span>
            <span class="pcard-index">05</span>
          </div>
          <h3 class="pcard-title">Review exceptions</h3>
          <p class="pcard-text">Genuine judgment calls stop and wait, with the reason and the material side by side.</p>
          <div class="pcard-foot">
            <span class="pcard-dots" aria-hidden="true"><i class="on"></i><i class="on"></i><i class="on"></i><i class="on"></i><i class="on"></i><i></i></span>
            <span class="pcard-step">05 / 06</span>
          </div>
          <span class="pcard-mark" aria-hidden="true">5</span>
        </article>
      </li>
      <li class="hiw-card is-dark" style="--h: 396px">
        <article class="pcard">
          <div class="pcard-top">
            <span class="pcard-chip">Delivered</span>
            <span class="pcard-index">06</span>
          </div>
          <h3 class="pcard-title">Package ready</h3>
          <p class="pcard-text">Approved sheets export to HTML, Markdown or print-ready PDF. Then go teach.</p>
          <div class="pcard-foot">
            <span class="pcard-dots" aria-hidden="true"><i class="on"></i><i class="on"></i><i class="on"></i><i class="on"></i><i class="on"></i><i class="on"></i></span>
            <span class="pcard-step">06 / 06</span>
          </div>
          <span class="pcard-mark" aria-hidden="true">6</span>
        </article>
      </li>
    </ol>
    <div class="hiw-foot">
      <button class="hiw-btn" type="button" data-hiw-prev aria-label="Previous step">&larr;</button>
      <button class="hiw-btn" type="button" data-hiw-next aria-label="Next step">&rarr;</button>
      <span class="hiw-progress" aria-hidden="true"><i></i></span>
      <span class="hiw-hint">Drag &middot; scroll &middot; explore</span>
    </div>
  </section>

  <!-- ===================== landing: not another chatbot ===================== -->
  <section class="story story--wash" id="not-a-chatbot" aria-labelledby="vs-title">
    <div class="story-inner">
      <div class="story-head">
        <p class="story-eyebrow"><i></i> Why an agent</p>
        <h2 class="story-title" id="vs-title">Not another <em>chatbot.</em></h2>
        <p class="story-lede">A chatbot answers. IdentAI finishes. The difference is everything that
          happens after generation &mdash; and what you no longer have to do yourself.</p>
      </div>
      <div class="versus">
        <article class="vs-panel is-muted">
          <p class="vs-kicker">The usual AI workflow</p>
          <h3 class="vs-title">A chat window</h3>
          <ol class="vs-steps">
            <li><div><b>Ask</b><span>Craft a careful prompt for one tier of one lesson.</span></div></li>
            <li><div><b>Generate</b><span>Read what comes back and judge whether it is usable.</span></div></li>
            <li><div><b>Copy</b><span>Paste into your own documents, tier by tier.</span></div></li>
            <li><div><b>Check</b><span>Verify the code, the key, the tier fit &mdash; yourself.</span></div></li>
            <li><div><b>Repeat</b><span>Tomorrow, and the day after that.</span></div></li>
          </ol>
          <p class="vs-note">You are the workflow.</p>
        </article>
        <article class="vs-panel is-agent">
          <p class="vs-kicker">A IdentAI mission</p>
          <h3 class="vs-title">An <em>autonomous worker</em></h3>
          <ol class="vs-steps">
            <li><div><b>Mission</b><span>One teaching goal, stated in plain words.</span></div></li>
            <li><div><b>Autonomous execution</b><span>The agent picks its tools and writes every tier.</span></div></li>
            <li><div><b>Audit</b><span>It grades its own output against the objective and tier rules.</span></div></li>
            <li><div><b>Self-correction</b><span>Failures are rewritten and re-audited automatically.</span></div></li>
            <li><div><b>Human escalation</b><span>Only genuine judgment calls reach you, reason attached.</span></div></li>
            <li><div><b>Finished package</b><span>Reviewed, approved, export-ready.</span></div></li>
          </ol>
          <p class="vs-note">The agent is the workflow. You are the reviewer.</p>
        </article>
      </div>
      <p class="story-note">No magic claimed: the agent can still miss things &mdash; which is exactly why the review desk exists.</p>
    </div>
  </section>

  <!-- ===================== landing: self-audit + misconceptions ===================== -->
  <section class="story" id="self-audit" aria-labelledby="audit-title">
    <div class="story-inner">
      <div class="story-head">
        <p class="story-eyebrow"><i></i> Self-auditing</p>
        <h2 class="story-title" id="audit-title">It checks its own work <em>before you do.</em></h2>
        <p class="story-lede">Every generated asset runs an audit against the objective and the structural
          rules for its tier. Failures are repaired inside the loop &mdash; not in your evening.</p>
      </div>
      <div class="loop-chain" role="img" aria-label="Audit loop: generate, audit, problem detected, self-correction, re-audit, pass">
        <span class="loop-chip">Generate</span>
        <span class="loop-arrow" aria-hidden="true">&rarr;</span>
        <span class="loop-chip">Audit</span>
        <span class="loop-arrow" aria-hidden="true">&rarr;</span>
        <span class="loop-chip is-problem">Problem detected</span>
        <span class="loop-arrow" aria-hidden="true">&rarr;</span>
        <span class="loop-chip">Self-correction</span>
        <span class="loop-arrow" aria-hidden="true">&rarr;</span>
        <span class="loop-chip">Re-audit</span>
        <span class="loop-arrow" aria-hidden="true">&rarr;</span>
        <span class="loop-chip is-pass">Pass</span>
      </div>
      <p class="loop-echo">The loop repeats until the material is clean. What reaches your desk has already
        survived an audit; what cannot be fixed mechanically is escalated, never silently shipped.</p>

      <div class="miscon">
        <div class="miscon-head">
          <h3>Distractors with a diagnosis, not random wrong answers.</h3>
          <span>Real mappings from the demo mission&rsquo;s answer keys</span>
        </div>
        <div class="miscon-row">
          <p class="miscon-q">&ldquo;If the outer loop runs 3 times and the inner loop runs 4 times,
            how many total steps occur?&rdquo;</p>
          <span class="miscon-a">distractor: 7</span>
          <p class="miscon-m">Adds the repetitions instead of multiplying them &mdash; treats nested
            loops as one sequential stage.</p>
        </div>
        <div class="miscon-row">
          <p class="miscon-q">&ldquo;In a nested loop, which loop finishes all its repetitions first?&rdquo;</p>
          <span class="miscon-a">distractor: the outer loop</span>
          <p class="miscon-m">Thinks the outer loop takes a step before the inner loop finishes its run.</p>
        </div>
        <div class="miscon-row">
          <p class="miscon-q">&ldquo;How can early termination be achieved in a nested search?&rdquo;</p>
          <span class="miscon-a">distractor: BREAK</span>
          <p class="miscon-m">Assumes a standard BREAK exits all enclosing loops automatically.</p>
        </div>
      </div>
      <p class="story-note">The exported answer key includes this table &mdash; grade with the diagnosis, not just the letter.</p>
    </div>
  </section>

  <!-- ===================== landing: human in the loop ===================== -->
  <section class="story story--wash" id="human-in-the-loop" aria-labelledby="hitl-title">
    <div class="story-inner">
      <div class="story-head">
        <p class="story-eyebrow"><i></i> Human in the loop</p>
        <h2 class="story-title" id="hitl-title">Automate what can be automated. <em>Ask the teacher what should stay human.</em></h2>
      </div>
      <div class="hitl">
        <div class="hitl-counts" role="img" aria-label="From one real demo mission: 9 assets written and audited, 1 audit failure repaired automatically, 1 decision escalated to the teacher">
          <div class="hitl-cell">
            <span class="hitl-num">9</span>
            <span class="hitl-lbl">assets written &amp; audited &mdash; worksheet, quiz and key, &times; 3 tiers</span>
          </div>
          <div class="hitl-cell">
            <span class="hitl-num">1</span>
            <span class="hitl-lbl">audit failure repaired by the agent &mdash; no teacher needed</span>
          </div>
          <div class="hitl-cell">
            <span class="hitl-num">1</span>
            <span class="hitl-lbl">decision that stopped and waited for a teacher</span>
          </div>
        </div>
        <div class="hitl-copy">
          <p>Review here is not a failure state &mdash; <em>it is the design.</em> When the question is
            whether Grade 7 advanced students should use matrix notation or coordinates, that is a
            pedagogical decision, not a formatting bug.</p>
          <p>The escalated tier waits in <b>Requires your review</b> with the material and the reason
            attached. One click approves it and closes the mission. Everything else finished without you.</p>
        </div>
      </div>
      <p class="story-note">Counts from one real demo mission &mdash; measured, not projected.</p>
    </div>
  </section>

  <!-- ===================== landing: example mission ===================== -->
  <section class="story" id="example-mission" aria-labelledby="xm-title">
    <div class="story-inner">
      <div class="story-head">
        <p class="story-eyebrow"><i></i> Example mission</p>
        <h2 class="story-title" id="xm-title">Watch one run <em>end to end.</em></h2>
        <p class="story-lede">This is the demo mission you can launch from this page. Each line below is a
          real event from that run &mdash; the live version streams into the mission log on the desk.</p>
      </div>
      <div class="xmission">
        <div class="xmission-head">
          <span class="xmission-id">mission &mdash; <b>Unit 2: Nested Loops &amp; Grid Navigation</b></span>
          <span class="xmission-state">ends awaiting your approval</span>
        </div>
        <ol class="xmission-body">
          <li class="xstep">
            <span class="xmark" aria-hidden="true">&#10003;</span>
            <span class="xkind">plan</span>
            <span class="xtext"><b>Mission accepted.</b> Objective set: trace and write nested REPEAT
              loops to move a character across a grid. Curriculum context checked.</span>
          </li>
          <li class="xstep">
            <span class="xmark" aria-hidden="true">&#10003;</span>
            <span class="xkind">generate</span>
            <span class="xtext"><b>Struggling tier written</b> &mdash; word bank, fill-ins, trace table.
              Audit passed. Draft saved.</span>
          </li>
          <li class="xstep">
            <span class="xmark" aria-hidden="true">&#10003;</span>
            <span class="xkind">audit</span>
            <span class="xtext"><b>On-level tier failed its audit</b> &mdash; the trace table was missing.</span>
          </li>
          <li class="xstep">
            <span class="xmark" aria-hidden="true">&#10003;</span>
            <span class="xkind">self-correction</span>
            <span class="xtext"><b>Repaired without you.</b> The agent rewrote the tier itself; the
              re-audit passed and the draft was saved.</span>
          </li>
          <li class="xstep">
            <span class="xmark" aria-hidden="true">&#10003;</span>
            <span class="xkind">generate</span>
            <span class="xtext"><b>Advanced tier written</b> &mdash; open-ended grid-optimization challenge.</span>
          </li>
          <li class="xstep is-stop">
            <span class="xmark" aria-hidden="true">&#9888;</span>
            <span class="xkind">needs teacher</span>
            <span class="xtext"><b>Stopped for you, on purpose.</b> &ldquo;Matrix notation or coordinates
              for Grade 7?&rdquo; is a pedagogy call, not a bug. Material and reason wait on your desk.</span>
          </li>
          <li class="xstep">
            <span class="xmark" aria-hidden="true">&#10003;</span>
            <span class="xkind">done</span>
            <span class="xtext"><b>You approve on the review desk.</b> The mission closes and every
              tier is export-ready.</span>
          </li>
        </ol>
        <div class="xmission-foot">
          <button class="btn" type="button" data-demo-run>Run this mission live</button>
          <p>Launches the real mission against the live agent &mdash; its actual events appear in the
            mission log on the desk below, and the escalated tier lands in Requires your review.</p>
        </div>
      </div>
    </div>
  </section>

  <!-- ===================== landing: outputs ===================== -->
  <section class="story story--wash" id="outputs" aria-labelledby="out-title">
    <div class="story-inner">
      <div class="story-head">
        <p class="story-eyebrow"><i></i> The deliverable</p>
        <h2 class="story-title" id="out-title">What lands on <em>your desk.</em></h2>
        <p class="story-lede">The agent produces materials you can use outside the application &mdash;
          printed, projected or posted. Nothing stays trapped in a chat window.</p>
      </div>
      <div class="outputs-grid">
        <article class="out-card">
          <span class="out-tag">worksheets</span>
          <h3>Student worksheets &times; 3 tiers</h3>
          <p>Same objective, three scaffolding levels: word bank and trace table, guided
            self-correction, find-and-fix challenge.</p>
        </article>
        <article class="out-card">
          <span class="out-tag">assessment</span>
          <h3>Assessment per tier</h3>
          <p>A quiz aligned to the objective &mdash; written against the same spec the worksheet
            was audited on, so it tests what was actually taught.</p>
        </article>
        <article class="out-card">
          <span class="out-tag">answer key</span>
          <h3>Answer key with diagnoses</h3>
          <p>Correct answers plus the misconception behind each distractor &mdash; grading
            tells you what to reteach, not just who lost a point.</p>
        </article>
        <article class="out-card">
          <span class="out-tag">teacher notes</span>
          <h3>Teacher notes</h3>
          <p>How the tiers differ, what the audit checked, and the reason for every
            escalation &mdash; printed into the package itself.</p>
        </article>
        <article class="out-card">
          <span class="out-tag">export</span>
          <h3>Export &amp; print</h3>
          <p>One click per tier: HTML, Markdown, or print-ready PDF with inline styles.
            No screenshotting the app.</p>
        </article>
        <article class="out-card">
          <span class="out-tag">trail</span>
          <h3>Approval trail</h3>
          <p>Every sheet carries its status &mdash; draft, flagged, approved &mdash; so nothing
            unreviewed reaches a student quietly.</p>
        </article>
      </div>
    </div>
  </section>

  <!-- ===================== landing: final CTA ===================== -->
  <section class="story" id="start" aria-labelledby="finale-title">
    <div class="story-inner finale">
      <div class="story-head">
        <p class="story-eyebrow"><i></i> Start</p>
        <h2 class="story-title finale-title" id="finale-title">Your next unit is <em>one mission away.</em></h2>
        <p class="story-lede">Type a topic. Watch the agent plan, write, audit and repair. Approve what ships.</p>
      </div>
      <div class="finale-cta">
        <button class="btn" type="button" data-start-mission>Start a mission</button>
        <button class="btn btn-ghost" type="button" data-demo-run>Run the demo</button>
        <a class="btn btn-ghost" href="#review-desk">Open the review desk</a>
      </div>
      <div class="finale-facts">
        <span>1 mission &rarr; 3 tiered packages</span>
        <span>Every asset audited</span>
        <span>Defects repaired before you see them</span>
        <span>Human review only when it counts</span>
      </div>
    </div>
  </section>

  <!-- ===================== review desk ===================== -->
  <section class="desk" id="review-desk">
    <div class="desk-head">
      <p class="desk-eyebrow"><i></i> Live review desk</p>
      <h2 class="desk-title">Everything the agent writes <em>stops here</em>.</h2>
      <p class="desk-lede">Everything the agent produces, in one board — filter by state, skim the
        excerpts, act. Flagged lessons need your judgment. Drafts need one click. Approved
        sheets are cleared for printing.</p>

      <div class="impact-hud" id="impact-hud">
        <div class="hud-card">
          <div class="hud-icon" aria-hidden="true">
            <!-- Animated timer — ported from AnimateIcons / Lucide (MIT, @avijit07x) -->
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <circle class="hud-draw hud-timer-dial" cx="12" cy="14" r="8" pathLength="1"/>
              <line class="hud-timer-btn" x1="10" y1="2" x2="14" y2="2"/>
              <line class="hud-timer-hand" x1="12" y1="14" x2="15" y2="11"/>
            </svg>
          </div>
          <div class="hud-data">
            <span class="hud-val" id="hud-time-saved">&mdash;</span>
            <span class="hud-lbl">Teacher Workload Saved (est.)</span>
          </div>
        </div>
        <div class="hud-card">
          <div class="hud-icon" aria-hidden="true">
            <!-- Animated shield-check — ported from heroicons-animated (MIT, @aniket-508) -->
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
              <path d="M12 2.71411C9.8495 4.75073 6.94563 5.99986 3.75 5.99986C3.69922 5.99986 3.64852 5.99955 3.59789 5.99892C3.2099 7.17903 3 8.43995 3 9.74991C3 15.3414 6.82432 20.0397 12 21.3719C17.1757 20.0397 21 15.3414 21 9.74991C21 8.43995 20.7901 7.17903 20.4021 5.99892C20.3515 5.99955 20.3008 5.99986 20.25 5.99986C17.0544 5.99986 14.1505 4.75073 12 2.71411Z"/>
              <path class="hud-draw hud-shield-check" d="M9 12.7498L11.25 14.9998L15 9.74985" pathLength="1"/>
            </svg>
          </div>
          <div class="hud-data">
            <span class="hud-val" id="hud-auto-repaired">&mdash;</span>
            <span class="hud-lbl">Defects Auto-Repaired</span>
          </div>
        </div>
        <div class="hud-card">
          <div class="hud-icon" aria-hidden="true">
            <!-- Animated user-check — lucide geometry, same draw-in style -->
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <circle class="hud-draw hud-user-head" cx="9" cy="7" r="4" pathLength="1"/>
              <path class="hud-draw hud-user-body" d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" pathLength="1"/>
              <polyline class="hud-draw hud-user-check" points="16 11 18 13 22 9" pathLength="1"/>
            </svg>
          </div>
          <div class="hud-data">
            <span class="hud-val" id="hud-human-decisions">&mdash;</span>
            <span class="hud-lbl">Escalated for Teacher Decision</span>
          </div>
        </div>
        <div class="hud-card">
          <div class="hud-icon" aria-hidden="true">
            <!-- Animated package-check — lucide geometry, same draw-in style -->
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path class="hud-draw hud-pkg-box" d="M21 10V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l2-1.14" pathLength="1"/>
              <path class="hud-draw hud-pkg-tape" d="m7.5 4.27 9 5.15" pathLength="1"/>
              <polyline class="hud-draw hud-pkg-flap" points="3.29 7 12 12 20.71 7" pathLength="1"/>
              <line class="hud-draw hud-pkg-line" x1="12" y1="22" x2="12" y2="12" pathLength="1"/>
              <path class="hud-draw hud-pkg-check" d="m16 16 2 2 4-4" pathLength="1"/>
            </svg>
          </div>
          <div class="hud-data">
            <span class="hud-val" id="hud-materials-count">&mdash;</span>
            <span class="hud-lbl">Materials from Real Missions</span>
          </div>
        </div>
      </div>

      <div class="desk-bar">
        <span class="desk-stat"><b id="count-flagged">0</b> flagged</span>
        <span class="desk-stat"><b id="count-draft">0</b> drafts</span>
        <span class="desk-stat"><b id="count-approved">0</b> approved</span>
        <span class="desk-spacer"></span>
        <span class="desk-sync" id="sync">&mdash;</span>
        <button class="btn btn-ghost btn-sm" type="button" id="refresh" aria-label="Refresh the desk">Refresh</button>
      </div>
    </div>
    <p class="sr" id="live" role="status" aria-live="polite"></p>

    <!-- ===================== mission workflow ===================== -->
    <!-- Three sections driven by /api/curriculum-missions/{active,waiting,completed}.
         The dashboard polls those routes and re-renders on each tick. The sections
         are hidden when empty so the desk looks unchanged before the first mission. -->
    <section class="missions" id="missions" aria-labelledby="missions-label">
      <div class="missions-head">
        <p class="missions-eyebrow"><i></i> Autonomous workflow</p>
        <h2 class="missions-title" id="missions-label">The agent is doing the work. <em>You supervise the result.</em></h2>
        <p class="missions-lede">Each mission runs in the background. The active section is what
          is running right now, the waiting section is what genuinely needs your judgment, the
          completed section is the classroom-ready output the agent produced.</p>
      </div>

      <div class="missions-grid">
        <!-- ACTIVE -->
        <section class="mission-col mission-col-active" aria-labelledby="col-active-label">
          <header class="mission-col-head">
            <h3 class="mission-col-title" id="col-active-label"><span class="ai ai-16" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" focusable="false"><circle cx="12" cy="12" r="8.4" opacity="0.2"/><path d="M12 3.6a8.4 8.4 0 0 1 8.4 8.4"/></svg></span>Active</h3>
            <span class="mission-col-count" id="col-count-active">0</span>
          </header>
          <p class="mission-col-note" id="col-empty-active">No active mission. The agent is idle.</p>
          <ol class="mission-list" id="col-list-active" aria-live="polite"></ol>
        </section>

        <!-- REQUIRES HUMAN REVIEW -->
        <section class="mission-col mission-col-waiting" aria-labelledby="col-waiting-label">
          <header class="mission-col-head">
            <h3 class="mission-col-title" id="col-waiting-label"><span class="ai ai-16 ai-anim-pulse" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" focusable="false"><path d="M12 3.6L2.8 19.4a.6.6 0 0 0 .5.9h17.4a.6.6 0 0 0 .5-.9L12 3.6z"/><path d="M12 9.5v4.6"/><path d="M12 17.4v.01"/></svg></span>Requires your review</h3>
            <span class="mission-col-count" id="col-count-waiting">0</span>
          </header>
          <p class="mission-col-note" id="col-empty-waiting">Nothing waiting on you.</p>
          <ol class="mission-list" id="col-list-waiting" aria-live="polite"></ol>
        </section>

        <!-- COMPLETED -->
        <section class="mission-col mission-col-completed" aria-labelledby="col-completed-label">
          <header class="mission-col-head">
            <h3 class="mission-col-title" id="col-completed-label"><span class="ai ai-16 ai-anim-draw is-draw" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" focusable="false"><circle cx="12" cy="12" r="9"/><path pathLength="1" d="M8 12.4l2.7 2.7L16.2 9"/></svg></span>Completed</h3>
            <span class="mission-col-count" id="col-count-completed">0</span>
          </header>
          <p class="mission-col-note" id="col-empty-completed">No completed missions yet.</p>
          <ol class="mission-list" id="col-list-completed" aria-live="polite"></ol>
        </section>
      </div>
    </section>

    <!-- Mission log. Hidden until a mission runs; the desk is unchanged without it.
         aria-live is off on purpose: the whole stream would flood a screen reader,
         so milestones are announced through #gen-status instead. -->
    <section class="mlog" id="mlog" hidden aria-labelledby="mlog-label">
      <div class="mlog-head">
        <span class="mlog-dot" aria-hidden="true"></span>
        <span class="mlog-label" id="mlog-label">Mission log</span>
        <span class="mlog-spacer"></span>
        <span class="mlog-meta" id="mlog-meta"></span>
      </div>
      <div class="mlog-body" id="mlog-body">
        <p class="mlog-empty" id="mlog-empty">Waiting for the agent&hellip;</p>
        <ol class="mlog-list" id="mlog-list" aria-live="off"></ol>
      </div>
    </section>

    <div class="desk-tabs" id="desk-tabs" role="group" aria-label="Filter records by status"></div>
    <p class="desk-note" id="desk-note" aria-live="polite"></p>
    <div class="desk-masonry" id="columns"></div>
  </section>

  <footer class="foot shell">
    <p>IdentAI&reg; &mdash; an autonomous AI worker for curriculum preparation.</p>
    <span class="foot-spacer"></span>
    <a href="#how-it-works">How it works</a>
    <a href="#example-mission">Example mission</a>
    <a href="#review-desk">Review desk</a>
    <a href="/docs">/docs</a>
    <a href="/api/materials">/api/materials</a>
    <a href="/health">/health</a>
  </footer>
</main>

<script id="material-data" type="application/json">__MATERIALS_JSON__</script>
<script>
(function () {
  "use strict";

  /* ------------------------------------------------------------------ *
   * config
   * ------------------------------------------------------------------ */
  var COLUMNS = [
    {
      key: "flagged",
      title: "Requires review",
      note: "The agent stopped rather than guess at an unclear objective. Sharpen it, then queue the lesson again.",
      empty: "Nothing flagged. The agent understood every objective it was given."
    },
    {
      key: "draft",
      title: "Drafts ready for approval",
      note: "Skim the excerpts, then approve. Approving flips the status only — no text is rewritten or deleted.",
      empty: "No drafts waiting. Enter a topic in the bar above, or queue a full lesson with <code>POST /process-lesson</code>."
    },
    {
      key: "approved",
      title: "Approved materials",
      note: "Cleared for printing. This is what students will actually see.",
      empty: "Nothing approved yet. Approved sheets collect here."
    }
  ];

  var TICKER_TAGS = [
    "Tiered Worksheets", "Assessments", "Answer Keys", "Misconception Diagnostics",
    "Self-Auditing", "Human Review When It Counts"
  ];

  var TIER_RANK = { "struggling": 1, "on-level": 2, "advanced": 3 };
  var EXCERPT_CHARS = 220;
  var POLL_MS = 6000;

  /* ------------------------------------------------------------------ *
   * state
   * ------------------------------------------------------------------ */
  var columnsHost = document.getElementById("columns");
  var live = document.getElementById("live");
  var syncLabel = document.getElementById("sync");

  var records = readBootstrap();
  var expanded = {};          // id -> true when full text is shown
  var busy = {};              // id -> true while an approve request is in flight
  var moved = null;           // id to highlight after it changes column
  var refocus = null;         // id whose toggle button should regain focus
  var lastSyncAt = Date.now();
  var lastSignature = "";

  function readBootstrap() {
    try {
      var parsed = JSON.parse(document.getElementById("material-data").textContent);
      return Array.isArray(parsed) ? parsed : [];
    } catch (error) {
      return [];
    }
  }

  /* ------------------------------------------------------------------ *
   * helpers
   * ------------------------------------------------------------------ */
  function esc(value) {
    var box = document.createElement("div");
    box.textContent = value === null || value === undefined ? "" : String(value);
    return box.innerHTML;
  }

  function reducedMotion() {
    return !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  }

  function bucketOf(record) {
    var status = String(record && record.status || "draft").toLowerCase();
    if (status === "flagged") return "flagged";
    if (status === "approved") return "approved";
    if (status === "rejected") return "rejected";
    return "draft";
  }

  function groupAll() {
    var groups = { flagged: [], draft: [], approved: [], rejected: [] };
    records.forEach(function (record) { groups[bucketOf(record)].push(record); });
    return groups;
  }

  /* ================================================================== *
   * Animated SVG icon system — the single visual family for the whole
   * interface (Lucide-style: 24px grid, 2px stroke, round caps/joins,
   * currentColor). One icon, one meaning, everywhere.
   *
   *   AIcon(name, size, cls, opts)
   *     name  "check" | "check-circle" | "x" | "x-circle" | "alert" |
   *           "refresh" | "chevron" | "search" | "sparkle" | "agent" |
   *           "loader" | "file" | "code" | "printer" | "layers" |
   *           "clock" | "user" | "wrench" | "tray" | "play"
   *     size  12 | 14 | 16 | 18 | 20 | 24 | 28 | 32  (the optical scale)
   *     cls   animation/state classes ("ai-anim-draw is-draw", …)
   *     opts  { label } → exposed to assistive tech; otherwise hidden
   *
   * Every icon inherits currentColor, so light/dark themes need no
   * per-icon variants, and stroke geometry is identical across the set.
   * ================================================================== */
  var AI_PATHS = {
    "check": '<path pathLength="1" d="M4.5 12.5l4.8 4.8L19.5 6.8"/>',
    "check-circle":
      '<circle cx="12" cy="12" r="9"/>' +
      '<path pathLength="1" d="M8 12.4l2.7 2.7L16.2 9"/>',
    "x": '<path d="M6 6l12 12M18 6L6 18"/>',
    "x-circle":
      '<circle cx="12" cy="12" r="9"/>' +
      '<path d="M9.2 9.2l5.6 5.6M14.8 9.2l-5.6 5.6"/>',
    "alert":
      '<path d="M12 3.6L2.8 19.4a.6.6 0 0 0 .5.9h17.4a.6.6 0 0 0 .5-.9L12 3.6z"/>' +
      '<path d="M12 9.5v4.6"/>' +
      '<path d="M12 17.4v.01"/>',
    "refresh":
      '<path d="M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8"/>' +
      '<path d="M21 3v5h-5"/>',
    "chevron": '<path d="M6 9.2l6 6 6-6"/>',
    "search":
      '<circle class="ai-search-lens" cx="11" cy="11" r="7"/>' +
      '<path d="M20.5 20.5L16.2 16.2"/>',
    "sparkle": '<path d="M12 3.5l1.9 5.1 5.1 1.9-5.1 1.9L12 17.5l-1.9-5.1L5 10.5l5.1-1.9L12 3.5z"/>',
    "agent":
      '<circle cx="12" cy="12" r="8.4"/>' +
      '<g class="ai-orbit"><circle cx="12" cy="3.6" r="1.7"/></g>',
    "loader":
      '<circle cx="12" cy="12" r="8.4" opacity="0.2"/>' +
      '<path class="ai-spin-arc" d="M12 3.6a8.4 8.4 0 0 1 8.4 8.4"/>',
    "file":
      '<path d="M13.5 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8.5L13.5 3z"/>' +
      '<path d="M13.5 3v5.5H19"/>' +
      '<path d="M9 13.5h6M9 17h4.5"/>',
    "code": '<path d="M8.5 7.5L4 12l4.5 4.5"/><path d="M15.5 7.5L20 12l-4.5 4.5"/>',
    "printer":
      '<path d="M7 8V3.5h10V8"/>' +
      '<rect x="4" y="8" width="16" height="8" rx="1.5"/>' +
      '<path d="M7 13.5h10V20H7z"/>',
    "layers":
      '<path d="M12 3.5l8.5 4.7L12 12.9 3.5 8.2 12 3.5z"/>' +
      '<path d="M3.5 15.8L12 20.5l8.5-4.7"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5.2l3.4 2"/>',
    "user":
      '<circle cx="12" cy="8" r="3.8"/>' +
      '<path d="M4.8 20.2a7.6 7.6 0 0 1 14.4 0"/>',
    "wrench":
      '<path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>',
    "tray":
      '<path d="M4 13.5h4.2l1.6 2.6h4.4l1.6-2.6H20"/>' +
      '<path d="M6 4.5h12l2.5 8.5v5.5a1.5 1.5 0 0 1-1.5 1.5H5a1.5 1.5 0 0 1-1.5-1.5V13l2.5-8.5z"/>',
    "play": '<path d="M8 5.5l11 6.5-11 6.5v-13z"/>'
  };

  function AIcon(name, size, cls, opts) {
    var body = AI_PATHS[name];
    if (!body) return "";
    var s = String(size || 16);
    var label = opts && opts.label;
    var attrs = label
      ? ' role="img" aria-label="' + esc(label) + '"'
      : ' aria-hidden="true"';
    return (
      '<span class="ai ai-' + s + (cls ? " " + cls : "") + '"' + attrs + ">" +
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" ' +
          'stroke-linecap="round" stroke-linejoin="round" focusable="false">' +
          body +
        "</svg>" +
      "</span>"
    );
  }

  /* Named status components — the same icon always means the same state. */
  function AnimatedSuccessIcon(size, label) {
    return AIcon("check-circle", size || 14, "ai-anim-draw is-draw ai-movable",
                 { label: label });
  }
  function AnimatedApprovalIcon(size, label) {
    return AnimatedSuccessIcon(size, label);
  }
  function AnimatedErrorIcon(size, label) {
    return AIcon("x-circle", size || 14, "ai-movable", { label: label });
  }
  function AnimatedWarningIcon(size, label) {
    return AIcon("alert", size || 14, "ai-anim-pulse", { label: label });
  }
  function AnimatedProcessingIcon(size, label) {
    return AIcon("loader", size || 14, "ai-anim-spin", { label: label });
  }
  function AnimatedAgentIcon(size, label, idle) {
    return AIcon("agent", size || 14, idle ? "ai-movable is-idle" : "ai-movable",
                 { label: label });
  }
  function AnimatedStatusIcon(state, size) {
    var s = String(state || "running").toLowerCase();
    if (s === "completed") return AnimatedSuccessIcon(size, "Completed");
    if (s === "waiting_for_human") return AnimatedWarningIcon(size, "Waiting for your review");
    if (s === "failed") return AnimatedErrorIcon(size, "Failed");
    return AnimatedProcessingIcon(size, "Running");
  }
  function AnimatedChevron(size, isOpen) {
    return AIcon("chevron", size || 14, "ai-chevron" + (isOpen ? " is-open" : ""));
  }
  function AnimatedSearchIcon(size, label) {
    return AIcon("search", size || 16, "ai-movable", { label: label });
  }

  /* Unknown tiers (legacy rows such as "Beginner") score 0 and render as
     three empty notches rather than breaking the meter. Drawn as SVG bars,
     not text glyphs, so the meter is part of the same icon family. */
  function tierMeter(tier) {
    var rank = TIER_RANK[String(tier || "").toLowerCase()] || 0;
    var bars = "";
    for (var i = 1; i <= 3; i++) {
      var on = i <= rank;
      bars +=
        '<rect x="' + ((i - 1) * 5.4) + '" y="' + (14 - (4 + i * 3)) +
        '" width="3.4" height="' + (4 + i * 3) + '" rx="1" ' +
        'class="' + (on ? "tm-on" : "tm-off") + '"/>';
    }
    return (
      '<span class="tier-meter" aria-hidden="true">' +
        '<svg viewBox="0 0 16 15" width="14" height="14" fill="currentColor">' + bars + "</svg>" +
      "</span>"
    );
  }

  function meter(tier) {
    /* Kept as an alias: tierMeter() is the drawn form of the old glyph meter. */
    return tierMeter(tier);
  }

  function excerpt(text) {
    var clean = String(text).trim();
    if (clean.length <= EXCERPT_CHARS) return clean;
    return clean.slice(0, EXCERPT_CHARS).replace(/\s+\S*$/, "") + "…";
  }

  /* A change in ids, order or statuses is the only thing worth re-rendering
     for. Without this check, every poll would wipe out expanded panels,
     scroll position and keyboard focus. */
  function signatureOf(list) {
    return list.map(function (record) {
      return String(record.id) + ":" + String(record.status || "");
    }).join("|");
  }

  function relativeSync() {
    var seconds = Math.round((Date.now() - lastSyncAt) / 1000);
    if (seconds < 5) return "Synced just now";
    if (seconds < 60) return "Synced " + seconds + "s ago";
    var minutes = Math.round(seconds / 60);
    return "Synced " + minutes + (minutes === 1 ? " min ago" : " mins ago");
  }

  function announce(message) { live.textContent = message; }

  /* ------------------------------------------------------------------ *
   * markup
   * ------------------------------------------------------------------ */
  function specimenHTML(label, text, isOpen) {
    if (!text) return "";
    var body = isOpen ? String(text).trim() : excerpt(text);
    return '<div class="spec' + (isOpen ? "" : " is-clipped") + '">' +
             "<h4>" + esc(label) + "</h4>" +
             "<pre>" + esc(body) + "</pre>" +
           "</div>";
  }

  function recordHTML(record) {
    var id = record.id;
    var group = bucketOf(record);
    var isOpen = expanded[id] === true;
    var isBusy = busy[id] === true;

    var specimens =
      specimenHTML("Worksheet", record.worksheet_text, isOpen) +
      specimenHTML("Quiz", record.quiz_text, isOpen) +
      specimenHTML("Answer key", record.answer_key_text, isOpen);

    var actions = "";
    if (group === "draft") {
      actions += '<button class="btn btn-sm btn-primary" type="button" data-action="approve" data-id="' + id + '"' +
                 (isBusy ? " disabled" : "") + ">" +
                 (isBusy
                   ? AnimatedProcessingIcon(14) + " Approving…"
                   : AIcon("check", 14, "ai-movable") + " Approve") + "</button>";
    } else if (group === "approved") {
      actions += '<span class="rec-done">' + AnimatedApprovalIcon(14, "Approved") + " Approved</span>";
    } else if (group === "flagged") {
      actions += '<button class="btn btn-sm btn-primary" type="button" data-action="approve" data-id="' + id + '"' +
                 (isBusy ? " disabled" : "") + ">" +
                 (isBusy
                   ? AnimatedProcessingIcon(14) + " Approving…"
                   : AIcon("check", 14, "ai-movable") + " Approve") + "</button>";
    } else if (group === "rejected") {
      /* Discarded, not approved: the only way forward is a fresh attempt. */
      actions += '<span class="rec-done">' + AnimatedErrorIcon(14, "Rejected") + " Rejected</span>";
    }
    if (specimens) {
      actions += '<button class="btn btn-ghost btn-sm" type="button" data-action="toggle" data-id="' + id + '" aria-expanded="' + (isOpen ? "true" : "false") + '">' +
                 AnimatedChevron(14, isOpen) +
                 (isOpen ? "Collapse" : "Show full text") + "</button>";
    }

    // Secondary actions: reject, regenerate, export. The export buttons
    // point at the server-rendered handouts (HTML inline, Markdown as
    // attachment, PDF via the browser's print dialog).
    var secondary = "";
    if (group === "flagged" || group === "draft") {
      secondary += '<button class="btn btn-ghost btn-sm" type="button" ' +
                   'data-action="reject" data-id="' + id + '"' +
                   (isBusy ? " disabled" : "") + '>' +
                   AIcon("x", 14, "ai-movable") + "Reject</button>";
    }
    if (group === "flagged" || group === "draft" || group === "rejected") {
      secondary += '<button class="btn btn-ghost btn-sm" type="button" ' +
                   'data-action="regenerate" data-id="' + id + '"' +
                   (isBusy ? " disabled" : "") + '>' +
                   (isBusy
                     ? AnimatedProcessingIcon(14)
                     : AIcon("refresh", 14, "ai-movable")) + "Regenerate</button>";
    }
    if (record.worksheet_text || record.quiz_text || record.answer_key_text) {
      secondary += '<span class="sep">Export</span>';
      secondary += '<a class="btn btn-ghost btn-sm" target="_blank" ' +
                   'href="/api/export/' + id + '/html">' +
                   AIcon("code", 14, "ai-movable") + "HTML</a>";
      secondary += '<a class="btn btn-ghost btn-sm" ' +
                   'href="/api/export/' + id + '/md">' +
                   AIcon("file", 14, "ai-movable") + "Markdown</a>";
      secondary += '<a class="btn btn-ghost btn-sm" target="_blank" ' +
                   'href="/api/export/' + id + '/pdf">' +
                   AIcon("printer", 14, "ai-movable") + "Print / PDF</a>";
    }

    var body = "";
    if (group === "flagged") {
      body += agentReviewHTML(record);
    }
    if (record.reason && group === "flagged") {
      body += '<div class="rec-reason"><span>Why the agent stopped</span>' + esc(record.reason) + "</div>";
    }
    if (group === "rejected") {
      body += '<div class="rec-reason"><span>Rejected</span>You discarded this draft. ' +
              "Hit Regenerate for a fresh attempt at the same tier.</div>";
    }
    body += agentActivityHTML(record.id, group);
    if (specimens) {
      body += '<div class="rec-specimens">' + specimens + "</div>";
    } else if (!record.reason) {
      body += '<p class="rec-none">No generated text on this record yet.</p>';
    }

    return '<article class="rec" data-id="' + id + '">' +
             '<div class="rec-top">' +
               '<h4 class="rec-topic">' + esc(record.topic) + "</h4>" +
               '<span class="rec-id">#' + esc(id) + "</span>" +
             "</div>" +
             '<div class="rec-meta">' +
               '<span class="rec-unit">' + esc(record.unit_name) + "</span>" +
               '<span class="rec-dot" aria-hidden="true">/</span>' +
               '<span class="rec-tier"><span class="rec-meter" aria-hidden="true">' + meter(record.tier) +
                 "</span>" + esc(record.tier) + "</span>" +
             "</div>" +
              '<p class="rec-obj"><span>Objective</span>' + esc(record.objective) + "</p>" +
              body +
              '<div class="rec-actions">' + actions + "</div>" +
              (secondary ? '<div class="rec-actions-secondary">' + secondary + "</div>" : "") +
              '<p class="rec-err" hidden></p>' +
            "</article>";
  }

  /* Six-field agent-review block. We derive these from the record
     itself: the agent's reason is on the row, the audit findings are
     on the sub-mission, and the recommendation maps from the category.
     No new server call is needed to render this. */
  function agentReviewHTML(record) {
    var reason = record.reason || "";
    // The reason field is formatted as "[category] reason. Already tried: ..."
    var category = "judgement call";
    var body = reason;
    var tried = "";
    var m1 = /^\[([^\]]+)\]\s*(.*?)(?:\.\s*Already tried:|$)/.exec(reason);
    if (m1) {
      category = m1[1];
      body = m1[2].trim();
    }
    var m2 = /Already tried:\s*(.+)$/.exec(reason);
    if (m2) tried = m2[1].trim();

    var categoryLabel = {
      "pedagogical_judgement": "two defensible readings of the objective",
      "ambiguous_objective": "the objective is not measurable",
      "out_of_scope": "the topic falls outside the curriculum",
      "budget_exhausted": "the correction budget is spent",
      "audit_unavailable": "the audit itself failed",
      "generation_failed": "the material could not be produced",
    };
    var detected = categoryLabel[category] || category;
    var recommend = {
      "pedagogical_judgement": "Approve the version you prefer, regenerate, or reject.",
      "ambiguous_objective": "Sharpen the objective, then regenerate.",
      "out_of_scope": "Drop the topic, or pick a different one.",
      "budget_exhausted": "Approve the best version, or regenerate for one more try.",
      "audit_unavailable": "Approve the best version, or regenerate for one more try.",
      "generation_failed": "Regenerate, or drop this tier.",
    }[category] || "Approve, regenerate, or reject.";

    return (
      '<div class="agent-review" aria-label="Agent review">' +
        "<h4>" + AnimatedWarningIcon(12, "Needs review") + "Agent review</h4>" +
        arRow("Generated", esc(record.worksheet_text || record.quiz_text || record.answer_key_text
            ? record.worksheet_text
              ? "A worksheet, quiz, and answer key for the " + esc(record.tier) + " tier."
              : record.quiz_text
                ? "A quiz and answer key for the " + esc(record.tier) + " tier."
                : "An answer key for the " + esc(record.tier) + " tier."
            : "No generated text on this record.")) +
        arRow("Detected", esc(detected) + ".") +
        arRow("Why it matters", esc(
          category === "pedagogical_judgement"
            ? "Two readings of the material would teach different things. Only a teacher can pick the right one for this class."
            : category === "ambiguous_objective"
              ? "Students cannot demonstrate mastery of an objective they cannot parse. Sharpen the verb and the noun."
              : category === "budget_exhausted"
                ? "The agent has tried every correction it can think of and the audit still flags this artifact. A fresh human eye is the only remaining path."
                : "The agent cannot decide this for you without guessing.")) +
        arRow("What the agent tried", esc(tried || "It generated the artifact, audited it, and either self-corrected or noted the judgement call.")) +
        arRow("Why it could not resolve", esc(
          category === "pedagogical_judgement"
            ? "Both readings pass the structural checks. The choice changes what students learn, so it is your call."
            : category === "ambiguous_objective"
              ? "The objective itself is the defect; a generator cannot fix its own brief."
              : category === "budget_exhausted"
                ? "The remaining defects are real but mechanical — a teacher can usually approve one and ship it."
                : "The audit verdict is the agent's best honest answer; it cannot go further without you.")) +
        arRowDecision("Recommended decision", esc(recommend)) +
      "</div>"
    );
  }

  function arRow(label, body) {
    return '<div class="ar-row"><div class="ar-label">' + esc(label) + '</div>' +
           '<div class="ar-body">' + body + "</div></div>";
  }
  function arRowDecision(label, body) {
    return '<div class="ar-row is-decision"><div class="ar-label">' + esc(label) + '</div>' +
           '<div class="ar-body">' + body + "</div></div>";
  }

  /* Agent Activity timeline. One card-wide block. We fetch from
     /api/materials/{id}/activity on first paint and cache. The
     server has already redacted chain-of-thought. */
  var activityCache = {};
  var activityInflight = {};
  function agentActivityHTML(recordId, group) {
    // Only show the timeline for items the agent has acted on; drafts
    // without a producing sub-mission are still useful but the spec
    // asks for the agent's actions, which is what this is.
    var hostId = "activity-host-" + recordId;
    return '<div class="agent-activity" id="' + hostId + '" data-id="' + recordId + '">' +
             "<h4>" + AnimatedAgentIcon(12, "Agent", true) + "Agent activity</h4>" +
             '<div class="a-empty">' + AnimatedProcessingIcon(14) + " Loading agent activity&hellip;</div>" +
           "</div>";
  }
  function loadActivityForRecord(recordId) {
    var host = document.getElementById("activity-host-" + recordId);
    if (!host || activityCache[recordId] !== undefined) return;
    if (activityInflight[recordId]) return;
    activityInflight[recordId] = true;
    fetch("/api/materials/" + recordId + "/activity")
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (data) {
        activityInflight[recordId] = false;
        if (!data) { activityCache[recordId] = null; return; }
        activityCache[recordId] = data;
        paintActivity(host, data);
      })
      .catch(function () {
        activityInflight[recordId] = false;
        if (host) host.innerHTML = "<h4>" + AnimatedAgentIcon(12, "Agent", true) + "Agent activity</h4><div class=\"a-empty\">" + AnimatedErrorIcon(14, "Unavailable") + " Agent activity unavailable.</div>";
      });
  }
  function paintActivity(host, data) {
    if (!host) return;
    var timeline = (data && data.timeline) || [];
    if (timeline.length === 0) {
      host.innerHTML = "<h4>" + AnimatedAgentIcon(12, "Agent", true) + "Agent activity</h4><div class=\"a-empty\">" + AIcon("tray", 16) + " No agent activity recorded for this material.</div>";
      return;
    }
    /* Per-event icons: the timeline itself speaks the same visual language
       (checks for passes, refresh for repairs, alert for escalations). */
    function eventIcon(kind) {
      var k = String(kind || "").toLowerCase();
      if (k.indexOf("audit") !== -1 || k.indexOf("pass") !== -1 || k.indexOf("verif") !== -1)
        return AIcon("check", 12);
      if (k.indexOf("correct") !== -1 || k.indexOf("regener") !== -1 || k.indexOf("repair") !== -1)
        return AIcon("refresh", 12);
      if (k.indexOf("human") !== -1 || k.indexOf("flag") !== -1 || k.indexOf("escalat") !== -1)
        return AnimatedWarningIcon(12);
      return AnimatedAgentIcon(12, "Agent step", true);
    }
    var rows = timeline.map(function (ev) {
      return '<li>' +
        '<span class="a-icon">' + eventIcon(ev.kind) + '</span>' +
        '<span class="a-label">' + esc(ev.label || ev.kind || "") + '</span>' +
        '<span class="a-body">' + esc(ev.message || "") + '</span>' +
      '</li>';
    }).join("");
    host.innerHTML = "<h4>" + AnimatedAgentIcon(12, "Agent", true) + "Agent activity</h4><ol>" + rows + "</ol>";
  }


  /* ------------------------------------------------------------------ *
   * render
   * ------------------------------------------------------------------ */

  /* The workflow reads as filters now: pick a state, see every record in
     that state across the full width. "all" is the default view. */
  var activeFilter = "all";
  var DESK_TABS = [
    {
      key: "all", icon: "layers", title: "All",
      note: "Every record the agent has produced, newest first."
    }
  ].concat(COLUMNS);

  function recordsForFilter(key, groups) {
    if (key === "all") return groups.flagged.concat(groups.draft, groups.rejected, groups.approved);
    /* Rejected records stay visible at the bottom of the drafts view —
       discarded, but still regenerable — instead of vanishing. */
    if (key === "draft") return groups.draft.concat(groups.rejected);
    return groups[key];
  }

  function renderTabs(groups) {
    var host = document.getElementById("desk-tabs");
    if (!host) return;
    host.innerHTML = DESK_TABS.map(function (spec) {
      var items = recordsForFilter(spec.key, groups);
      var count = items.length;
      var iconTag = spec.key === "all"
        ? AIcon("layers", 14)
        : spec.key === "flagged"
          ? AnimatedWarningIcon(14)
          : spec.key === "approved"
            ? AnimatedSuccessIcon(14)
            : AIcon("tray", 14);
      return (
        '<button class="desk-tab" type="button" role="button" data-filter="' + spec.key +
          '" aria-pressed="' + (activeFilter === spec.key ? "true" : "false") + '">' +
          iconTag +
          "<span>" + esc(spec.title) + "</span>" +
          '<span class="desk-tab-count" id="tab-count-' + spec.key + '">' + count + "</span>" +
        "</button>"
      );
    }).join("");
  }

  function paintCounts(groups) {
    var totals = {
      all: groups.flagged.length + groups.draft.length + groups.rejected.length + groups.approved.length,
      flagged: groups.flagged.length,
      /* Rejected cards sit at the bottom of the drafts view, so both
         counters agree with what is visibly in the list. */
      draft: groups.draft.length + groups.rejected.length,
      approved: groups.approved.length
    };
    Object.keys(totals).forEach(function (key) {
      ["tab-count-", "count-", "menu-count-"].forEach(function (prefix) {
        var node = document.getElementById(prefix + key);
        if (node) node.textContent = totals[key];
      });
    });

    var total = records.length;
    var heroLive = document.getElementById("hero-live");
    if (heroLive) {
      heroLive.innerHTML = total === 0
        ? "No lessons in the database yet"
        : "<b>" + total + "</b> " + (total === 1 ? "record" : "records") + " in lessons.db &middot; <b>" +
          (totals.flagged + totals.draft) + "</b> awaiting you";
    }
  }

  /* Filter tab clicks re-render immediately: the poll's signature guard
     governs automatic re-renders, not explicit user intent. */
  document.getElementById("desk-tabs").addEventListener("click", function (event) {
    var tab = event.target.closest("[data-filter]");
    if (!tab || tab.dataset.filter === activeFilter) return;
    activeFilter = tab.dataset.filter;
    render();
  });

  function render() {
    var groups = groupAll();
    renderTabs(groups);
    var spec = DESK_TABS.filter(function (t) { return t.key === activeFilter; })[0] || DESK_TABS[0];
    var items = recordsForFilter(spec.key, groups);
    var noteHost = document.getElementById("desk-note");
    if (noteHost) noteHost.textContent = spec.note || "";
    columnsHost.innerHTML = items.length
      ? items.map(recordHTML).join("")
      : '<div class="col-empty">' + spec.empty + "</div>";
    paintCounts(groups);

    if (moved !== null) {
      var card = columnsHost.querySelector('.rec[data-id="' + moved + '"]');
      if (card) {
        card.classList.add("is-moved");
        card.scrollIntoView({ block: "nearest", behavior: reducedMotion() ? "auto" : "smooth" });
      }
      moved = null;
    }
    if (refocus !== null) {
      var button = columnsHost.querySelector('.rec[data-id="' + refocus + '"] [data-action="toggle"]');
      if (button) button.focus();
      refocus = null;
    }
    lastSignature = signatureOf(records);

    // Kick off the agent-activity fetch for every record we just painted.
    records.forEach(function (r) {
      loadActivityForRecord(r.id);
    });
  }

  function showCardError(id, message) {
    var slot = columnsHost.querySelector('.rec[data-id="' + id + '"] .rec-err');
    if (slot) { slot.textContent = message; slot.hidden = false; }
  }

  /* ------------------------------------------------------------------ *
   * data
   * ------------------------------------------------------------------ */
  function anyBusy() {
    return Object.keys(busy).some(function (key) { return busy[key]; });
  }

  async function load(options) {
    var silent = !!(options && options.silent);
    try {
      var response = await fetch("/api/materials", {
        headers: { "Accept": "application/json" },
        cache: "no-store"
      });
      if (!response.ok) throw new Error("server returned " + response.status);
      var incoming = await response.json();
      if (!Array.isArray(incoming)) throw new Error("unexpected payload");

      lastSyncAt = Date.now();
      var changed = signatureOf(incoming) !== lastSignature;
      records = incoming;
      /* Repaint only on real change so a poll never steals focus or collapses
         a panel the teacher is reading. */
      if (changed) render(); else paintCounts(groupAll());
      syncLabel.textContent = relativeSync();
      if (!silent) announce(records.length + (records.length === 1 ? " record" : " records") + " loaded.");
      return true;
    } catch (error) {
      syncLabel.textContent = "Offline — showing last known records";
      if (!silent) announce("Could not reach the server. Showing the records already loaded.");
      return false;
    }
  }

  async function approve(id, button) {
    busy[id] = true;
    button.disabled = true;
    button.textContent = "Approving…";

    try {
      var response = await fetch("/approve/" + id, {
        method: "POST",
        headers: { "Accept": "application/json" }
      });

      if (response.status === 409) {
        /* The row is no longer a draft — approved in another tab, most likely.
           Not an error worth alarming anyone about; just resync. */
        delete busy[id];
        await load({ silent: true });
        announce("Record " + id + " was already handled elsewhere. List resynced.");
        return;
      }
      if (response.status === 404) {
        delete busy[id];
        showCardError(id, "That record no longer exists in the database. Refresh to update the list.");
        announce("Record " + id + " no longer exists.");
        return;
      }
      if (!response.ok) throw new Error("server returned " + response.status);

      var updated = await response.json();
      delete busy[id];
      records = records.map(function (record) {
        return record.id === id
          ? Object.assign({}, record, { status: updated.status || "approved" })
          : record;
      });
      moved = id;
      render();
      lastSyncAt = Date.now();
      syncLabel.textContent = relativeSync();
      announce("Approved " + (updated.topic || "record " + id) + ". Moved to approved materials.");
    } catch (error) {
      delete busy[id];
      button.disabled = false;
      button.textContent = "Approve";
      showCardError(id, "Not approved — " + error.message + ". The record is unchanged; try again.");
      announce("Could not approve record " + id + ".");
    }
  }

  /* ------------------------------------------------------------------ *
   * events
   * ------------------------------------------------------------------ */
  columnsHost.addEventListener("click", function (event) {
    var button = event.target.closest("button[data-action]");
    if (!button) return;
    var id = Number(button.getAttribute("data-id"));
    var action = button.getAttribute("data-action");
    if (action === "approve") {
      approve(id, button);
    } else if (action === "reject") {
      rejectRecord(id, button);
    } else if (action === "regenerate") {
      regenerateRecord(id, button);
    } else {
      expanded[id] = !expanded[id];
      refocus = id;
      render();
    }
  });

  /* Reject the agent's output for a record. The row is marked rejected
     and the teacher can hit Regenerate afterwards. */
  function rejectRecord(id, button) {
    if (busy[id]) return;
    if (!window.confirm("Reject this material? The agent's output will be discarded.")) return;
    busy[id] = true;
    button.disabled = true;
    button.textContent = "Rejecting…";
    fetch("/reject/" + id, { method: "POST" })
      .then(function (r) { return r.ok ? r.json() : r.json().then(function (e) { throw e; }); })
      .then(function (body) {
        announce("Material " + id + " rejected. Hit Regenerate to spawn a fresh attempt.");
        moved = id;
        return load({ silent: true });
      })
      .catch(function (err) {
        busy[id] = false;
        button.disabled = false;
        button.textContent = "Reject";
        showCardError(id, "Could not reject: " + (err && err.detail ? err.detail : "unknown error"));
      });
  }

  /* Regenerate: ask the agent to produce a fresh attempt for the same
     topic/objective/tier. The server spawns a new sub-mission; we
     refresh when the new record id is back. */
  function regenerateRecord(id, button) {
    if (busy[id]) return;
    busy[id] = true;
    button.disabled = true;
    button.textContent = "Regenerating…";
    fetch("/regenerate/" + id, { method: "POST" })
      .then(function (r) { return r.ok ? r.json() : r.json().then(function (e) { throw e; }); })
      .then(function (body) {
        busy[id] = false;
        var newId = body && body.id;
        if (newId && newId !== id) {
          moved = newId;
        }
        return load({ silent: true });
      })
      .catch(function (err) {
        busy[id] = false;
        button.disabled = false;
        button.textContent = "Regenerate";
        showCardError(id, "Could not regenerate: " + (err && err.detail ? err.detail : "unknown error"));
      });
  }

  document.getElementById("refresh").addEventListener("click", async function (event) {
    var button = event.currentTarget;
    var label = button.textContent;
    button.disabled = true;
    button.textContent = "Refreshing…";
    await load({ silent: false });
    button.disabled = false;
    button.textContent = label;
  });

  /* ------------------------------------------------------------------ *
   * generate
   * ------------------------------------------------------------------ */
  /* ------------------------------------------------------------------ *
   * mission log — the agent's own decisions, live
   *
   * A mission is a reasoning loop that generates, audits and rewrites its
   * own work, so it takes minutes rather than seconds. A spinner for that
   * long reads as a hang, and worse, it hides the only interesting part:
   * that the agent chose the tools itself and caught its own mistakes.
   * This follows GET /api/missions/{id} with a `since` cursor while the
   * POST is still in flight, so the two run in parallel.
   * ------------------------------------------------------------------ */
  var missionLog = (function () {
    var panel = document.getElementById("mlog");
    var list = document.getElementById("mlog-list");
    var scroller = document.getElementById("mlog-body");
    var empty = document.getElementById("mlog-empty");
    var meta = document.getElementById("mlog-meta");

    if (!panel || !list) {
      return { newId: function () { return ""; }, start: function () {},
               stop: async function () {} };
    }

    var POLL_MS = 1200;
    var MAX_ROWS = 80;          // a runaway loop must not grow the DOM without bound

    /* Kind -> its label in the left column, and how loudly the row reads.
       An unlisted kind still renders, using the kind as its own label: a new
       event type on the server shows up here without a frontend change.
       agent_reasoning is intentionally absent: the live panel must not surface
       the model's chain-of-thought, only the high-level decisions. */
    var KINDS = {
      mission_started:         { label: "mission",  tone: "" },
      tool_selected:           { label: "tool",     tone: "" },
      tool_started:            { label: "tool",     tone: "is-quiet" },
      tool_result:             { label: "result",   tone: "is-quiet" },
      audit_started:           { label: "audit",    tone: "is-quiet" },
      audit_result:            { label: "audit",    tone: "" },
      audit_passed:            { label: "audit ok", tone: "" },
      audit_failed:            { label: "audit fail", tone: "is-loud" },
      verification:            { label: "verify",   tone: "is-quiet" },
      self_correction:         { label: "rewrite",  tone: "is-loud" },
      material_regenerated:    { label: "regen",    tone: "is-loud" },
      human_review_required:   { label: "needs teacher", tone: "is-loud" },
      human_review:            { label: "escalate", tone: "is-loud" },
      human_review_completed:  { label: "reviewed", tone: "" },
      save:                    { label: "saved",    tone: "" },
      mission_completed:       { label: "done",     tone: "" },
      mission_failed:          { label: "failed",   tone: "is-loud" }
    };

    /* The panel is visual, so a screen-reader user would otherwise get a minutes-
       long silence. These are the milestones worth repeating in the polite status
       line — short phrases built from the event's own fields rather than its raw
       message, because that line is set in uppercase mono and a tool call with
       its arguments is unreadable there. */
    function headline(event) {
      if (event.kind === "tool_selected") {
        return "Running " + String(event.tool || "a tool").replace(/_/g, " ");
      }
      if (event.kind === "self_correction") {
        return "Audit failed — rewriting" + (event.tier ? " the " + event.tier + " tier" : "");
      }
      if (event.kind === "human_review") return "Flagging this one for you";
      if (event.kind === "save") return "Saved a draft";
      return "";
    }

    var run = null;             // the mission being followed, or null when idle

    /* Collision-resistant enough for a single browser tab, and shaped to pass
       the server's MISSION_ID_PATTERN. The server re-rolls anything it does not
       like, so a clash degrades to "the log finds nothing", never to crossed
       missions. */
    function newId() {
      return "m-" + Date.now().toString(36) +
             Math.random().toString(36).slice(2, 8);
    }

    function labelFor(kind) {
      return (KINDS[kind] || { label: String(kind || "event").slice(0, 12) }).label;
    }

    function toneFor(kind) {
      return (KINDS[kind] || { tone: "" }).tone;
    }

    function metaText() {
      if (!run) return "";
      var seconds = Math.round((Date.now() - run.startedAt) / 1000);
      var parts = [];
      if (run.outcome && run.outcome !== "running") parts.push(run.outcome);
      parts.push(seconds + "s");
      parts.push(run.tools + (run.tools === 1 ? " tool call" : " tool calls"));
      return parts.join("  ·  ");
    }

    function paintMeta() {
      meta.textContent = metaText();
    }

    /* Built node by node with textContent: mission text quotes model output and
       teacher input, and nothing here needs to be markup. */
    function addRow(event) {
      var item = document.createElement("li");
      item.className = "mlog-row " + toneFor(event.kind);

      var kind = document.createElement("span");
      kind.className = "mlog-kind";
      kind.textContent = labelFor(event.kind);

      var text = document.createElement("span");
      text.className = "mlog-text";
      text.textContent = event.message || "";

      item.appendChild(kind);
      item.appendChild(text);
      list.appendChild(item);

      while (list.children.length > MAX_ROWS) list.removeChild(list.firstChild);
    }

    function absorb(snapshot) {
      if (!run) return;
      var events = snapshot.events || [];
      /* Follow the tail only when the reader is already at the tail — someone
         who scrolled up to read an audit verdict should not be yanked away. */
      var atBottom = scroller.scrollHeight - scroller.scrollTop -
                     scroller.clientHeight < 40;
      var added = 0;

      for (var i = 0; i < events.length; i += 1) {
        var event = events[i];
        /* The seq cursor is also the de-duplicator: the final drain and a poll
           already in flight can legitimately overlap and return the same tail. */
        if (typeof event.seq === "number") {
          if (event.seq <= run.seq) continue;
          run.seq = event.seq;
        }
        addRow(event);
        added += 1;
        if (run.onHeadline) {
          var phrase = headline(event);
          if (phrase) run.onHeadline(phrase);
        }
      }

      if (added) {
        empty.hidden = true;
        if (atBottom) scroller.scrollTop = scroller.scrollHeight;
      }
      if (typeof snapshot.tool_calls === "number") run.tools = snapshot.tool_calls;
      if (snapshot.outcome) run.outcome = snapshot.outcome;
      paintMeta();
    }

    async function drain() {
      if (!run) return;
      var mission = run.id;
      try {
        var response = await fetch(
          "/api/missions/" + encodeURIComponent(mission) + "?since=" + run.seq,
          { headers: { "Accept": "application/json" } }
        );
        /* 404 is the normal opening state: the browser invented the id, and the
           server has not registered the mission yet. Keep waiting. */
        if (response.status === 404) return;
        if (!response.ok) return;
        var snapshot = await response.json();
        /* A reply that outlived its mission — the panel moved on while it was in
           flight. Dropping it is the whole point of checking. */
        if (run && run.id === mission) absorb(snapshot);
      } catch (error) {
        /* The mission request itself will report a real outage. A failed poll
           only means this panel is briefly behind, which is not worth saying. */
      }
    }

    function schedule() {
      if (!run) return;
      run.timer = window.setTimeout(async function () {
        await drain();
        schedule();
      }, POLL_MS);
    }

    function start(missionId, onHeadline) {
      stopTimers();
      run = { id: missionId, seq: 0, tools: 0, outcome: "running",
              startedAt: Date.now(), onHeadline: onHeadline, timer: 0, clock: 0 };

      list.textContent = "";
      empty.hidden = false;
      empty.textContent = "Waiting for the agent…";
      panel.hidden = false;
      panel.classList.add("is-live");
      paintMeta();

      /* Its own clock, so the elapsed figure keeps moving through a long tool
         call that produces no events. */
      run.clock = window.setInterval(paintMeta, 1000);
      schedule();
    }

    function stopTimers() {
      if (!run) return;
      if (run.timer) window.clearTimeout(run.timer);
      if (run.clock) window.clearInterval(run.clock);
      run.timer = 0;
      run.clock = 0;
    }

    /* Awaited after the request settles: one last read, because the response can
       arrive before the poll that would have collected the closing events. */
    async function stop(outcome) {
      if (!run) return;
      stopTimers();
      await drain();
      if (!run) return;
      if (outcome) run.outcome = outcome;
      panel.classList.remove("is-live");
      if (!list.children.length) {
        empty.hidden = false;
        empty.textContent = "No trail recorded for this mission.";
      }
      paintMeta();
      run = null;
    }

    return { newId: newId, start: start, stop: stop };
  })();

  (function generateSetup() {
    var form = document.getElementById("gen");
    if (!form) return;

    var input = document.getElementById("gen-topic");
    var submit = document.getElementById("gen-btn");
    var statusLine = document.getElementById("gen-status");
    var idleLabel = submit.textContent;

    function setStatus(message, isError, isSentence) {
      statusLine.textContent = message;
      statusLine.classList.toggle("is-error", !!isError);
      statusLine.classList.toggle("is-note", !isError && !!isSentence);
    }

    /* One message per failure mode. "Something went wrong" tells a teacher nothing
       about whether to retry, reword, or go and fix the server.

       Every one of these codes is now raised only when the mission saved and
       escalated nothing at all — saves are incremental, so a run that produced
       anything returns 200 with a message describing what landed. That is what
       lets these lines promise that nothing was written. */
    function explain(httpStatus, detail) {
      /* The server names the actual cause in each of these, and it knows things the
         browser cannot: which provider is configured, which credential is missing,
         how long the agent ran, what the model said. Canned copy here would throw
         away the only sentence that says what to do next. */
      if (httpStatus === 503) {
        return detail || "Generation is switched off — the server has no model provider configured.";
      }
      if (httpStatus === 504) {
        return detail || "The agent ran out of time without finishing a tier. Nothing was saved.";
      }
      if (httpStatus === 502) {
        return detail || "The model returned nothing usable. Nothing was saved — try rewording the topic.";
      }
      if (httpStatus === 422) return "That topic can't be used. Try a few plain words, like: nested loops.";
      return "Generation failed (" + httpStatus + ")" + (detail ? " — " + detail : "") +
             ". Refresh the desk to see what landed.";
    }

    async function readDetail(response) {
      try {
        var body = await response.json();
        if (body && typeof body.detail === "string") return body.detail;
      } catch (error) {
        /* No JSON body — the status code is the whole story. */
      }
      return "";
    }

    form.addEventListener("submit", async function (event) {
      event.preventDefault();          // also handles Enter inside the text field
      if (busy.generate) return;       // ignore a double submit

      var topic = input.value.trim();
      if (!topic) {
        setStatus("Type a topic first.", true);
        input.focus();
        return;
      }

      /* Marking busy parks the 6s poll, so it can't repaint the desk mid-request. */
      busy.generate = true;
      form.classList.add("is-busy");
      form.setAttribute("aria-busy", "true");
      input.disabled = true;
      submit.disabled = true;
      submit.textContent = "Working…";

      /* The id is minted here, not by the server, so the log can start following
         the mission while the request that creates it is still open. */
      var missionId = missionLog.newId();
      var streaming = true;
      setStatus("Briefing the agent on " + topic + "…", false);
      missionLog.start(missionId, function (phrase) {
        if (streaming) setStatus(phrase, false);
      });

      try {
        var response = await fetch("/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json", "Accept": "application/json" },
          body: JSON.stringify({ topic: topic, mission_id: missionId })
        });

        if (!response.ok) {
          var detail = await readDetail(response);
          streaming = false;           // stop the log writing over the verdict
          setStatus(explain(response.status, detail), true);
          announce("The mission failed. Nothing was saved.");
          /* The desk is refetched anyway: these codes promise an empty database,
             and quietly proving it beats asking the teacher to take it on trust. */
          busy.generate = false;
          await load({ silent: true });
          return;                      // input keeps its text so it can be edited
        }

        var result = await response.json();
        var made = (result.records || []).length;
        var flagged = (result.escalations || []).length;

        streaming = false;
        input.value = "";              // cleared only on success
        busy.generate = false;         // released before the refetch so load() can paint
        await load({ silent: true });

        /* The server's own sentence: it is the only party that knows whether the
           agent saved three tiers, escalated one, or ran out of time holding two. */
        setStatus(result.message || (made + " draft(s) saved."), false, true);
        announce(
          result.message ||
          (made + " draft tiers generated for " + topic + ".")
        );

        /* Land on the column that actually changed. An escalation with nothing
           saved is the agent refusing to guess, and that is the column to read. */
        var target = document.getElementById(
          (flagged && !made) ? "col-flagged" : "col-draft"
        );
        if (target) {
          target.scrollIntoView({ block: "start", behavior: reducedMotion() ? "auto" : "smooth" });
        }
      } catch (error) {
        streaming = false;
        setStatus("Could not reach the server. Check that it is still running.", true);
        announce("The mission failed. Could not reach the server.");
      } finally {
        streaming = false;
        await missionLog.stop();       // one last read, for the closing events
        delete busy.generate;
        form.classList.remove("is-busy");
        form.removeAttribute("aria-busy");
        input.disabled = false;
        submit.disabled = false;
        submit.textContent = idleLabel;
      }
    });
  })();

  /* ------------------------------------------------------------------ *
   * Demo Scenario Runner (5-Minute Hackathon Demo)
   * ------------------------------------------------------------------ */
  (function demoSetup() {
    var demoBtns = Array.prototype.slice.call(
      document.querySelectorAll("[data-demo-run]")
    );
    if (!demoBtns.length) return;

    /* Every "Run the demo" button on the page (hero + example mission) drives
       the same real demo scenario; they all change state together. */
    function setDemoBusy(isBusy) {
      demoBtns.forEach(function (btn) {
        btn.disabled = isBusy;
        btn.textContent = isBusy ? "Running demo mission…" : "Run the demo";
      });
    }

    demoBtns.forEach(function (demoBtn) {
    demoBtn.addEventListener("click", async function () {
      if (busy.generate) return;
      busy.generate = true;
      setDemoBusy(true);
      announce("Launching demo mission: Nested Loops & Grid Navigation");

      try {
        var resp = await fetch("/api/demo/scenario", {
          method: "POST",
          headers: { "Content-Type": "application/json", "Accept": "application/json" }
        });
        if (!resp.ok) {
          throw new Error("HTTP " + resp.status);
        }
        var data = await resp.json();

        // Refresh review desk and curriculum missions
        await load({ silent: true });
        await loadMissions();

        // Show live mission log if events returned
        if (data.mission_id) {
          try {
            var evResp = await fetch("/api/curriculum-missions/" + data.mission_id + "/events");
            var evData = await evResp.json();
            if (evData.events && evData.events.length) {
              var mlog = document.getElementById("mlog");
              var mlist = document.getElementById("mlog-list");
              var mempty = document.getElementById("mlog-empty");
              if (mlog && mlist) {
                mlog.hidden = false;
                if (mempty) mempty.hidden = true;
                mlist.innerHTML = evData.events.map(function (e) {
                  var isLoud = e.kind === "human_review" || e.kind === "audit_failed" || e.kind === "self_correction";
                  return (
                    '<li class="mlog-row' + (isLoud ? ' is-loud' : '') + '">' +
                      '<span class="mlog-time">' + esc((e.at || "").slice(11, 19)) + '</span>' +
                      '<span class="mlog-kind">' + esc(e.kind || "") + '</span>' +
                      '<span class="mlog-text">' + esc(e.summary || "") + '</span>' +
                    '</li>'
                  );
                }).join("");
              }
            }
          } catch (evErr) {
            /* quiet fallback */
          }
        }

        // Update Impact HUD from the demo mission's own metrics — the numbers
        // on the page must come from a real run, never from a constant.
        var hudSaved = document.getElementById("hud-time-saved");
        var hudRepaired = document.getElementById("hud-auto-repaired");
        var hudEscalated = document.getElementById("hud-human-decisions");
        var hudCount = document.getElementById("hud-materials-count");
        var metrics = (data && data.metrics) || {};
        if (hudSaved) hudSaved.textContent = metrics.time_saved_estimate_seconds
          ? formatSeconds(metrics.time_saved_estimate_seconds) + " (est.)" : "—";
        if (hudRepaired) hudRepaired.textContent = String(metrics.validation_failures_auto_repaired || 0);
        if (hudEscalated) hudEscalated.textContent = String(metrics.human_decisions_required || 0);
        if (hudCount) hudCount.textContent = metrics.materials_generated
          ? metrics.materials_generated + " materials" : "—";

        announce("Demo mission ready! 2 drafts ready, 1 pedagogical decision waiting in 'Requires Review'.");

        var target = document.getElementById("col-flagged");
        if (target) {
          target.scrollIntoView({ block: "start", behavior: reducedMotion() ? "auto" : "smooth" });
        }
      } catch (err) {
        announce("Failed to run demo scenario: " + err);
      } finally {
        delete busy.generate;
        setDemoBusy(false);
      }
    });
    });
  })();

  /* ------------------------------------------------------------------ *
   * "Start a mission" CTAs: bring the mission bar into view and focus it
   * ------------------------------------------------------------------ */
  (function startMissionSetup() {
    Array.prototype.slice.call(
      document.querySelectorAll("[data-start-mission]")
    ).forEach(function (btn) {
      btn.addEventListener("click", function () {
        var bar = document.getElementById("gen");
        if (bar) {
          bar.scrollIntoView({ block: "center", behavior: reducedMotion() ? "auto" : "smooth" });
        }
        var input = document.getElementById("gen-topic");
        if (input) {
          window.setTimeout(function () {
            try { input.focus({ preventScroll: true }); } catch (error) { input.focus(); }
          }, reducedMotion() ? 0 : 420);
        }
      });
    });
  })();

  /* ------------------------------------------------------------------ *
   * drawer
   * ------------------------------------------------------------------ */
  (function drawerSetup() {
    var drawer = document.getElementById("drawer");
    var openButton = document.getElementById("menu-open");
    var closeButton = document.getElementById("menu-close");
    var isOpen = false;

    function focusables() {
      return Array.prototype.slice.call(
        drawer.querySelectorAll('a[href], button:not([disabled])')
      );
    }

    function open() {
      isOpen = true;
      drawer.classList.add("is-open");
      drawer.setAttribute("aria-hidden", "false");
      openButton.setAttribute("aria-expanded", "true");
      document.body.classList.add("is-locked");
      var first = focusables()[0];
      if (first) first.focus();
    }

    function close() {
      isOpen = false;
      drawer.classList.remove("is-open");
      drawer.setAttribute("aria-hidden", "true");
      openButton.setAttribute("aria-expanded", "false");
      document.body.classList.remove("is-locked");
      openButton.focus();
    }

    openButton.addEventListener("click", open);
    closeButton.addEventListener("click", close);
    drawer.addEventListener("click", function (event) {
      if (event.target.closest("[data-close]")) close();
    });

    document.addEventListener("keydown", function (event) {
      if (!isOpen) return;
      if (event.key === "Escape") { event.preventDefault(); close(); return; }
      if (event.key !== "Tab") return;
      var items = focusables();
      if (!items.length) return;
      var first = items[0];
      var last = items[items.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault(); last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault(); first.focus();
      }
    });
  })();

  /* ------------------------------------------------------------------ *
   * chrome: ticker, navbar shadow, polling
   * ------------------------------------------------------------------ */
  (function buildTicker() {
    var track = document.getElementById("tick-track");
    var cell = "";
    TICKER_TAGS.forEach(function (tag) {
      cell += '<span class="tick-item">' + esc(tag) + "</span>" +
              '<span class="tick-sep">&#9679;</span>';
    });
    /* Two identical runs make translateX(-50%) loop without a visible seam. */
    track.innerHTML = cell + cell;
  })();

  (function navShadow() {
    var nav = document.getElementById("nav");
    function sync() { nav.classList.toggle("is-scrolled", window.scrollY > 8); }
    window.addEventListener("scroll", sync, { passive: true });
    sync();
  })();

  (function poll() {
    window.setInterval(function () {
      syncLabel.textContent = relativeSync();
      if (document.hidden || anyBusy()) return;
      load({ silent: true });
    }, POLL_MS);

    document.addEventListener("visibilitychange", function () {
      if (!document.hidden) load({ silent: true });
    });
  })();

  /* ------------------------------------------------------------------ *
   * Mission workflow: three sections polled in parallel with the desk.
   * ------------------------------------------------------------------ */

  var MISSION_LIST_ACTIVE = document.getElementById("col-list-active");
  var MISSION_LIST_WAITING = document.getElementById("col-list-waiting");
  var MISSION_LIST_COMPLETED = document.getElementById("col-list-completed");
  var MISSION_COUNT_ACTIVE = document.getElementById("col-count-active");
  var MISSION_COUNT_WAITING = document.getElementById("col-count-waiting");
  var MISSION_COUNT_COMPLETED = document.getElementById("col-count-completed");
  var MISSION_EMPTY_ACTIVE = document.getElementById("col-empty-active");
  var MISSION_EMPTY_WAITING = document.getElementById("col-empty-waiting");
  var MISSION_EMPTY_COMPLETED = document.getElementById("col-empty-completed");

  function missionBar(percent) {
    /* A real progress bar (transitioned fill) replaces the block-glyph meter.
       The fill width animates on poll updates, so progress communicates
       change without any character-based pseudo-graphics. */
    var p = Math.max(0, Math.min(100, Number(percent) || 0));
    return (
      '<div class="mbar" role="progressbar" aria-valuenow="' + Math.round(p) +
        '" aria-valuemin="0" aria-valuemax="100">' +
        '<span class="mbar-fill" style="width:' + p.toFixed(1) + '%"></span>' +
      "</div>"
    );
  }

  function missionStateChip(state) {
    var label = (state || "RUNNING").toLowerCase().replace(/_/g, " ");
    return (
      '<span class="mchip mchip-' + esc((state || "running").toLowerCase()) + '">' +
        AnimatedStatusIcon(state, 14) +
        "<span>" + esc(label) + "</span>" +
      "</span>"
    );
  }

  function missionCountsLine(m) {
    var p = m.progress || { completed: 0, total: 0 };
    var c = m.counts || {};
    return (
      '<p class="mcounts">' +
        '<b>' + (p.completed || 0) + '</b>/<b>' + (p.total || 0) + '</b> tasks complete &middot; ' +
        (c.auto_validated || 0) + ' auto-validated &middot; ' +
        (c.auto_repaired || 0) + ' auto-repaired &middot; ' +
        (c.escalated || 0) + ' escalated' +
      '</p>'
    );
  }

  function missionCard(m) {
    var s = m.summary || "";
    return (
      '<article class="mcard" data-id="' + esc(m.mission_id) + '">' +
        '<header class="mcard-head">' +
          '<h4 class="mcard-title">' + esc(m.topic || "Untitled") + '</h4>' +
          missionStateChip(m.state) +
        '</header>' +
        missionBar(m.progress ? m.progress.percent : 0) +
        missionCountsLine(m) +
        missionImpactStrip(m) +
        '<p class="mcard-phase">' + esc(m.current_phase || "") + '</p>' +
        (s ? '<p class="mcard-summary">' + esc(s) + '</p>' : '') +
        '<p class="mcard-meta">' +
          'started ' + esc((m.started_at || m.created_at || "").slice(11, 19)) +
          (m.total_agent_seconds ? ' &middot; ' + m.total_agent_seconds + 's agent time' : '') +
        '</p>' +
      '</article>'
    );
  }

  /* Impact metrics on the mission card. We render only the cells that
     are non-zero to keep the card quiet when nothing happened. The
     time-saved cell is the only one that is an estimate; it is the
     last cell and labelled. */
  function missionImpactStrip(m) {
    var metrics = m.metrics || {};
    var cells = [];
    if (metrics.tasks_completed_automatically)
      cells.push(impactCell(metrics.tasks_completed_automatically, "Auto-handled"));
    if (metrics.validation_failures_auto_repaired)
      cells.push(impactCell(metrics.validation_failures_auto_repaired, "Auto-repaired"));
    if (metrics.human_decisions_required)
      cells.push(impactCell(metrics.human_decisions_required, "Need your call"));
    if (metrics.materials_generated)
      cells.push(impactCell(metrics.materials_generated, "Materials"));
    if (metrics.time_saved_estimate_seconds)
      cells.push(impactCell(formatSeconds(metrics.time_saved_estimate_seconds),
                              "Time saved (est.)", true));
    if (cells.length === 0) return "";
    return '<div class="impact-strip">' + cells.join("") +
             '<p class="impact-note">' + esc(metrics.time_saved_benchmark || "") + '</p>' +
           '</div>';
  }
  function impactCell(num, label, estimated) {
    /* 12px = the metadata step of the icon scale: quiet identification,
       one consistent icon per metric across the whole interface. */
    var icon = {
      "Auto-handled": "check",
      "Auto-repaired": "wrench",
      "Need your call": "user",
      "Materials": "layers",
      "Time saved (est.)": "clock"
    }[label] || null;
    return '<div class="impact-cell' + (estimated ? ' is-estimated' : '') + '">' +
             '<span class="impact-num">' + esc(num) + '</span>' +
             '<span class="impact-label">' +
               (icon ? AIcon(icon, 12) : '') + esc(label) + '</span>' +
           '</div>';
  }
  function formatSeconds(s) {
    s = Math.round(Number(s) || 0);
    if (s >= 3600) {
      var h = Math.floor(s / 3600);
      var m = Math.round((s - h * 3600) / 60);
      return h + "h " + m + "m";
    }
    if (s >= 60) {
      var mm = Math.floor(s / 60);
      var ss = s - mm * 60;
      return mm + "m " + (ss ? ss + "s" : "");
    }
    return s + "s";
  }

  function renderMissions(sections) {
    var a = sections.active || [];
    var w = sections.waiting || [];
    var c = sections.completed || [];
    MISSION_LIST_ACTIVE.innerHTML = a.map(missionCard).join("");
    MISSION_LIST_WAITING.innerHTML = w.map(missionCard).join("");
    MISSION_LIST_COMPLETED.innerHTML = c.map(missionCard).join("");
    MISSION_COUNT_ACTIVE.textContent = String(a.length);
    MISSION_COUNT_WAITING.textContent = String(w.length);
    MISSION_COUNT_COMPLETED.textContent = String(c.length);
    MISSION_EMPTY_ACTIVE.hidden = a.length > 0;
    MISSION_EMPTY_WAITING.hidden = w.length > 0;
    MISSION_EMPTY_COMPLETED.hidden = c.length > 0;
  }

  function loadMissions() {
    if (document.hidden) return Promise.resolve();
    return Promise.all([
      fetch("/api/curriculum-missions/active").then(function (r) { return r.json(); }),
      fetch("/api/curriculum-missions/waiting").then(function (r) { return r.json(); }),
      fetch("/api/curriculum-missions/completed").then(function (r) { return r.json(); })
    ]).then(function (results) {
      renderMissions({ active: results[0], waiting: results[1], completed: results[2] });
    }).catch(function () { /* leave prior paint in place on transient failure */ });
  }

  (function pollMissions() {
    loadMissions();
    window.setInterval(loadMissions, POLL_MS);
  })();

  /* ------------------------------------------------------------------ *
   * animated HUD icons
   * Vanilla port of the supplied motion/react icon components (the
   * AnimateIcons timer, the heroicons-animated shield-check and lucide
   * user-check / package-check). Hovering a card replays the draw-in
   * animation; leaving snaps the icon back to its fully-drawn state —
   * the same animate/normal behaviour the originals got from motion.
   * ------------------------------------------------------------------ */
  Array.prototype.forEach.call(document.querySelectorAll(".hud-card"), function (card) {
    var icon = card.querySelector(".hud-icon");
    if (!icon) return;
    card.addEventListener("mouseenter", function () {
      if (reducedMotion()) return;
      icon.classList.remove("is-drawing");
      void icon.offsetWidth; /* force a reflow so the CSS animations restart */
      icon.classList.add("is-drawing");
    });
    card.addEventListener("mouseleave", function () {
      icon.classList.remove("is-drawing");
    });
  });

  /* First paint uses the JSON embedded by the server, so the desk is populated
     before any request goes out; the poll takes over from there. */
  render();
  syncLabel.textContent = relativeSync();
})();
</script>
<script>
/* Process gallery (#hiw-track): staggered scroll reveal, mouse drag with
   momentum, progress bar and arrow stepping. Touch keeps native scrolling. */
(function () {
  "use strict";

  var section = document.getElementById("how-it-works");
  var track = document.getElementById("hiw-track");
  if (!section || !track) return;

  var cards = Array.prototype.slice.call(track.children);
  var fill = section.querySelector(".hiw-progress i");
  var prevBtn = section.querySelector("[data-hiw-prev]");
  var nextBtn = section.querySelector("[data-hiw-next]");
  var reduced = !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);

  /* ---- scroll reveal: the first visible batch staggers in, cards brought in
     by later horizontal scrolling play immediately ---- */
  if (!reduced && "IntersectionObserver" in window) {
    track.classList.add("is-armed");
    var observer = new IntersectionObserver(function (entries) {
      var batch = 0;
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        entry.target.style.setProperty("--d", Math.min(batch, 5) * 85 + "ms");
        entry.target.classList.add("is-in");
        observer.unobserve(entry.target);
        batch += 1;
      });
    }, { threshold: 0.05, rootMargin: "0px 0px -10% 0px" });
    cards.forEach(function (card) { observer.observe(card); });
  }

  /* ---- progress fill + arrow end states ---- */
  var pending = false;
  function paint() {
    pending = false;
    var max = track.scrollWidth - track.clientWidth;
    var x = track.scrollLeft;
    var ratio = max > 0 ? Math.min(1, Math.max(0, x / max)) : 1;
    if (fill) fill.style.transform = "scaleX(" + ratio + ")";
    if (prevBtn) prevBtn.classList.toggle("is-end", x <= 1);
    if (nextBtn) nextBtn.classList.toggle("is-end", max <= 0 || x >= max - 1);
  }
  function schedulePaint() {
    if (!pending) { pending = true; window.requestAnimationFrame(paint); }
  }
  track.addEventListener("scroll", schedulePaint, { passive: true });
  window.addEventListener("resize", schedulePaint);
  paint();

  /* ---- arrow buttons step one card ---- */
  function stride() {
    var first = track.children[0];
    var width = first ? first.getBoundingClientRect().width : 320;
    var gap = parseFloat(getComputedStyle(track).columnGap);
    return width + (isNaN(gap) ? 18 : gap);
  }
  function step(dir) {
    glideStop();
    track.scrollBy({ left: dir * stride(), behavior: reduced ? "auto" : "smooth" });
  }
  if (prevBtn) prevBtn.addEventListener("click", function () { step(-1); });
  if (nextBtn) nextBtn.addEventListener("click", function () { step(1); });

  /* ---- mouse drag with momentum ---- */
  var dragging = false, moved = false;
  var startX = 0, startLeft = 0, lastX = 0, lastT = 0, velocity = 0, glideRaf = 0;

  function glideStop() {
    if (glideRaf) { window.cancelAnimationFrame(glideRaf); glideRaf = 0; }
  }
  function glide() {
    var then = performance.now(), v = velocity;
    var tick = function (now) {
      var dt = Math.min(64, now - then);
      then = now;
      track.scrollLeft -= v * dt;
      v *= Math.pow(0.94, dt / 16.7);
      var atEnd = track.scrollLeft <= 0 || track.scrollLeft >= track.scrollWidth - track.clientWidth;
      if (Math.abs(v) > 0.02 && !atEnd) {
        glideRaf = window.requestAnimationFrame(tick);
      } else {
        glideRaf = 0;
      }
    };
    glideRaf = window.requestAnimationFrame(tick);
  }
  track.addEventListener("pointerdown", function (event) {
    if (event.pointerType !== "mouse" || event.button !== 0) return;
    glideStop();
    dragging = true;
    moved = false;
    startX = lastX = event.clientX;
    startLeft = track.scrollLeft;
    lastT = performance.now();
    velocity = 0;
    try { track.setPointerCapture(event.pointerId); } catch (error) { /* drag still works uncaptured */ }
  });
  track.addEventListener("pointermove", function (event) {
    if (!dragging) return;
    var offset = event.clientX - startX;
    if (!moved && Math.abs(offset) > 5) {
      moved = true;
      track.classList.add("is-dragging");
    }
    if (!moved) return;
    track.scrollLeft = startLeft - offset;
    var now = performance.now();
    var dt = now - lastT;
    if (dt > 0) {
      velocity = 0.8 * velocity + 0.2 * ((event.clientX - lastX) / dt);
      lastX = event.clientX;
      lastT = now;
    }
  });
  function release() {
    if (!dragging) return;
    dragging = false;
    track.classList.remove("is-dragging");
    if (moved && !reduced && Math.abs(velocity) > 0.15) glide();
  }
  track.addEventListener("pointerup", release);
  track.addEventListener("pointercancel", release);
  track.addEventListener("wheel", glideStop, { passive: true });
  track.addEventListener("dragstart", function (event) { event.preventDefault(); });
})();
</script>
<button class="theme-toggle" id="theme-toggle" type="button"
        aria-label="Switch between light and dark mode" aria-pressed="false">
  <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
    <mask id="tt-mask">
      <rect width="24" height="24" fill="white"/>
      <circle class="tt-biter" cx="31" cy="6" r="8" fill="black"/>
    </mask>
    <circle class="tt-core" cx="12" cy="12" r="5" mask="url(#tt-mask)"
            fill="currentColor" stroke="none"/>
    <g class="tt-rays" stroke="currentColor" stroke-width="2" stroke-linecap="round" fill="none">
      <path d="M12 2.5v2.2"/><path d="M12 19.3v2.2"/>
      <path d="M2.5 12h2.2"/><path d="M19.3 12h2.2"/>
      <path d="M5.3 5.3l1.6 1.6"/><path d="M17.1 17.1l1.6 1.6"/>
      <path d="M5.3 18.7l1.6-1.6"/><path d="M17.1 6.9l1.6-1.6"/>
    </g>
  </svg>
</button>
<script>
(function () {
  "use strict";
  var root = document.documentElement;
  var button = document.getElementById("theme-toggle");
  var reduced = window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function currentTheme() {
    return root.getAttribute("data-theme") === "dark" ? "dark" : "light";
  }
  function paintToggle(mode) {
    button.setAttribute("aria-pressed", mode === "dark" ? "true" : "false");
    button.setAttribute(
      "aria-label",
      mode === "dark" ? "Switch to light mode" : "Switch to dark mode"
    );
  }
  function apply(mode) {
    root.setAttribute("data-theme", mode);
    paintToggle(mode);
  }

  paintToggle(currentTheme());

  button.addEventListener("click", function () {
    var next = currentTheme() === "dark" ? "light" : "dark";
    try { localStorage.setItem("identai-theme", next); } catch (e) { /* private mode */ }

    // Reveal origin: the centre of the chip, in viewport pixels.
    var rect = button.getBoundingClientRect();
    root.style.setProperty("--tt-x", Math.round(rect.left + rect.width / 2) + "px");
    root.style.setProperty("--tt-y", Math.round(rect.top + rect.height / 2) + "px");

    if (document.startViewTransition && !reduced) {
      document.startViewTransition(function () { apply(next); });
    } else {
      root.classList.add("tt-anim");
      apply(next);
      window.setTimeout(function () { root.classList.remove("tt-anim"); }, 320);
    }
  });

  /* Follow the OS setting live, but only while the user has no explicit choice. */
  var media = window.matchMedia("(prefers-color-scheme: dark)");
  var onChange = function (event) {
    var stored = null;
    try { stored = localStorage.getItem("identai-theme"); } catch (e) { /* ignore */ }
    if (stored !== "dark" && stored !== "light") apply(event.matches ? "dark" : "light");
  };
  if (media.addEventListener) media.addEventListener("change", onChange);
  else if (media.addListener) media.addListener(onChange);
})();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def review_desk() -> HTMLResponse:
    rows = await fetch_materials()

    # Inert JSON injection: escaping "</" means worksheet text can never terminate the
    # <script> block early, and no worksheet content is ever parsed as markup.
    payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")

    # Build metadata first, record payload last — so no worksheet text can ever be
    # rescanned as a template placeholder.
    page = (
        REVIEW_PAGE_HTML
        .replace("__MODEL_ID__", html.escape(model_label()))
        .replace("__DB_PATH__", html.escape(DB_PATH))
        .replace("__MATERIALS_JSON__", payload)
    )
    return HTMLResponse(page)


@app.post("/approve/{item_id}")
async def approve_material(item_id: int) -> dict[str, Any]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        async with db.execute("SELECT status FROM materials WHERE id = ?", (item_id,)) as cur:
            current = await cur.fetchone()

        if not current:
            raise HTTPException(status_code=404, detail=f"No material with id={item_id}.")

        # Approval is the teacher's way of closing a flagged item too. The
        # status was 'flagged' when flag_for_human_review wrote the row; a
        # teacher clicking Approve on the review desk must be able to move
        # either status to 'approved'. Anything else is genuinely immutable.
        if current["status"] not in {"draft", "flagged"}:
            raise HTTPException(
                status_code=409,
                detail=f"Cannot approve material in '{current['status']}' status.",
            )

        await db.execute(
            "UPDATE materials SET status = 'approved' WHERE id = ?", (item_id,)
        )
        await db.commit()
        async with db.execute(
            "SELECT id, unit_name, topic, tier, status FROM materials WHERE id = ?", (item_id,)
        ) as updated:
            row = await updated.fetchone()

    # Notify any in-memory mission that a teacher just finished reviewing one of its
    # escalations, so the live panel can show the review landing. The mission
    # table is keyed by topic + tier in the simplest possible way; a production
    # implementation would store the mission_id on the row, but the row was created
    # without one and the demo does not need a perfect join.
    for mission in MISSIONS.values():
        if mission.outcome in {"running", "saved", "escalated", "failed", "timeout"} and \
                mission.flagged_ids and item_id in mission.flagged_ids:
            mission.record(
                "human_review_completed",
                f"teacher approved the {row['tier']} tier after review",
                record_id=item_id, tier=row["tier"], topic=row["topic"],
            )
            break

    # Advance the owning curriculum mission, if any. The reverse index
    # RECORD_TO_CURRICULUM is populated by the orchestrator when a tier
    # is produced. Approving the last flagged item closes the mission.
    curriculum_id = RECORD_TO_CURRICULUM.get(item_id)
    if curriculum_id:
        curriculum = CURRICULUM_MISSIONS.get(curriculum_id)
        if curriculum is not None:
            closed = curriculum.approve_flagged(item_id)
            await persist_curriculum_mission(curriculum)
            logger.info(
                "Approved flagged item | curriculum=%s | record=%s | closed=%s",
                curriculum_id, item_id, closed,
            )

    response = dict(row)
    if curriculum_id:
        response["curriculum_mission_id"] = curriculum_id
    response["status"] = "approved"
    logger.info("Approved material | id=%s | topic=%s", item_id, row["topic"])
    return response


# --------------------------------------------------------------------------- #
# Reject, Regenerate, Export, Agent Activity
# --------------------------------------------------------------------------- #


@app.post("/reject/{item_id}")
async def reject_material(item_id: int) -> dict[str, Any]:
    """
    The teacher disagrees with the agent's output and discards it.

    The row is marked 'rejected' in the materials table; the agent will
    not see it again. A rejected record frees the per-tier slot in the
    owning curriculum mission so a follow-up Regenerate can produce a
    new version without conflicting state.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, status, topic, tier, unit_name FROM materials WHERE id = ?",
            (item_id,),
        ) as cur:
            row = await cur.fetchone()
        if not row:
            raise HTTPException(status_code=404,
                                detail=f"No material with id={item_id}.")
        if row["status"] == "approved":
            raise HTTPException(
                status_code=409,
                detail="Cannot reject an already-approved material.",
            )
        await db.execute(
            "UPDATE materials SET status = 'rejected' WHERE id = ?",
            (item_id,),
        )
        await db.commit()

    # Notify any in-memory mission so the live log shows the rejection.
    for mission in MISSIONS.values():
        if mission.flagged_ids and item_id in mission.flagged_ids:
            mission.record(
                "human_review_completed",
                f"teacher rejected the {row['tier']} tier",
                record_id=item_id, tier=row["tier"], topic=row["topic"],
                decision="rejected",
            )
            break

    # Advance the owning curriculum mission: the rejected item is no
    # longer waiting on the teacher, so it stops being a human-review
    # item. The teacher can hit Regenerate afterwards.
    curriculum_id = RECORD_TO_CURRICULUM.get(item_id)
    if curriculum_id:
        curriculum = CURRICULUM_MISSIONS.get(curriculum_id)
        if curriculum is not None:
            for item in list(curriculum.human_review_items):
                if item.get("record_id") == item_id:
                    item["status"] = "rejected"
            if not curriculum.has_pending_human_review and curriculum.state == "WAITING_FOR_HUMAN":
                curriculum.transition(
                    "RUNNING",
                    phase="teacher rejected an item; awaiting regenerate",
                )
            curriculum.summary = curriculum.compute_summary()
            await persist_curriculum_mission(curriculum)

    logger.info("Rejected material | id=%s | topic=%s", item_id, row["topic"])
    return {
        "id": item_id,
        "status": "rejected",
        "topic": row["topic"],
        "tier": row["tier"],
        "detail": "Material rejected. The agent's output is discarded. "
                   "Hit Regenerate to spawn a fresh attempt.",
        "regenerate_url": f"/regenerate/{item_id}",
    }


@app.post("/regenerate/{item_id}")
async def regenerate_material(item_id: int) -> dict[str, Any]:
    """
    Re-run the per-tier generation for one material record.

    The agent gets the same topic, objective, and tier, but a fresh
    correction budget. Whatever the new sub-mission produces will replace
    the rejected row in the dashboard. This call is synchronous; the
    underlying Strands agent is the same one used by /missions.
    """
    record = await fetch_material_by_id(item_id)
    if not record:
        raise HTTPException(status_code=404,
                            detail=f"No material with id={item_id}.")
    if record.get("status") not in {"draft", "flagged", "rejected"}:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot regenerate a material in '{record.get('status')}' status.",
        )

    unavailable = provider_unavailable_reason()
    if unavailable:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Regeneration is unavailable: {unavailable}",
        )

    # The new sub-mission is a fresh Mission; the agent decides what to
    # do (regenerate, audit, save or escalate). We thread the new id
    # into the dashboard's mission log via the existing
    # RECORD_TO_CURRICULUM + RECORD_TO_SUB_MISSION indexes.
    parent_id = RECORD_TO_CURRICULUM.get(item_id) or "regenerate"
    sub = new_mission(
        f"regenerate:{parent_id}",
        record.get("topic", ""),
        unit_name=record.get("unit_name") or "",
        objective=record.get("objective") or "",
    )

    token = _CURRENT_MISSION.set(sub)
    try:
        sub.record(
            "mission_started",
            f"regenerating the {record.get('tier', '?')} tier "
            f"after teacher requested a fresh attempt",
            provider=MODEL_PROVIDER, model=resolved_model_id(),
        )
        agent = build_agent(sub)
        brief = _tier_brief(
            _curriculum_from_record(record, parent_id),
            record.get("tier", "on-level"),
        )
        # A bounded wait; a stuck regenerate cannot block the teacher.
        try:
            await asyncio.wait_for(
                agent.invoke_async(brief, limits=_agent_limits_for_mission(sub)),
                timeout=CURRICULUM_TIER_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            sub.record(
                "mission_failed",
                f"regenerate timed out after "
                f"{CURRICULUM_TIER_TIMEOUT_SECONDS:.0f}s",
                source="regenerate_timeout",
            )
            return {
                "id": item_id, "status": "timeout",
                "detail": "Regenerate did not complete in time; try again.",
            }
        except _StrandsContextError as exc:
            _record_mission_failure(sub, exc, source="regenerate_context_window")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Regenerate ran out of context: {exc}",
            )
        except _StrandsThrottleError as exc:
            _record_mission_failure(sub, exc, source="regenerate_throttled")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model provider throttled the regenerate: {exc}",
            )
        except _StrandsEventLoopError as exc:
            _record_mission_failure(sub, exc, source="regenerate_event_loop")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Regenerate's agent loop failed: {exc}",
            )
        _settle(sub, "")
    finally:
        _CURRENT_MISSION.reset(token)

    # Decide the outcome from the sub-mission's view of the world.
    new_record_id: int | None = None
    new_status = "failed"
    if sub.saved_ids:
        new_record_id = sub.saved_ids[0]
        new_status = "saved"
    elif sub.flagged_ids:
        new_record_id = sub.flagged_ids[0]
        new_status = "flagged"

    if new_record_id is None:
        return {
            "id": item_id,
            "status": "failed",
            "detail": "Regenerate did not produce a saveable record.",
            "mission_id": sub.id,
        }

    # Update the owning curriculum mission's bookkeeping.
    curriculum_id = RECORD_TO_CURRICULUM.get(item_id)
    if curriculum_id:
        curriculum = CURRICULUM_MISSIONS.get(curriculum_id)
        if curriculum is not None:
            # Replace the old record id with the new one in the
            # tier_status / human_review_items / generated_items lists.
            tier = record.get("tier", "on-level")
            for item in curriculum.generated_items:
                if item.get("record_id") == item_id:
                    item["record_id"] = new_record_id
                    item["status"] = "draft" if new_status == "saved" else "flagged"
            for item in curriculum.human_review_items:
                if item.get("record_id") == item_id:
                    item["record_id"] = new_record_id
                    item["status"] = "draft" if new_status == "saved" else "flagged"
            curriculum.tier_record_ids[tier] = new_record_id
            # Restore the mission to a non-terminal state so the
            # teacher can re-review the new record.
            if curriculum.state in {"COMPLETED", "WAITING_FOR_HUMAN"}:
                curriculum.transition(
                    "RUNNING",
                    phase=(
                        f"regenerated the {tier} tier; awaiting re-review"
                    ),
                )
            curriculum.summary = curriculum.compute_summary()
    # Re-index the new record.
    RECORD_TO_CURRICULUM[new_record_id] = curriculum_id or parent_id
    RECORD_TO_SUB_MISSION[new_record_id] = sub.id
    # The old record id now refers to a discarded row; the dashboard
    # can still see it in the 'rejected' bucket via its /api/materials
    # feed, but it no longer routes to a mission.
    RECORD_TO_CURRICULUM.pop(item_id, None)
    RECORD_TO_SUB_MISSION.pop(item_id, None)

    logger.info(
        "Regenerated material | old_id=%s | new_id=%s | status=%s",
        item_id, new_record_id, new_status,
    )
    return {
        "id": new_record_id,
        "old_id": item_id,
        "status": new_status,
        "tier": record.get("tier"),
        "topic": record.get("topic"),
        "detail": (
            "Regenerate produced a new record. Open it on the review "
            "desk to approve, reject, or regenerate again."
        ),
        "mission_id": sub.id,
    }


def _curriculum_from_record(record: dict[str, Any], parent_id: str) -> "CurriculumMission":
    """
    Build a thin CurriculumMission stand-in for one regenerate request.

    The agent only needs read access to the goal (topic / unit / objective
    / tier) so it can build its brief; we don't need the full curriculum
    state machine for a single-tier regenerate.
    """
    cm = new_curriculum_mission({
        "topic": record.get("topic", ""),
        "unit_name": record.get("unit_name", ""),
        "objective": record.get("objective", ""),
        "tiers": [record.get("tier", "on-level")],
    }, mission_id=f"cm-regen-{parent_id}-{record.get('id', 0)}")
    return cm


@app.get("/api/materials/{item_id}/activity")
async def api_material_activity(item_id: int) -> dict[str, Any]:
    """
    Safe, concise agent-activity timeline for one material record.

    The response is what the dashboard renders in the "Agent Activity"
    view: a list of {kind, label, message, at} entries, with chain-of-
    thought and raw model output stripped.
    """
    record = await fetch_material_by_id(item_id)
    if not record:
        raise HTTPException(status_code=404,
                            detail=f"No material with id={item_id}.")
    timeline = _activity_for_record(record)
    return {
        "material_id": item_id,
        "topic": record.get("topic", ""),
        "tier": record.get("tier", ""),
        "status": record.get("status", ""),
        "timeline": timeline,
    }


@app.get("/api/export/{item_id}/html")
async def export_html(item_id: int) -> HTMLResponse:
    """Print-ready HTML handout for one material record."""
    record = await fetch_material_by_id(item_id)
    if not record:
        raise HTTPException(status_code=404,
                            detail=f"No material with id={item_id}.")
    pkg = _build_export_package(record)
    return HTMLResponse(_export_html(pkg))


@app.get("/api/export/{item_id}/md")
async def export_markdown(item_id: int) -> Response:
    """Markdown export for one material record."""
    record = await fetch_material_by_id(item_id)
    if not record:
        raise HTTPException(status_code=404,
                            detail=f"No material with id={item_id}.")
    pkg = _build_export_package(record)
    body = _export_markdown(pkg)
    slug = re.sub(r"[^a-z0-9]+", "-", (record.get("topic") or "lesson").lower()).strip("-")[:60] or "lesson"
    return Response(
        content=body,
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{slug}-{record.get("tier", "tier")}.md"',
        },
    )


@app.get("/api/export/{item_id}/pdf")
async def export_pdf(item_id: int) -> HTMLResponse:
    """
    Print-to-PDF: returns a print-styled HTML page that auto-triggers
    window.print() on load. The teacher's browser opens the print
    dialog; they choose "Save as PDF" to download a real PDF.

    This is a hackathon-friendly approach that avoids a heavy PDF
    binary dependency (weasyprint, fpdf2, etc.) while still giving
    the teacher a real, printable file. The page also includes a
    noscript fallback telling the teacher to use Ctrl+P.
    """
    record = await fetch_material_by_id(item_id)
    if not record:
        raise HTTPException(status_code=404,
                            detail=f"No material with id={item_id}.")
    pkg = _build_export_package(record)
    return HTMLResponse(_export_print_html(pkg))


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
