"""
Bounded pseudo-code interpreter + autonomous learning-agent runtime.

This module is two layers with one hard rule between them (the hackathon
brief: "the agent runs autonomously and only surfaces when there's a real
decision to make"):

*The interpreter core* — `execute_pseudocode` and friends. Pure, bounded,
deterministic, no I/O. It is the substrate every silent background check
runs on, and the public surface existing callers import.

*The autonomous runtime* — `AutonomousAgentEngine`. A long-lived background
agent that continuously ingests student submissions / learning events from
`data/lessons.db`, analyses them silently (syntax analysis, unit-test runs,
milestone evaluation — all deterministic, all free), commits state
transitions autonomously, and pauses to surface a structured "Escalation
Decision Brief" only when a human judgement is genuinely required. The
single human entry point that resumes it is `resolve_decision`.

Strands Agents SDK is the orchestration substrate for the runtime, used
natively rather than as decoration:

  * tool routing — the tools the reasoning agent may call during an
    escalation (`run_code_tests`, `fetch_student_history`,
    `submit_decision_brief`, `unlock_advanced_module`) are real Strands
    `@tool` callables; the model decides which to route to;
  * state memory — the escalation agent carries Strands `AgentState` plus a
    `FileSessionManager`, so conversation memory and engine state survive
    process restarts instead of living in one invocation;
  * orchestration — `strands.Agent`'s event loop drives the escalation
    reasoning; an `InterventionHandler` enforces the human gate at the
    tool-routing layer itself: `unlock_advanced_module` is DENIED unless a
    human has approved that exact unlock, no matter what the model tries.

Deterministic-first: with no model provider configured (or Strands missing)
every escalation still happens — the brief is simply composed by rule
instead of by a model. The agent never stops being autonomous because a
credential expired; it only stops being chatty.
"""
from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import aiosqlite

from .evaluator import (
    AutonomyDecision,
    ClaimAssessment,
    MilestoneProfile,
    MilestoneVerdict,
    SyntaxReport,
    UnitCase,
    analyze_syntax,
    assess_claim,
    decide_autonomy,
    evaluate_milestone,
    interp_eval,
    run_unit_tests,
)
from .models import ExecutionResult, _IfFrame, _RepeatFrame, _StackFrame, _WhileFrame

# Bounded pseudo-code interpreter constants
INTERPRETER_STEP_CAP = 5000

INTERPRETER_ERRORS: dict[str, str] = {
    "syntax":  "the pseudo-code is not in the taught subset",
    "undef":   "the pseudo-code reads a variable that was never SET or INPUT",
    "infinite": "the pseudo-code did not finish within the step cap; execution may not terminate",
    "divzero": "the pseudo-code divides by zero",
    "type":    "the pseudo-code mixes types in a way the interpreter cannot follow",
}

DEFAULT_INPUTS: dict[str, dict[str, int | str | float]] = {
    "counter": {"counter": 0},
    "score":   {"score": 0},
    "total":   {"total": 0},
    "count":   {"count": 0},
    "x":       {"x": 0},
    "n":       {"n": 0},
}

PSEUDO_KEYWORDS: tuple[str, ...] = (
    "SET", "INPUT", "PRINT", "DISPLAY", "IF", "ELSE IF", "ELSE", "END IF", "ENDIF",
    "REPEAT", "END REPEAT", "WHILE", "END WHILE", "STOP", "END",
)


class _PseudoRun:
    """Mutable execution context for a single run."""

    __slots__ = ("vars", "output", "steps", "cap", "error")

    def __init__(self, cap: int) -> None:
        self.vars: dict[str, Any] = {}
        self.output: list[str] = []
        self.steps = 0
        self.cap = cap
        self.error: tuple[str, str] | None = None

    def step(self) -> bool:
        """Return False if the cap has been reached; otherwise advance and return True."""
        self.steps += 1
        if self.steps > self.cap:
            self.error = ("infinite", INTERPRETER_ERRORS["infinite"])
            return False
        return True


def leading_keyword(line: str) -> str | None:
    """The pseudo-code keyword a line opens with, if it opens with one."""
    upper = line.strip().upper()
    for keyword in sorted(PSEUDO_KEYWORDS, key=len, reverse=True):
        if upper == keyword or upper.startswith(keyword + " "):
            return keyword
    return None


def code_lines(text: str) -> list[tuple[int, str]]:
    """
    Extract pseudo-code lines (line number, stripped line) from mixed prose.
    """
    found: list[tuple[int, str]] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if leading_keyword(line) is not None:
            found.append((number, line))
    return found


def normalise_output_line(s: str) -> str:
    """Lower-case, collapse whitespace, strip trailing punctuation."""
    s = " ".join(s.strip().lower().split())
    return s.rstrip(".,;:").rstrip()


def outputs_match(actual: list[str], expected: list[str]) -> bool:
    """Two output lists match iff the same lines match after whitespace and case normalisation."""
    if not expected:
        return True
    if len(actual) != len(expected):
        return False
    return all(
        normalise_output_line(a) == normalise_output_line(e)
        for a, e in zip(actual, expected)
    )


def run_pseudocode(
    parsed_lines: list[tuple[int, str]],
    *,
    inputs: dict[str, Any] | None = None,
    line_offset: int = 0,
    step_cap: int = INTERPRETER_STEP_CAP,
) -> ExecutionResult:
    """
    Execute a list of pre-parsed pseudo-code lines (line_number, raw_text).
    """
    run = _PseudoRun(step_cap)
    known_inputs = {name: dict(values) for name, values in DEFAULT_INPUTS.items()}
    if inputs:
        for k, v in inputs.items():
            run.vars[k] = v

    stack: list[_StackFrame] = []
    pc = 0
    # Nesting depth of IF blocks being skipped while a false branch's body
    # is bypassed (an IF inside a skipped body nests its END IF too).
    skip_depth = 0

    while pc < len(parsed_lines):
        line_no, raw = parsed_lines[pc]
        line = raw.strip()
        upper = line.upper()

        if not run.step():
            return ExecutionResult(
                ok=False,
                output=run.output,
                steps=run.steps,
                error=("infinite", INTERPRETER_ERRORS["infinite"]),
                line=line_no + line_offset,
            )

        # False branch: the innermost open IF frame has not run (condition
        # false, or an earlier ELSE IF / ELSE branch already matched), so
        # body lines do not execute. Skipped lines only watch for the
        # chain's own markers — ELSE IF / ELSE / END IF at depth 0 resume
        # normal dispatch; anything else (including nested IF blocks,
        # loops, prints) is bypassed without executing.
        if stack and isinstance(stack[-1], _IfFrame) and not stack[-1].condition_met:
            chain_marker = (
                upper in {"END IF", "ENDIF"}
                or upper == "ELSE"
                or upper.startswith("ELSE IF ")
                or upper.startswith("ELSEIF ")
            )
            if skip_depth == 0 and chain_marker:
                pass  # the false branch's own chain marker: dispatch below
            else:
                if upper.startswith("IF "):
                    skip_depth += 1
                elif skip_depth and upper in {"END IF", "ENDIF"}:
                    skip_depth -= 1
                pc += 1
                continue

        if upper.startswith("SET "):
            try:
                target, value_expr = line[4:].split(" TO ", 1) if " TO " in line[4:].upper() \
                    else line[4:].split("=", 1)
                target = target.strip()
                value = interp_eval(value_expr.strip(), run.vars)
            except (ValueError, KeyError, ZeroDivisionError) as exc:
                kind = "divzero" if isinstance(exc, ZeroDivisionError) else \
                    "undef" if isinstance(exc, KeyError) else "type"
                run.error = (kind, INTERPRETER_ERRORS[kind] + f" (line {line_no})")
                return ExecutionResult(
                    ok=False,
                    output=run.output,
                    steps=run.steps,
                    error=run.error,
                    line=line_no + line_offset,
                )
            run.vars[target] = value

        elif upper.startswith("INPUT "):
            var = line[6:].strip().split()[0] if len(line) > 6 else ""
            if not var:
                run.error = ("syntax", "INPUT expects a variable name")
                return ExecutionResult(
                    ok=False,
                    output=run.output,
                    steps=run.steps,
                    error=run.error,
                    line=line_no + line_offset,
                )
            # Pick a stub value if not already provided in environment
            if var not in run.vars:
                if var.lower() in known_inputs:
                    run.vars[var] = next(iter(known_inputs[var.lower()].values()))
                elif any(ch.isdigit() for ch in var) or var.lower() in {"n", "count", "x"}:
                    run.vars[var] = 0
                else:
                    run.vars[var] = ""

        elif upper.startswith("PRINT ") or upper.startswith("DISPLAY "):
            value_expr = line.split(" ", 1)[1] if " " in line else ""
            try:
                value = interp_eval(value_expr, run.vars)
            except (KeyError, ValueError, ZeroDivisionError) as exc:
                kind = "divzero" if isinstance(exc, ZeroDivisionError) else \
                    "undef" if isinstance(exc, KeyError) else "type"
                run.error = (kind, INTERPRETER_ERRORS[kind] + f" (line {line_no})")
                return ExecutionResult(
                    ok=False,
                    output=run.output,
                    steps=run.steps,
                    error=run.error,
                    line=line_no + line_offset,
                )
            run.output.append(str(value))

        elif upper.startswith("IF "):
            cond_expr = line[3:].rstrip().rstrip(".") if line.endswith(".") else line[3:]
            upper_cond = cond_expr.upper()
            if " THEN" in upper_cond:
                cond_expr = cond_expr[: upper_cond.find(" THEN")]
            try:
                cond = interp_eval(cond_expr.strip(), run.vars)
            except (KeyError, ValueError, ZeroDivisionError) as exc:
                kind = "divzero" if isinstance(exc, ZeroDivisionError) else \
                    "undef" if isinstance(exc, KeyError) else "type"
                run.error = (kind, INTERPRETER_ERRORS[kind] + f" (line {line_no})")
                return ExecutionResult(
                    ok=False,
                    output=run.output,
                    steps=run.steps,
                    error=run.error,
                    line=line_no + line_offset,
                )
            stack.append(_IfFrame(start_pc=pc, condition_met=bool(cond)))

        elif upper.startswith("ELSE IF ") or upper.startswith("ELSEIF "):
            if not stack or not isinstance(stack[-1], _IfFrame):
                run.error = ("syntax", "ELSE IF without matching IF")
                return ExecutionResult(
                    ok=False,
                    output=run.output,
                    steps=run.steps,
                    error=run.error,
                    line=line_no + line_offset,
                )
            if stack[-1].condition_met:
                # IF (or earlier ELSE IF) was true; skip the rest of the chain.
                depth = 1
                pc += 1
                while pc < len(parsed_lines) and depth:
                    inner_upper = parsed_lines[pc][1].strip().upper()
                    if inner_upper.startswith("IF "):
                        depth += 1
                    elif inner_upper in {"END IF", "ENDIF"}:
                        depth -= 1
                    pc += 1
                stack.pop()
                continue
            stack.pop()  # discard the false IF frame
            cond_expr = line.split(" ", 2)[2] if len(line.split()) > 2 else ""
            upper_cond = cond_expr.upper()
            if " THEN" in upper_cond:
                cond_expr = cond_expr[: upper_cond.find(" THEN")]
            try:
                cond = interp_eval(cond_expr.strip(), run.vars)
            except (KeyError, ValueError, ZeroDivisionError) as exc:
                kind = "divzero" if isinstance(exc, ZeroDivisionError) else \
                    "undef" if isinstance(exc, KeyError) else "type"
                run.error = (kind, INTERPRETER_ERRORS[kind] + f" (line {line_no})")
                return ExecutionResult(
                    ok=False,
                    output=run.output,
                    steps=run.steps,
                    error=run.error,
                    line=line_no + line_offset,
                )
            stack.append(_IfFrame(start_pc=pc, condition_met=bool(cond)))

        elif upper == "ELSE":
            if not stack or not isinstance(stack[-1], _IfFrame):
                run.error = ("syntax", "ELSE without matching IF")
                return ExecutionResult(
                    ok=False,
                    output=run.output,
                    steps=run.steps,
                    error=run.error,
                    line=line_no + line_offset,
                )
            stack[-1] = _IfFrame(start_pc=stack[-1].start_pc, condition_met=not stack[-1].condition_met)

        elif upper in {"END IF", "ENDIF"}:
            if stack and isinstance(stack[-1], _IfFrame):
                stack.pop()

        elif upper.startswith("REPEAT "):
            tail = line[7:].strip()
            upper_tail = tail.upper()
            n_value: int | None = None
            if upper_tail.endswith(" TIMES"):
                count_expr = tail[: -len(" TIMES")].strip()
                try:
                    n_value = int(interp_eval(count_expr, run.vars))
                except (KeyError, ValueError):
                    pass
            if n_value is None:
                run.error = ("syntax", "REPEAT without a TIMES count")
                return ExecutionResult(
                    ok=False,
                    output=run.output,
                    steps=run.steps,
                    error=run.error,
                    line=line_no + line_offset,
                )
            stack.append(_RepeatFrame(start_pc=pc, target=n_value, done=0))

        elif upper == "END REPEAT":
            if not stack or not isinstance(stack[-1], _RepeatFrame):
                run.error = ("syntax", "END REPEAT without matching REPEAT")
                return ExecutionResult(
                    ok=False,
                    output=run.output,
                    steps=run.steps,
                    error=run.error,
                    line=line_no + line_offset,
                )
            frame = stack[-1]
            frame.done += 1
            if frame.done < frame.target:
                # Jump back to line after REPEAT
                pc = frame.start_pc + 1
                continue
            stack.pop()

        elif upper.startswith("WHILE "):
            cond_expr = line[6:].strip()
            try:
                cond = interp_eval(cond_expr, run.vars)
            except (KeyError, ValueError, ZeroDivisionError) as exc:
                kind = "divzero" if isinstance(exc, ZeroDivisionError) else \
                    "undef" if isinstance(exc, KeyError) else "type"
                run.error = (kind, INTERPRETER_ERRORS[kind] + f" (line {line_no})")
                return ExecutionResult(
                    ok=False,
                    output=run.output,
                    steps=run.steps,
                    error=run.error,
                    line=line_no + line_offset,
                )
            while_pc = pc
            if not cond:
                depth = 1
                pc += 1
                while pc < len(parsed_lines) and depth:
                    inner_upper = parsed_lines[pc][1].strip().upper()
                    if inner_upper.startswith("WHILE "):
                        depth += 1
                    elif inner_upper == "END WHILE":
                        depth -= 1
                    pc += 1
                if stack and isinstance(stack[-1], _WhileFrame) and stack[-1].start_pc == while_pc:
                    stack.pop()
                continue
            if not (stack and isinstance(stack[-1], _WhileFrame) and stack[-1].start_pc == pc):
                stack.append(_WhileFrame(start_pc=pc))

        elif upper == "END WHILE":
            if not stack or not isinstance(stack[-1], _WhileFrame):
                run.error = ("syntax", "END WHILE without matching WHILE")
                return ExecutionResult(
                    ok=False,
                    output=run.output,
                    steps=run.steps,
                    error=run.error,
                    line=line_no + line_offset,
                )
            pc = stack[-1].start_pc
            continue

        elif upper in {"STOP", "END"}:
            return ExecutionResult(
                ok=True,
                output=run.output,
                steps=run.steps,
                error=None,
                line=line_no + line_offset,
            )

        else:
            run.error = ("syntax", f"Unknown pseudo-code keyword: {line!r}")
            return ExecutionResult(
                ok=False,
                output=run.output,
                steps=run.steps,
                error=run.error,
                line=line_no + line_offset,
            )

        pc += 1

    if stack:
        return ExecutionResult(
            ok=False,
            output=run.output,
            steps=run.steps,
            error=("syntax", f"unclosed block starting at line {parsed_lines[stack[-1].start_pc][0] + line_offset}"),
            line=parsed_lines[stack[-1].start_pc][0] + line_offset,
        )

    return ExecutionResult(
        ok=True,
        output=run.output,
        steps=run.steps,
        error=None,
        line=None,
    )


def execute_pseudocode(
    text: str,
    *,
    inputs: dict[str, Any] | None = None,
    line_offset: int = 0,
    step_cap: int = INTERPRETER_STEP_CAP,
) -> ExecutionResult:
    """
    Parse and execute pseudo-code lines from `text`.
    """
    parsed = code_lines(text)
    if not parsed:
        return ExecutionResult(
            ok=False,
            output=[],
            steps=0,
            error=("syntax", "No pseudo-code lines found in the text."),
            line=None,
        )
    return run_pseudocode(parsed, inputs=inputs, line_offset=line_offset, step_cap=step_cap)


# ============================================================================ #
#
# AUTONOMOUS RUNTIME
#
# The contract this section implements (hackathon rule):
#
#   AUTONOMOUS — happens in the background, never interrupts a human:
#     * ingesting submissions / learning events (any moment, any volume)
#     * syntax analysis, unit-test runs, milestone evaluation
#     * telemetry for every silent action
#     * state transitions: pending -> auto_passed / auto_failed,
#       milestone counters, completion, advice re-injection and resume
#
#   ESCALATION — the ONLY three reasons a human ever hears from the agent:
#     * stuck_loop          — student is past the consecutive-failure gate
#     * hallucination_risk  — a claimed output contradicts deterministic
#                             execution, so some material cannot be trusted
#     * permission_gate     — an advanced module stays locked until a human
#                             explicitly approves the unlock
#   Each pause writes one structured "Escalation Decision Brief" row and
#   freezes the affected work. `resolve_decision(decision_id, human_input)`
#   is the single hook that applies a human's approve / reject / advise and
#   resumes the background loop.
#
# ============================================================================ #

_runtime_log = logging.getLogger("lesson-agent.learning-engine")

DecisionCategory = Literal["stuck_loop", "hallucination_risk", "permission_gate"]
DecisionResolution = Literal["approve", "reject", "advise"]

#: Submissions stuck in 'processing' longer than this are considered
#: abandoned by a dead worker and re-claimed automatically.
PROCESSING_STALE_SECONDS = 300.0

ESCALATION_SYSTEM_PROMPT = """\
You are the review stage of an autonomous learning agent. You never talk to
students and you never chat. A deterministic pipeline has already executed
the student's code, graded it against unit tests, and measured progress; it
has hit a situation whose resolution is a judgement call, not effort.

Your only job: review the evidence and call the `submit_decision_brief`
tool EXACTLY ONCE with a decision brief a teacher can act on in under a
minute. Use one of the categories you are given. Ground every claim in the
evidence provided — do not speculate about the student. Then stop.

You may call `fetch_student_history` or `run_code_tests` first if the
evidence is genuinely insufficient, but the evidence is usually enough.
Never call `unlock_advanced_module` yourself: that tool is gated behind an
explicit human approval and will refuse.
"""


# --------------------------------------------------------------------------- #
# Strands availability — the runtime is deterministic-first, so a missing or
# unusable SDK degrades the *narrative*, never the autonomy.
# --------------------------------------------------------------------------- #

try:  # pragma: no cover - exercised implicitly by import in every mode
    from strands import Agent as _StrandsAgent, tool as _strands_tool
    from strands.agent.state import AgentState as _AgentState
    from strands.interventions import Deny as _Deny, Proceed as _Proceed
    from strands.interventions.handler import InterventionHandler as _InterventionHandler
    from strands.session.file_session_manager import FileSessionManager as _FileSessionManager

    STRANDS_AVAILABLE = True
except ImportError:  # pragma: no cover - strands is a hard dep of main.py but the
    # interpreter package stays importable without it for pure interpreter use.
    _StrandsAgent = None  # type: ignore[assignment]
    _strands_tool = None  # type: ignore[assignment]
    _AgentState = None  # type: ignore[assignment]
    _Deny = _Proceed = None  # type: ignore[assignment]
    _InterventionHandler = None  # type: ignore[assignment]
    _FileSessionManager = None  # type: ignore[assignment]
    STRANDS_AVAILABLE = False


def _resolve_model() -> Any | None:
    """
    Build the Strands model provider for escalation reasoning, mirroring
    app/main.py's provider selection.

    Returns None whenever no real provider is usable (mock provider, missing
    credentials, missing SDK extras) — the engine then composes briefs
    deterministically. Construction never calls AWS; only invocation does,
    and invocation failures are caught by the caller.
    """
    if not STRANDS_AVAILABLE:
        return None
    provider = os.getenv("MODEL_PROVIDER", "bedrock").strip().lower() or "bedrock"
    if provider == "mock":
        return None
    if provider == "bedrock":
        try:
            from strands.models import BedrockModel  # noqa: PLC0415
        except ImportError:  # pragma: no cover - import path moved across releases
            try:
                from strands.models.bedrock import BedrockModel  # noqa: PLC0415
            except ImportError:
                return None
        settings: dict[str, Any] = {"temperature": 0.2, "max_tokens": 2048}
        if os.getenv("BEDROCK_MODEL_ID"):
            settings["model_id"] = os.getenv("BEDROCK_MODEL_ID")
        region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
        if region:
            settings["region_name"] = region
        try:
            return BedrockModel(**settings)
        except Exception:  # noqa: BLE001 - provider misconfiguration must not kill the loop
            _runtime_log.warning("Bedrock model unavailable; escalation briefs stay deterministic.")
            return None
    if provider == "litellm":
        try:
            from strands.models.litellm import LiteLLMModel  # noqa: PLC0415
        except ImportError:
            return None
        if not os.getenv("GEMINI_API_KEY"):
            return None
        try:
            return LiteLLMModel(
                client_args={"api_key": os.getenv("GEMINI_API_KEY")},
                model_id=os.getenv("GEMINI_MODEL_ID", "gemini/gemini-2.0-flash"),
                params={"max_tokens": 2048},
            )
        except Exception:  # noqa: BLE001
            return None
    return None


# --------------------------------------------------------------------------- #
# Runtime value types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DecisionBrief:
    """
    The structured "Escalation Decision Brief" a mentor sees.

    One brief per pause. It is deliberately small: a title, a summary a
    teacher can act on in under a minute, the machine evidence behind it,
    and the three resolutions the runtime knows how to apply. This object
    is the *only* thing the autonomous agent ever shows a human.
    """

    decision_id: str
    category: DecisionCategory
    severity: Literal["low", "medium", "high"]
    student_id: str
    milestone_id: str
    title: str
    summary: str
    evidence: dict[str, Any] = field(default_factory=dict)
    recommended_action: DecisionResolution | None = None
    options: tuple[DecisionResolution, ...] = ("approve", "reject", "advise")
    status: Literal["open", "resolved"] = "open"
    resolution: DecisionResolution | None = None
    human_input: str | None = None
    submission_id: int | None = None
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "category": self.category,
            "severity": self.severity,
            "student_id": self.student_id,
            "milestone_id": self.milestone_id,
            "title": self.title,
            "summary": self.summary,
            "evidence": self.evidence,
            "recommended_action": self.recommended_action,
            "options": list(self.options),
            "status": self.status,
            "resolution": self.resolution,
            "human_input": self.human_input,
            "submission_id": self.submission_id,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class SubmissionRow:
    """Typed mirror of one `student_submissions` row."""

    id: int
    student_id: str
    milestone_id: str
    pseudocode: str
    status: str
    attempts: int
    expected_output: tuple[str, ...] = ()
    claimed_output: str | None = None
    inputs: dict[str, Any] = field(default_factory=dict)
    advice: str | None = None
    resolution: str | None = None
    requires_permission: bool = False
    last_error_kind: str | None = None


@dataclass(frozen=True)
class MilestoneRow:
    """Typed mirror of one `student_milestones` row (post-update view)."""

    state: str
    requires_permission: bool
    unlocked: bool
    consecutive_failures: int
    total_attempts: int


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _load_json(text: Any, default: Any) -> Any:
    """Parse a JSON string, passing through already-parsed objects untouched."""
    if text is None or text == "":
        return default
    if isinstance(text, (dict, list, int, float, bool)):
        return text
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# Persistence — the runtime's tables live alongside the lesson tables in the
# same lessons.db; migrations are additive CREATE TABLE IF NOT EXISTS, so an
# existing database is never touched.
# --------------------------------------------------------------------------- #

CREATE_SUBMISSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS student_submissions (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id           TEXT NOT NULL,
    milestone_id         TEXT NOT NULL,
    tier                 TEXT,
    pseudocode           TEXT NOT NULL,
    expected_output      TEXT,
    claimed_output       TEXT,
    inputs               TEXT,
    requires_permission  INTEGER NOT NULL DEFAULT 0,
    status               TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'processing', 'auto_passed', 'auto_failed', 'escalated', 'resolved')),
    attempts             INTEGER NOT NULL DEFAULT 0,
    advice               TEXT,
    resolution           TEXT,
    last_error_kind      TEXT,
    created_at           TEXT NOT NULL,
    updated_at           TEXT NOT NULL
)
"""

CREATE_MILESTONES_TABLE = """
CREATE TABLE IF NOT EXISTS student_milestones (
    student_id           TEXT NOT NULL,
    milestone_id         TEXT NOT NULL,
    title                TEXT,
    state                TEXT NOT NULL DEFAULT 'in_progress'
        CHECK (state IN ('in_progress', 'completed', 'blocked', 'advanced')),
    requires_permission  INTEGER NOT NULL DEFAULT 0,
    unlocked             INTEGER NOT NULL DEFAULT 0,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    total_attempts       INTEGER NOT NULL DEFAULT 0,
    updated_at           TEXT NOT NULL,
    PRIMARY KEY (student_id, milestone_id)
)
"""

CREATE_DECISIONS_TABLE = """
CREATE TABLE IF NOT EXISTS agent_decisions (
    decision_id     TEXT PRIMARY KEY,
    submission_id   INTEGER,
    student_id      TEXT NOT NULL,
    milestone_id    TEXT NOT NULL,
    category        TEXT NOT NULL
        CHECK (category IN ('stuck_loop', 'hallucination_risk', 'permission_gate')),
    severity        TEXT NOT NULL,
    brief_json      TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'resolved')),
    resolution      TEXT CHECK (resolution IN ('approve', 'reject', 'advise')),
    human_input     TEXT,
    created_at      TEXT NOT NULL,
    resolved_at     TEXT
)
"""

CREATE_TELEMETRY_TABLE = """
CREATE TABLE IF NOT EXISTS agent_telemetry (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    at            TEXT NOT NULL,
    level         TEXT NOT NULL DEFAULT 'info',
    event         TEXT NOT NULL,
    payload_json  TEXT NOT NULL DEFAULT '{}'
)
"""

SUBMISSION_SELECT = """
SELECT id, student_id, milestone_id, tier, pseudocode, expected_output, claimed_output,
       inputs, requires_permission, status, attempts, advice, resolution, last_error_kind
FROM student_submissions
"""


def _submission_from_row(row: Any) -> SubmissionRow:
    expected = tuple(str(item) for item in _load_json(row["expected_output"], []))
    return SubmissionRow(
        id=int(row["id"]),
        student_id=str(row["student_id"]),
        milestone_id=str(row["milestone_id"]),
        pseudocode=str(row["pseudocode"]),
        status=str(row["status"]),
        attempts=int(row["attempts"]),
        expected_output=expected,
        claimed_output=row["claimed_output"],
        inputs=_load_json(row["inputs"], {}),
        advice=row["advice"],
        resolution=row["resolution"],
        requires_permission=bool(row["requires_permission"]),
        last_error_kind=row["last_error_kind"],
    )


# --------------------------------------------------------------------------- #
# The Strands tool surface. These are the ONLY capabilities the escalation
# reasoning loop can route to. They read the engine from a ContextVar — the
# same pattern app/main.py uses for missions — so the model can never hand a
# tool the wrong student or invent a database handle.
# --------------------------------------------------------------------------- #

_CURRENT_ENGINE: contextvars.ContextVar["AutonomousAgentEngine | None"] = \
    contextvars.ContextVar("current_learning_engine", default=None)


def _engine_from_context() -> "AutonomousAgentEngine":
    engine = _CURRENT_ENGINE.get()
    if engine is None:
        raise RuntimeError("No learning engine is bound to this context.")
    return engine


if STRANDS_AVAILABLE:

    @_strands_tool
    async def run_code_tests(
        pseudocode: str, expected_output: str = "", inputs_json: str = ""
    ) -> dict:
        """
        Run pseudo-code through the bounded interpreter as unit tests.

        Args:
            pseudocode: The student's pseudo-code to execute.
            expected_output: Newline-separated output lines the code should print.
                Leave empty to only check that the code runs clean.
            inputs_json: Optional JSON object of variable bindings, e.g. '{"n": 5}'.

        Returns:
            A dict with "ok", per-run "output", "steps" and any "error".
        """
        try:
            inputs = _load_json(inputs_json, {}) or {}
            if not isinstance(inputs, dict):
                return {"ok": False, "error": ("inputs", "inputs_json must be a JSON object")}
            execution = execute_pseudocode(pseudocode, inputs=inputs)
            expected = [line for line in expected_output.splitlines() if line.strip()]
            return {
                "ok": bool(execution.ok) and outputs_match(execution.output, expected),
                "output": execution.output,
                "steps": execution.steps,
                "error": execution.error,
                "line": execution.line,
                "matches_expected": outputs_match(execution.output, expected) if expected else None,
            }
        except Exception as exc:  # noqa: BLE001 - tool results go back to the model
            return {"ok": False, "error": ("tool", f"run_code_tests failed: {exc!r}")}

    @_strands_tool
    async def fetch_student_history(student_id: str, limit: int = 10) -> dict:
        """
        Read a student's recent submissions and milestone states from the database.

        Args:
            student_id: The student to look up.
            limit: Maximum number of recent submissions to return (1-50).

        Returns:
            A dict with "submissions" (most recent first) and "milestones".
        """
        try:
            engine = _engine_from_context()
            bounded = max(1, min(int(limit), 50))
            async with aiosqlite.connect(engine.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    SUBMISSION_SELECT + " WHERE student_id = ? ORDER BY id DESC LIMIT ?",
                    (student_id, bounded),
                ) as cursor:
                    rows = await cursor.fetchall()
                async with db.execute(
                    "SELECT * FROM student_milestones WHERE student_id = ? ORDER BY updated_at DESC",
                    (student_id,),
                ) as cursor:
                    milestones = await cursor.fetchall()
            return {
                "submissions": [
                    {
                        "id": r["id"], "milestone_id": r["milestone_id"],
                        "status": r["status"], "attempts": r["attempts"],
                        "advice": r["advice"],
                    }
                    for r in rows
                ],
                "milestones": [dict(r) for r in milestones],
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": f"fetch_student_history failed: {exc!r}"}

    @_strands_tool
    async def submit_decision_brief(
        category: str,
        title: str,
        summary: str,
        evidence_json: str = "",
        recommended_action: str = "",
    ) -> dict:
        """
        THE human gate: pause background execution and surface a structured
        Escalation Decision Brief to the educator/mentor on duty.

        Call this exactly once when the evidence shows a judgement call a
        human must make. The agent then waits; a human resolves the brief
        through resolve_decision and the loop resumes autonomously.

        Args:
            category: One of "stuck_loop", "hallucination_risk", "permission_gate".
            title: Under ten words, naming the concrete decision.
            summary: Two or three sentences a teacher can act on, grounded in evidence.
            evidence_json: Optional JSON object of supporting facts.
            recommended_action: One of "approve", "reject", "advise", if the evidence
                clearly favours one.

        Returns:
            A dict with "status": "paused" and the new "decision_id", or a refusal.
        """
        try:
            engine = _engine_from_context()
            return await engine.agent_submit_decision(
                category=category, title=title, summary=summary,
                evidence=_load_json(evidence_json, {}) or {},
                recommended_action=recommended_action or None,
            )
        except Exception as exc:  # noqa: BLE001
            return {"status": "refused", "reason": f"submit_decision_brief failed: {exc!r}"}

    @_strands_tool
    async def unlock_advanced_module(student_id: str, milestone_id: str) -> dict:
        """
        Unlock a gated advanced module for a student.

        HUMAN-GATED TOOL: an InterventionHandler denies the call unless a
        mentor has already approved this exact unlock through a resolved
        permission decision. There is no prompt that unlocks it.

        Args:
            student_id: The student to unlock the module for.
            milestone_id: The advanced milestone to unlock.

        Returns:
            A dict with "unlocked": true on success, or a refusal explaining the gate.
        """
        try:
            engine = _engine_from_context()
            return await engine.apply_unlock(student_id=student_id, milestone_id=milestone_id)
        except Exception as exc:  # noqa: BLE001
            return {"unlocked": False, "reason": f"unlock_advanced_module failed: {exc!r}"}

    _AGENT_TOOLS = [run_code_tests, fetch_student_history, submit_decision_brief, unlock_advanced_module]

    class PermissionGateIntervention(_InterventionHandler):
        """
        Native Strands HITL enforcement at the tool-routing layer.

        `unlock_advanced_module` is the one tool whose effect cannot be
        undone by regenerating text, so the intervention checks the human's
        recorded decision before Strands executes it: approved in
        `agent_decisions` -> Proceed; anything else -> Deny with routing
        instructions back to `submit_decision_brief`. The model cannot
        reason its way past this — the refusal happens below the model.
        """

        @property
        def name(self) -> str:  # noqa: D102 - InterventionHandler protocol
            return "permission-gate"

        async def before_tool_call(self, event: Any) -> Any:
            tool_use = getattr(event, "tool_use", None) or {}
            selected = getattr(event, "selected_tool", None) or tool_use.get("name")
            if selected != "unlock_advanced_module":
                return _Proceed()
            payload = tool_use.get("input", {}) if isinstance(tool_use, dict) \
                else getattr(tool_use, "input", {}) or {}
            student_id = str(payload.get("student_id", ""))
            milestone_id = str(payload.get("milestone_id", ""))
            if await _engine_from_context().has_approved_permission(student_id, milestone_id):
                return _Proceed(
                    reason=f"Mentor approved unlocking '{milestone_id}' for '{student_id}'."
                )
            return _Deny(
                reason=(
                    "No human approval is on record for this unlock. Call "
                    "submit_decision_brief with category 'permission_gate' and stop — "
                    "the module stays locked until a mentor resolves it."
                )
            )

else:  # Strands unavailable — the runtime still runs, deterministically.

    _AGENT_TOOLS = []

    class PermissionGateIntervention:  # type: ignore[no-redef]
        """Placeholder when Strands is absent; the DB-level gate still applies."""

        name = "permission-gate"


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #


class AutonomousAgentEngine:
    """
    The autonomous background agent over the lessons.db learning stream.

    Lifecycle: `start()` migrates the schema and spawns one worker task;
    the task drains new submissions every `poll_interval` seconds forever.
    Submissions arrive via `ingest_submission` (from routes, LTI callbacks,
    or any producer); the agent takes it from there. `close()` stops the
    worker cleanly. Nothing in the class blocks on a human except an
    explicit `resolve_decision` call.

    Concurrency model: one worker, sequential processing, and a decision
    lock shared with `resolve_decision` so a human resolution can never
    interleave with the state transition it resolves.
    """

    def __init__(
        self,
        db_path: str | Path = "data/lessons.db",
        *,
        poll_interval: float = 2.0,
        batch_size: int = 20,
        reasoning_timeout: float = 25.0,
        agent_id: str = "curriculum-learning-agent",
        session_dir: str | Path = "data/agent_sessions",
    ) -> None:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.db_path = Path(db_path)
        self.poll_interval = poll_interval
        self.batch_size = batch_size
        self.reasoning_timeout = reasoning_timeout
        self.agent_id = agent_id
        self.session_dir = Path(session_dir)
        self._task: asyncio.Task[None] | None = None
        self._decision_lock = asyncio.Lock()
        self._agent: Any = None  # cached Strands agent, built lazily
        # Context for the escalation currently being composed; the
        # submit_decision_brief tool reads it to bind the brief to the
        # submission/student being paused. Set only inside the reasoning call.
        self._pending_escalation: dict[str, Any] | None = None
        self._agent_decision_id: str | None = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def start(self, *, run_worker: bool = True) -> None:
        """
        Migrate the schema (additive only) and start the background loop.

        Idempotent: calling start twice neither duplicates the worker nor
        rewrites existing rows. This is the point where the agent becomes
        autonomous — everything before it is configuration.

        `run_worker=False` migrates the schema but leaves the queue to be
        drained by explicit `drain_once()` calls — for embedders that own
        their own scheduling and for deterministic tests.
        """
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(CREATE_SUBMISSIONS_TABLE)
            await db.execute(CREATE_MILESTONES_TABLE)
            await db.execute(CREATE_DECISIONS_TABLE)
            await db.execute(CREATE_TELEMETRY_TABLE)
            await db.commit()
        if run_worker and (self._task is None or self._task.done()):
            self._task = asyncio.create_task(
                self._run_loop(), name=f"{self.agent_id}-loop"
            )
        _runtime_log.info(
            "Autonomous learning agent started (db=%s, poll=%.1fs, reasoning=%s, worker=%s).",
            self.db_path, self.poll_interval,
            "strands" if self.reasoning_enabled else "deterministic", run_worker,
        )

    async def close(self) -> None:
        """Stop the background loop and wait for the current item to settle."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        _runtime_log.info("Autonomous learning agent stopped.")

    @property
    def reasoning_enabled(self) -> bool:
        """True when escalation briefs can be composed by a Strands agent."""
        return _resolve_model() is not None

    async def _run_loop(self) -> None:
        """
        The autonomous loop: drain, sleep, repeat — forever.

        Autonomous boundary: everything in here runs without a human. Any
        exception is logged to telemetry and absorbed so one bad submission
        can never kill the loop; the loop only exits when cancelled by
        `close()`.
        """
        while True:
            try:
                processed = await self.drain_once()
                if processed == 0:
                    await asyncio.sleep(self.poll_interval)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - the loop must survive anything
                await self._telemetry("loop_error", level="error", error=repr(exc))
                await asyncio.sleep(min(self.poll_interval * 4, 30.0))

    async def drain_once(self) -> int:
        """
        Reclaim stale in-flight rows, then atomically claim and process up
        to `batch_size` pending submissions. Returns the number processed.

        The claim is a single IMMEDIATE transaction (select + mark), so two
        concurrent drains can never pick up the same submission — which is
        what prevents one paused student from opening two identical briefs.
        """
        await self._reclaim_stale()
        submissions = await self._claim_batch()

        processed = 0
        for submission in submissions:
            try:
                await self._process_submission(submission)
            except Exception as exc:  # noqa: BLE001 - one bad submission never stops the queue
                await self._telemetry(
                    "submission_processing_error", level="error",
                    submission_id=submission.id, error=repr(exc),
                )
                await self._set_submission_status(submission.id, "pending")
            else:
                processed += 1
        return processed

    async def _reclaim_stale(self) -> None:
        """Return rows a dead worker left in 'processing' to the queue."""
        cutoff = datetime.now(timezone.utc).timestamp() - PROCESSING_STALE_SECONDS
        stale_cut = datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE student_submissions SET status = 'pending', updated_at = ? "
                "WHERE status = 'processing' AND updated_at < ?",
                (_utcnow(), stale_cut),
            )
            await db.commit()

    async def _claim_batch(self) -> list[SubmissionRow]:
        """
        Atomically mark up to `batch_size` pending rows as 'processing' and
        return them. BEGIN IMMEDIATE serialises concurrent workers: the
        second worker's claim starts only after the first commits, so rows
        marked by the first are no longer pending.
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("BEGIN IMMEDIATE")
            try:
                async with db.execute(
                    SUBMISSION_SELECT + " WHERE status = 'pending' ORDER BY id LIMIT ?",
                    (self.batch_size,),
                ) as cursor:
                    rows = await cursor.fetchall()
                now = _utcnow()
                for row in rows:
                    await db.execute(
                        "UPDATE student_submissions SET status = 'processing', updated_at = ? "
                        "WHERE id = ?",
                        (now, row["id"]),
                    )
                await db.commit()
            except BaseException:
                await db.rollback()
                raise
        return [_submission_from_row(row) for row in rows]

    # ------------------------------------------------------------------ #
    # Ingestion — the event stream's front door
    # ------------------------------------------------------------------ #

    async def ingest_submission(
        self,
        student_id: str,
        milestone_id: str,
        pseudocode: str,
        *,
        expected_output: list[str] | None = None,
        claimed_output: str | None = None,
        inputs: dict[str, Any] | None = None,
        tier: str | None = None,
        requires_permission: bool = False,
    ) -> int:
        """
        Enqueue one student submission / learning event for silent
        background processing. Returns the submission id immediately —
        callers (routes, LTI callbacks, student UIs) are never blocked by
        the agent.

        Autonomous boundary: ingestion is fire-and-forget by design. No
        validation beyond "the columns must exist" is done here; failures
        are normal assessment outcomes, and only `resolve_decision` needs
        the caller again.
        """
        if not student_id or not milestone_id:
            raise ValueError("student_id and milestone_id are required")
        if not pseudocode or not pseudocode.strip():
            raise ValueError("pseudocode is required")
        expected = [str(line) for line in (expected_output or [])]
        now = _utcnow()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO student_milestones
                    (student_id, milestone_id, state, requires_permission, updated_at)
                VALUES (?, ?, 'in_progress', ?, ?)
                ON CONFLICT(student_id, milestone_id) DO UPDATE SET
                    requires_permission = CASE WHEN excluded.requires_permission = 1
                        THEN 1 ELSE student_milestones.requires_permission END,
                    updated_at = excluded.updated_at
                """,
                (student_id, milestone_id, 1 if requires_permission else 0, now),
            )
            cursor = await db.execute(
                """
                INSERT INTO student_submissions
                    (student_id, milestone_id, tier, pseudocode, expected_output,
                     claimed_output, inputs, requires_permission, status,
                     created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    student_id, milestone_id, tier, pseudocode,
                    _dump_json(expected) if expected else None,
                    claimed_output,
                    _dump_json(inputs) if inputs else None,
                    1 if requires_permission else 0,
                    now, now,
                ),
            )
            submission_id = int(cursor.lastrowid or 0)
            await db.commit()
        await self._telemetry(
            "submission_ingested", submission_id=submission_id,
            student_id=student_id, milestone_id=milestone_id,
        )
        return submission_id

    # ------------------------------------------------------------------ #
    # THE HITL DECISION GATE
    # ------------------------------------------------------------------ #

    async def resolve_decision(
        self,
        decision_id: str,
        human_input: str,
        *,
        resolution: DecisionResolution = "approve",
    ) -> DecisionBrief:
        """
        THE hook a mentor/educator calls to resume the paused agent.

        Applies the human's verdict to the paused work, closes the brief,
        writes a `human_intervention` telemetry event, and — where the
        verdict re-queues work — returns the submission to the pending
        queue so the background loop picks it up within `poll_interval`.

        Args:
            decision_id: Id of an open Escalation Decision Brief.
            human_input: The mentor's free-text note; stored verbatim and
                re-injected as advice when resolution is "advise".
            resolution: "approve" (accept the agent's escalation position /
                grant the permission / confirm the artefact), "reject"
                (decline it — the paused work stays stopped), or "advise"
                (inject guidance and let the agent continue autonomously).

        Returns:
            The resolved DecisionBrief.

        Raises:
            LookupError: the id is unknown or the brief is already resolved.
            ValueError: the resolution is not one of the three verbs.
        """
        if resolution not in ("approve", "reject", "advise"):
            raise ValueError(f"resolution must be approve/reject/advise, got {resolution!r}")

        async with self._decision_lock:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT * FROM agent_decisions WHERE decision_id = ? AND status = 'open'",
                    (decision_id,),
                ) as cursor:
                    row = await cursor.fetchone()
                if row is None:
                    raise LookupError(
                        f"Unknown or already-resolved decision {decision_id!r}."
                    )
                brief = _brief_from_row(row)

                await db.execute(
                    "UPDATE agent_decisions SET status = 'resolved', resolution = ?, "
                    "human_input = ?, resolved_at = ? WHERE decision_id = ?",
                    (resolution, human_input, _utcnow(), decision_id),
                )
                await self._apply_resolution(db, brief, resolution, human_input)
                await db.commit()

        await self._telemetry(
            "decision_resolved", level="warning",
            decision_id=decision_id, category=brief.category,
            resolution=resolution, human_input=human_input,
            human_intervention=True,
        )
        resolved = await self.get_decision(decision_id)
        assert resolved is not None  # it was just written
        return resolved

    async def _apply_resolution(
        self,
        db: aiosqlite.Connection,
        brief: DecisionBrief,
        resolution: DecisionResolution,
        human_input: str,
    ) -> None:
        """
        Translate a human verdict into state transitions.

        The boundary stays intact on this side too: applying the human's
        decision is mechanical, so the agent does it autonomously — a
        mentor approves once, the runtime does all the bookkeeping.
        """
        submission_id = brief.submission_id
        now = _utcnow()

        async def set_submission(status: str, *, advice: str | None = None,
                                 requeue: bool = False) -> None:
            if submission_id is None:
                return
            final_status = "pending" if requeue else status
            await db.execute(
                "UPDATE student_submissions SET status = ?, advice = COALESCE(?, advice), "
                "resolution = ?, updated_at = ? WHERE id = ?",
                (final_status, advice, resolution, now, submission_id),
            )

        async def set_milestone(state: str, *, reset_failures: bool = False,
                                unlocked: bool | None = None) -> None:
            await db.execute(
                """
                UPDATE student_milestones SET state = ?, updated_at = ?,
                    consecutive_failures = CASE WHEN ? = 1 THEN 0 ELSE consecutive_failures END,
                    unlocked = CASE WHEN ? IS NOT NULL THEN ? ELSE unlocked END
                WHERE student_id = ? AND milestone_id = ?
                """,
                (
                    state, now, 1 if reset_failures else 0,
                    1 if unlocked is not None else None, 1 if unlocked else 0,
                    brief.student_id, brief.milestone_id,
                ),
            )

        if brief.category == "permission_gate":
            if resolution == "approve":
                # The human granted the unlock; applying it is agent work.
                await set_milestone("advanced", reset_failures=True, unlocked=True)
                await set_submission("resolved")
            elif resolution == "reject":
                await set_milestone("in_progress")  # stays locked, stays visible
                await set_submission("resolved")
            else:  # advise: keep locked, hand the guidance back into the loop
                await set_milestone("in_progress")
                await set_submission("pending", advice=human_input, requeue=True)

        elif brief.category == "stuck_loop":
            if resolution == "approve":
                # Mentor believes the student is ready to keep going: clear
                # the stuck counter and resume autonomous support.
                await set_milestone("in_progress", reset_failures=True)
                await set_submission("pending", requeue=True)
            elif resolution == "reject":
                await set_milestone("blocked")
                await set_submission("resolved")
            else:  # advise: inject the hint, keep counting — three more
                # failures re-open the gate honestly.
                await set_milestone("in_progress")
                await set_submission("pending", advice=human_input, requeue=True)

        else:  # hallucination_risk
            if resolution == "approve":
                # The human confirms the claimed output is the intended
                # truth; the run counts as passed, flagged as human-verified.
                await set_milestone("in_progress", reset_failures=True)
                await set_submission("auto_passed",
                                     advice=f"Human-verified override: {human_input}")
            elif resolution == "reject":
                await set_milestone("in_progress")
                await set_submission("auto_failed",
                                     advice=f"Human-confirmed mismatch: {human_input}")
            else:
                await set_milestone("in_progress")
                await set_submission("pending", advice=human_input, requeue=True)

    async def open_decisions(self) -> list[DecisionBrief]:
        """Every unresolved brief, oldest first — the mentor's work queue."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM agent_decisions WHERE status = 'open' ORDER BY created_at"
            ) as cursor:
                rows = await cursor.fetchall()
        return [_brief_from_row(row) for row in rows]

    async def get_decision(self, decision_id: str) -> DecisionBrief | None:
        """One brief by id, open or resolved."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM agent_decisions WHERE decision_id = ?", (decision_id,)
            ) as cursor:
                row = await cursor.fetchone()
        return _brief_from_row(row) if row is not None else None

    async def has_approved_permission(self, student_id: str, milestone_id: str) -> bool:
        """True once a mentor has approved this exact unlock (the gate's memory)."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """
                SELECT 1 FROM agent_decisions
                WHERE category = 'permission_gate' AND student_id = ? AND milestone_id = ?
                  AND status = 'resolved' AND resolution = 'approve'
                LIMIT 1
                """,
                (student_id, milestone_id),
            ) as cursor:
                return await cursor.fetchone() is not None

    async def agent_submit_decision(
        self,
        *,
        category: str,
        title: str,
        summary: str,
        evidence: dict[str, Any],
        recommended_action: str | None,
    ) -> dict[str, Any]:
        """
        Tool-side entry point: the Strands escalation agent's way to open a
        decision. Validates the category, binds the paused
        submission/student from the pending escalation context, and writes
        the brief. Returns a refusal dict (not an exception) so the model
        sees actionable feedback instead of a crash.
        """
        if category not in ("stuck_loop", "hallucination_risk", "permission_gate"):
            return {
                "status": "refused",
                "reason": f"category must be stuck_loop/hallucination_risk/permission_gate, got {category!r}",
            }
        context = self._pending_escalation or {}
        recommended: DecisionResolution | None = (
            recommended_action if recommended_action in ("approve", "reject", "advise") else None
        )
        brief = await self._open_decision(
            category=category,  # type: ignore[arg-type]
            severity=str(context.get("severity", "medium")),
            student_id=str(context.get("student_id", "unknown")),
            milestone_id=str(context.get("milestone_id", "unknown")),
            submission_id=context.get("submission_id"),
            title=title or context.get("title", "Decision required"),
            summary=summary or context.get("summary", ""),
            evidence={**context.get("evidence", {}), **(evidence or {})},
            recommended_action=recommended,
        )
        self._agent_decision_id = brief.decision_id
        return {
            "status": "paused",
            "decision_id": brief.decision_id,
            "resolution_options": list(brief.options),
            "note": "Background execution is paused for this student until a mentor resolves the brief.",
        }

    async def apply_unlock(self, *, student_id: str, milestone_id: str) -> dict[str, Any]:
        """
        Tool-side entry point for `unlock_advanced_module`. The DB-level
        gate is checked again here so the unlock is safe even without the
        Strands intervention layer in the loop.
        """
        if not await self.has_approved_permission(student_id, milestone_id):
            return {
                "unlocked": False,
                "reason": "A mentor must approve this unlock first — open a permission_gate brief.",
            }
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE student_milestones SET unlocked = 1, state = 'advanced', updated_at = ? "
                "WHERE student_id = ? AND milestone_id = ?",
                (_utcnow(), student_id, milestone_id),
            )
            await db.commit()
        await self._telemetry(
            "advanced_module_unlocked", level="warning", human_intervention=True,
            student_id=student_id, milestone_id=milestone_id,
        )
        return {"unlocked": True, "student_id": student_id, "milestone_id": milestone_id}

    # ------------------------------------------------------------------ #
    # The silent pipeline — one submission, end to end
    # ------------------------------------------------------------------ #

    async def _process_submission(self, submission: SubmissionRow) -> None:
        """
        Autonomously process one submission through the full silent
        pipeline, then either commit the state transition or open the gate.

        AUTONOMOUS (steps 1-6): mark in-flight -> syntax analysis -> unit
        tests -> hallucination check -> milestone verdict -> autonomy
        decision -> commit (status + counters + telemetry). A human is
        never shown routine passes, routine failures, or struggling-band
        progress.

        ESCALATION (step 7): only the three decided triggers pause here,
        freeze the submission as 'escalated', and surface exactly one
        structured brief.

        GUIDED RE-RUN (the human said "advise"): the submission re-enters
        the queue carrying the mentor's advice. This is NOT a new student
        attempt, so its attempt counters are not re-counted and its
        escalation triggers are suppressed — the human just decided on this
        work item, and re-pausing on identical evidence would spam the
        mentor the autonomy rule forbids. The next *new* submission from
        the student is what re-arms the gate.
        """
        await self._set_submission_status(submission.id, "processing")
        guided = submission.resolution == "advise"

        # 1-2. Silent syntax analysis, then silent unit-test run.
        syntax = analyze_syntax(submission.pseudocode)
        tests = None
        if syntax.ok:
            case = UnitCase(
                name=f"submission-{submission.id}",
                inputs=dict(submission.inputs),
                expected_output=submission.expected_output,
            )
            tests = run_unit_tests(submission.pseudocode, [case])

        # 3. Silent hallucination check: claimed vs deterministic output.
        actual = tests.cases[0].actual_output if tests is not None and tests.cases else []
        claim = assess_claim(submission.claimed_output, list(actual))

        # 4. Silent milestone bookkeeping (the agent's own state memory).
        # A guided re-run is not a new attempt: count it only once, on the
        # submission's first pass through the pipeline.
        milestone = await self._record_attempt(
            submission, passed=bool(tests and tests.ok), count_attempt=not guided,
        )
        permission_approved = await self.has_approved_permission(
            submission.student_id, submission.milestone_id
        )
        profile = MilestoneProfile(
            student_id=submission.student_id,
            milestone_id=submission.milestone_id,
            consecutive_failures=milestone.consecutive_failures,
            total_attempts=milestone.total_attempts,
            requires_permission=submission.requires_permission or milestone.requires_permission,
            permission_approved=permission_approved,
        )
        verdict = evaluate_milestone(profile)

        # 5. THE boundary — one function, three escalation triggers.
        decision = decide_autonomy(
            syntax=syntax, tests=tests, claim=claim, milestone=verdict,
        )
        if guided and decision.mode == "escalate":
            decision = AutonomyDecision(
                mode="autonomous",
                outcome=decision.outcome if decision.outcome != "escalated" else "auto_failed",
                reason=(
                    "Guided re-run: a mentor has already resolved this work item with "
                    "advice, so the escalation gate stays closed until a new submission."
                ),
            )

        # 6-7. Commit autonomously, or pause and surface.
        if decision.mode == "autonomous":
            await self._commit_autonomous(submission, decision, syntax, tests, verdict)
            return
        await self._escalate(submission, decision, syntax, tests, claim, verdict, milestone)

    async def _record_attempt(
        self, submission: SubmissionRow, *, passed: bool, count_attempt: bool = True,
    ) -> MilestoneRow:
        """
        Upsert counters for this attempt and return the fresh milestone row.

        `count_attempt=False` (guided re-runs) updates nothing but the
        timestamp: the human's resolution already covered this attempt, and
        counting it twice would inflate the stuck gate.
        """
        now = _utcnow()
        async with aiosqlite.connect(self.db_path) as db:
            if count_attempt:
                await db.execute(
                    """
                    INSERT INTO student_milestones
                        (student_id, milestone_id, state, total_attempts,
                         consecutive_failures, updated_at)
                    VALUES (?, ?, 'in_progress', 1, ?, ?)
                    ON CONFLICT(student_id, milestone_id) DO UPDATE SET
                        total_attempts = student_milestones.total_attempts + 1,
                        consecutive_failures = CASE WHEN ? = 1
                            THEN 0 ELSE student_milestones.consecutive_failures + 1 END,
                        updated_at = excluded.updated_at
                    """,
                    (submission.student_id, submission.milestone_id,
                     0 if passed else 1, now, 1 if passed else 0),
                )
            else:
                await db.execute(
                    "UPDATE student_milestones SET updated_at = ? "
                    "WHERE student_id = ? AND milestone_id = ?",
                    (now, submission.student_id, submission.milestone_id),
                )
            await db.execute(
                "UPDATE student_submissions SET attempts = attempts + ?, "
                "last_error_kind = ?, updated_at = ? WHERE id = ?",
                (1 if count_attempt else 0, submission.last_error_kind, now, submission.id),
            )
            await db.commit()
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM student_milestones WHERE student_id = ? AND milestone_id = ?",
                (submission.student_id, submission.milestone_id),
            ) as cursor:
                row = await cursor.fetchone()
        return MilestoneRow(
            state=str(row["state"]),
            requires_permission=bool(row["requires_permission"]),
            unlocked=bool(row["unlocked"]),
            consecutive_failures=int(row["consecutive_failures"]),
            total_attempts=int(row["total_attempts"]),
        )

    async def _commit_autonomous(
        self,
        submission: SubmissionRow,
        decision: AutonomyDecision,
        syntax: SyntaxReport,
        tests: Any,
        verdict: MilestoneVerdict,
    ) -> None:
        """
        Commit a routine outcome silently.

        This is the "no human ever hears about it" path: status flip,
        milestone completion, telemetry. Telemetry is the audit trail that
        makes silence accountable — every autonomous transition is
        recorded even though nobody is notified.
        """
        status = decision.outcome  # auto_passed | auto_failed
        now = _utcnow()
        milestone_state = "completed" if status == "auto_passed" else "in_progress"
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE student_submissions SET status = ?, updated_at = ? WHERE id = ?",
                (status, now, submission.id),
            )
            # Never downgrade a milestone out of 'blocked' silently — only a
            # human resolution can unblock (see _apply_resolution).
            await db.execute(
                """
                UPDATE student_milestones SET state = CASE
                        WHEN state = 'blocked' THEN 'blocked'
                        WHEN state = 'advanced' THEN 'advanced'
                        ELSE ? END,
                    updated_at = ?
                WHERE student_id = ? AND milestone_id = ?
                """,
                (milestone_state, now, submission.student_id, submission.milestone_id),
            )
            await db.commit()

        await self._telemetry(
            "submission_auto_processed",
            submission_id=submission.id,
            student_id=submission.student_id,
            milestone_id=submission.milestone_id,
            outcome=status,
            syntax_ok=syntax.ok,
            tests_passed=(tests.passed if tests is not None else None),
            tests_failed=(tests.failed if tests is not None else None),
            verdict=verdict.state,
            reason=decision.reason,
        )

    async def _escalate(
        self,
        submission: SubmissionRow,
        decision: AutonomyDecision,
        syntax: SyntaxReport,
        tests: Any,
        claim: ClaimAssessment,
        verdict: MilestoneVerdict,
        milestone_row: MilestoneRow,
    ) -> DecisionBrief:
        """
        Pause the affected work and surface exactly one Escalation Decision
        Brief. The submission freezes as 'escalated' and the milestone is
        marked blocked (for progress gates) until `resolve_decision` runs.

        The brief narrative is composed by the Strands escalation agent
        when a model provider is configured (tools routed, state carried,
        permission tool gated by the intervention handler); otherwise the
        deterministic composer below builds the same structure from the
        machine evidence. Either way the *pause* and the *gate* are engine
        behaviour, not model behaviour.
        """
        now = _utcnow()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE student_submissions SET status = 'escalated', updated_at = ? WHERE id = ?",
                (now, submission.id),
            )
            if decision.category in ("stuck_loop", "hallucination_risk"):
                await db.execute(
                    "UPDATE student_milestones SET state = 'blocked', updated_at = ? "
                    "WHERE student_id = ? AND milestone_id = ? AND state != 'advanced'",
                    (now, submission.student_id, submission.milestone_id),
                )
            await db.commit()

        evidence = _build_evidence(submission, syntax, tests, claim, verdict, milestone_row)
        fallback_title, fallback_summary = _compose_brief_text(decision, evidence)

        context = {
            "student_id": submission.student_id,
            "milestone_id": submission.milestone_id,
            "submission_id": submission.id,
            "severity": decision.severity,
            "title": fallback_title,
            "summary": fallback_summary,
            "evidence": evidence,
        }

        brief: DecisionBrief | None = None
        composed_by = "deterministic"
        if self.reasoning_enabled:
            brief = await self._compose_brief_with_agent(decision, context)
            if brief is not None:
                composed_by = "strands"
        if brief is None:
            brief = await self._open_decision(
                category=decision.category or "stuck_loop",
                severity=decision.severity,
                student_id=submission.student_id,
                milestone_id=submission.milestone_id,
                submission_id=submission.id,
                title=fallback_title,
                summary=fallback_summary,
                evidence=evidence,
                recommended_action=_recommended_action_for(decision.category),
            )

        await self._telemetry(
            "escalation_raised", level="warning",
            decision_id=brief.decision_id, category=brief.category,
            submission_id=submission.id,
            student_id=submission.student_id,
            milestone_id=submission.milestone_id,
            composed_by=composed_by,
        )
        return brief

    async def _open_decision(
        self,
        *,
        category: DecisionCategory,
        severity: str,
        student_id: str,
        milestone_id: str,
        submission_id: int | None,
        title: str,
        summary: str,
        evidence: dict[str, Any],
        recommended_action: DecisionResolution | None,
    ) -> DecisionBrief:
        """Write one open decision row (the pause is durable across restarts)."""
        brief = DecisionBrief(
            decision_id=uuid.uuid4().hex,
            category=category,
            severity=severity if severity in ("low", "medium", "high") else "medium",
            student_id=student_id,
            milestone_id=milestone_id,
            title=title,
            summary=summary,
            evidence=evidence,
            recommended_action=recommended_action,
            submission_id=submission_id,
            created_at=_utcnow(),
        )
        async with self._decision_lock:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    INSERT INTO agent_decisions
                        (decision_id, submission_id, student_id, milestone_id, category,
                         severity, brief_json, status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'open', ?)
                    """,
                    (
                        brief.decision_id, submission_id, student_id, milestone_id,
                        category, brief.severity, _dump_json(brief.to_dict()), brief.created_at,
                    ),
                )
                await db.commit()
        return brief

    # ------------------------------------------------------------------ #
    # Strands escalation reasoning (optional narrative layer)
    # ------------------------------------------------------------------ #

    def _build_agent(self) -> Any:
        """
        Build (once) the Strands escalation agent.

        Natively wired: tools are routed by the model through Strands'
        event loop, `AgentState` + `FileSessionManager` give the agent
        durable state memory across restarts, and the
        `PermissionGateIntervention` hooks the loop so the gated tool is
        denied below the model. Returns None when no provider is usable.
        """
        if not STRANDS_AVAILABLE or self._agent is not None:
            return self._agent
        model = _resolve_model()
        if model is None:
            return None
        try:
            self.session_dir.mkdir(parents=True, exist_ok=True)
            self._agent = _StrandsAgent(
                model=model,
                tools=_AGENT_TOOLS,
                system_prompt=ESCALATION_SYSTEM_PROMPT,
                callback_handler=None,  # chain-of-thought stays out of surfaces
                state=_AgentState({"agent_id": self.agent_id, "role": "escalation-review"}),
                session_manager=_FileSessionManager(
                    f"{self.agent_id}-escalations",
                    storage_dir=str(self.session_dir),
                ),
                interventions=[PermissionGateIntervention()],
            )
        except Exception as exc:  # noqa: BLE001 - a broken agent must not break the loop
            _runtime_log.warning("Could not build the Strands escalation agent: %s", exc)
            self._agent = None
        return self._agent

    async def _compose_brief_with_agent(
        self, decision: AutonomyDecision, context: dict[str, Any]
    ) -> DecisionBrief | None:
        """
        Ask the Strands agent to compose the brief, routing through its
        tools. Returns None (and falls back to the deterministic composer)
        on any failure — the pause itself never depends on the model.
        """
        agent = self._build_agent()
        if agent is None:
            return None
        token = _CURRENT_ENGINE.set(self)
        self._pending_escalation = context
        self._agent_decision_id = None
        try:
            prompt = (
                "Compose the escalation decision brief now.\n"
                f"CATEGORY: {decision.category}\n"
                f"SEVERITY: {decision.severity}\n"
                f"REASON: {decision.reason}\n"
                f"EVIDENCE: {_dump_json(context.get('evidence', {}))}\n"
                "Call submit_decision_brief exactly once with a grounded title and "
                "summary, then stop."
            )
            await asyncio.wait_for(
                agent.invoke_async(prompt), timeout=self.reasoning_timeout
            )
            if self._agent_decision_id:
                return await self.get_decision(self._agent_decision_id)
            return None
        except asyncio.TimeoutError:
            await self._telemetry("reasoning_timeout", level="warning",
                                  category=decision.category)
            return None
        except Exception as exc:  # noqa: BLE001 - any model failure falls back
            await self._telemetry("reasoning_fallback", level="warning",
                                  category=decision.category, error=repr(exc))
            return None
        finally:
            self._pending_escalation = None
            _CURRENT_ENGINE.reset(token)

    # ------------------------------------------------------------------ #
    # Telemetry — the audit trail that makes silence accountable
    # ------------------------------------------------------------------ #

    async def _telemetry(self, event: str, *, level: str = "info", **payload: Any) -> None:
        """Record one telemetry event row and mirror it to the python log."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "INSERT INTO agent_telemetry (at, level, event, payload_json) VALUES (?, ?, ?, ?)",
                    (_utcnow(), level, event, _dump_json(payload)),
                )
                await db.commit()
        except Exception:  # noqa: BLE001 - telemetry must never break the loop
            _runtime_log.exception("Telemetry write failed for %s", event)
        log = _runtime_log.warning if level in ("warning", "error") else _runtime_log.debug
        log("telemetry[%s] %s %s", level, event, _dump_json(payload))

    async def _set_submission_status(self, submission_id: int, status: str) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE student_submissions SET status = ?, updated_at = ? WHERE id = ?",
                (status, _utcnow(), submission_id),
            )
            await db.commit()


def _brief_from_row(row: Any) -> DecisionBrief:
    """Rebuild a DecisionBrief from an agent_decisions row."""
    data = _load_json(row["brief_json"], {})
    return DecisionBrief(
        decision_id=str(row["decision_id"]),
        category=row["category"],
        severity=row["severity"],
        student_id=str(row["student_id"]),
        milestone_id=str(row["milestone_id"]),
        title=str(data.get("title", "Decision required")),
        summary=str(data.get("summary", "")),
        evidence=_load_json(data.get("evidence"), {}) or {},
        recommended_action=data.get("recommended_action"),
        options=tuple(data.get("options", ("approve", "reject", "advise"))),
        status=row["status"],
        resolution=row["resolution"],
        human_input=row["human_input"],
        submission_id=row["submission_id"],
        created_at=str(row["created_at"]),
    )


def _build_evidence(
    submission: SubmissionRow,
    syntax: SyntaxReport,
    tests: Any,
    claim: ClaimAssessment,
    verdict: MilestoneVerdict,
    milestone_row: MilestoneRow,
) -> dict[str, Any]:
    """
    Assemble the machine evidence attached to a brief.

    Deliberately compact: every entry is something a mentor can verify at a
    glance, not a log dump. This is what "grounded" means for the brief —
    the Strands agent is instructed to reason only from these facts.
    """
    cases = []
    if tests is not None:
        cases = [
            {
                "case": c.case,
                "passed": c.passed,
                "actual_output": list(c.actual_output),
                "expected_output": list(submission.expected_output),
                "error": list(c.error) if c.error else None,
                "steps": c.steps_used,
            }
            for c in tests.cases
        ]
    return {
        "submission_id": submission.id,
        "pseudocode": submission.pseudocode,
        "advice_already_given": submission.advice,
        "syntax_errors": [
            {"code": f.code, "message": f.message, "line": f.line}
            for f in syntax.errors
        ],
        "unit_tests": {"passed": getattr(tests, "passed", 0),
                       "failed": getattr(tests, "failed", 0),
                       "cases": cases},
        "claim": {
            "claimed_output": list(claim.claimed),
            "actual_output": list(claim.actual),
            "hallucination_risk": claim.hallucination_risk,
            "detail": claim.detail,
        },
        "milestone": {
            "state": verdict.state,
            "consecutive_failures": milestone_row.consecutive_failures,
            "total_attempts": milestone_row.total_attempts,
            "unlocked": milestone_row.unlocked,
            "requires_permission": milestone_row.requires_permission,
            "reason": verdict.reason,
        },
    }


def _compose_brief_text(
    decision: AutonomyDecision, evidence: dict[str, Any]
) -> tuple[str, str]:
    """
    Deterministic brief narrative for a category, grounded in evidence.

    Same structure the Strands path produces — title under ten words,
    summary a mentor can act on in under a minute — so the human surface
    does not change with the provider in use.
    """
    milestone = evidence.get("milestone", {})
    student = evidence.get("submission_id")
    if decision.category == "stuck_loop":
        title = "Student is stuck past the retry gate"
        summary = (
            f"Submission #{student} failed again and the student has now exceeded the "
            f"consecutive-failure gate ({milestone.get('reason', '')}). More automatic "
            f"retries are effort, not progress. Approve to resume autonomous support, "
            f"advise to inject a hint and continue, or reject to stop and redirect."
        )
    elif decision.category == "hallucination_risk":
        title = "Claimed output contradicts execution"
        summary = (
            f"Submission #{student} claims output that deterministic execution does not "
            f"produce ({evidence.get('claim', {}).get('detail', 'mismatch')}). Either the "
            f"material or the claim cannot be trusted. Approve to confirm the claim as "
            f"the intended truth, reject to confirm the mismatch, or advise."
        )
    else:  # permission_gate
        title = "Permission needed to unlock advanced module"
        summary = (
            f"Submission #{student} passed everything leading up to an advanced module. "
            f"The module stays locked until a mentor approves the unlock "
            f"({milestone.get('reason', '')}). Approve to unlock, reject to keep it "
            f"locked, or advise to send guidance first."
        )
    return title, summary


def _recommended_action_for(category: DecisionCategory | None) -> DecisionResolution | None:
    """The default verdict the runtime would pick, surfaced as a suggestion only."""
    return {
        "stuck_loop": "advise",
        "hallucination_risk": None,  # genuinely the human's call, no suggestion
        "permission_gate": "approve",
    }.get(category or "")


__all__ = [
    # Interpreter core (existing public surface)
    "DEFAULT_INPUTS",
    "INTERPRETER_ERRORS",
    "INTERPRETER_STEP_CAP",
    "PSEUDO_KEYWORDS",
    "code_lines",
    "execute_pseudocode",
    "leading_keyword",
    "normalise_output_line",
    "outputs_match",
    "run_pseudocode",
    # Autonomous runtime
    "AutonomousAgentEngine",
    "DecisionBrief",
    "DecisionCategory",
    "DecisionResolution",
    "MilestoneRow",
    "SubmissionRow",
    "STRANDS_AVAILABLE",
]
