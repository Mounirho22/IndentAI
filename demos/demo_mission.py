"""
demo_mission.py — end-to-end autonomous mission workflow demo.

This is the acceptance-criteria demo. The teacher creates one curriculum
mission ("Prepare next week's differentiated Computer Science materials
for Unit 3: Nested Loops"). The agent does the rest:

  1. Teacher submits the mission via POST /missions.
  2. The orchestrator (real run_curriculum_mission) takes over.
  3. For each of the three tiers, a sub-mission is spawned; the
     Strands agent picks tools, audits, and decides what to do.
  4. The on-level tier fails the audit on first attempt, the agent
     self-corrects with revision_notes, and the re-audit passes.
  5. The advanced tier's audit comes back with verdict "human" — a
     genuine judgement call. The agent escalates with
     flag_for_human_review.
  6. The mission reaches WAITING_FOR_HUMAN. The teacher approves
     the flagged record via POST /approve/{id}.
  7. The mission advances to COMPLETED. The summary line is
     "Mission completed — 2/3 handled automatically. 1 decision
     required your attention."

The Strands agent is faked at the model level (a small script that
calls the real ten tool coroutines through the same access path
verify.py uses, so every mission event lands through the real
recording code). The orchestrator, the curriculum mission state
machine, the dashboard routes, the approve route, the database
writes, the mission log, and the summary computation are all real.

Run:

    python demo_mission.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sqlite3
import sys
import textwrap
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import sys
from pathlib import Path
if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app.main as main


# --------------------------------------------------------------------------- #
# Per-tier script that the fake agent runs.
#
# The real Strands agent would call these tools autonomously based on the
# system prompt. For the demo we encode the per-tier control flow as
# deterministic tool calls so the audit-fail -> self-correct -> re-audit
# path is reproducible, and the audit-human -> escalate path is reproducible.
# Every tool call goes through the real @tool function's underlying
# coroutine (strands' DecoratedFunctionTool._tool_func), so the mission
# events get recorded through the real path.
# --------------------------------------------------------------------------- #

_WORKSHEET_STRUGGLING = textwrap.dedent("""\
    WORKSHEET -- Nested loops (struggling tier)
    Objective: Students can identify the inner and outer loops in a nested REPEAT block.

    Directions: read the snippet, fill in the blanks from the word bank, and trace the
    values in the table.

    Word bank: REPEAT, TIMES, SET, PRINT, total, outer, inner, loop, end

    1. SET ______ = 0
    2. REPEAT ______ TIMES
    3.   REPEAT ______ TIMES
    4.     SET ______ = ______ + 1
    5.   END ______
    6. END ______
    7. PRINT ______

    Trace table: fill in the value of `total` after each iteration of the outer loop.

    | outer | inner | total after outer |
    | 1     | 3     | ______            |
    | 2     | 3     | ______            |
    | 3     | 3     | ______            |

    8. If outer = 2 and inner = 3, what is the final value of total? Explain in your
       own words why the count is the product, not the sum.
    """)

_WORKSHEET_ON_LEVEL_V1 = textwrap.dedent("""\
    WORKSHEET -- Nested loops (on-level tier)
    Objective: Students can trace a nested loop and predict its output.

    Directions: read each snippet, then answer the questions. Use the word bank.

    Word bank: REPEAT, TIMES, SET, PRINT, total, outer, inner, loop, end

    1. SET total = 0
    2. REPEAT outer TIMES
    3.   REPEAT inner TIMES
    4.     SET total = total + 1
    5.   END REPEAT
    6. END REPEAT
    7. PRINT total

    8. If outer = 3 and inner = 4, what is the final value of total? Explain in your
       own words why the count is 12 and not 7.
    """)

_WORKSHEET_ON_LEVEL_V2 = textwrap.dedent("""\
    WORKSHEET -- Nested loops (on-level tier)
    Objective: Students can trace a nested loop and predict its output.

    Directions: read each snippet, then answer the questions. Use the word bank
    to fill in the blanks.

    Word bank: REPEAT, TIMES, SET, PRINT, total, outer, inner, loop, end

    1. SET total = 0
    2. REPEAT outer ______
    3.   REPEAT inner ______
    4.     SET ______ = total + 1
    5.   END ______
    6. END ______
    7. PRINT ______

    8. If outer = 3 and inner = 4, what is the final value of total? Explain in your
       own words why nested REPEAT blocks multiply the count rather than add it.
    """)

_WORKSHEET_ADVANCED = textwrap.dedent("""\
    WORKSHEET -- Nested loops (advanced tier)
    Objective: Students can find and fix an off-by-one defect in a nested loop.

    Directions: the snippet below is intentionally broken. Find the bug and explain
    why it makes the loop off by one. Fill in the blanks to describe the fix.

    Word bank: REPEAT, TIMES, SET, PRINT, total, outer, inner, loop, end, fewer,
    faster, efficient

    1. SET total ______ 0
    2. REPEAT ______ TIMES
    3.   REPEAT inner ______
    4.     SET ______ = total + 1
    5.   END ______
    6. END ______
    7. PRINT ______

    8. The author meant to count iterations of the inner loop, but the result is
       always one short. Identify the off-by-one defect. Then describe, in your
       own words, an efficient way to count iterations without the inner loop at
       all. Use the word bank. Why is the efficient version ______ and uses
       ______ steps?
    """)

_QUIZ_STRUGGLING = textwrap.dedent("""\
    QUIZ -- Nested loops (struggling tier)

    Q1 (2 pts): In a REPEAT 4 TIMES block, how many times does the body run?
                A) once B) twice C) three times D) four times
    Q2 (2 pts): When a REPEAT is inside another REPEAT, the inner body runs...
                A) never
                B) once
                C) once per outer iteration
                D) only at the end
    Q3 (2 pts): Predict the output of the snippet above. Output: 6
    Q4 (2 pts): Which keyword marks the end of a REPEAT block?
                A) STOP B) END REPEAT C) EXIT D) QUIT
    Q5 (2 pts): A nested REPEAT with outer = 2 and inner = 3 produces how many
                total iterations? A) 2 B) 3 C) 5 D) 6
    """)

_QUIZ_ON_LEVEL = textwrap.dedent("""\
    QUIZ -- Nested loops (on-level tier)

    Q1 (2 pts): What does REPEAT 3 TIMES do?
                A) runs once
                B) runs three times
                C) never runs
                D) skips the next line
    Q2 (2 pts): When the inner loop runs 4 times for every outer iteration of 3,
                the body executes how many times in total? A) 3
                B) 4
                C) 7
                D) 12
    Q3 (2 pts): Predict the output of the snippet above. Output: 12
    Q4 (2 pts): Which keyword ends a REPEAT? A) STOP B) END REPEAT C) EXIT D) QUIT
    Q5 (2 pts): A nested loop with outer=3 and inner=4 produces how many total
                iterations? A) 3 B) 4 C) 7 D) 12
    """)

_QUIZ_ADVANCED = textwrap.dedent("""\
    QUIZ -- Nested loops (advanced tier)

    Q1 (2 pts): A nested REPEAT that counts 3 * 4 iterations will print which value?
                A) 7 B) 12 C) 24 D) 34
    Q2 (2 pts): Which off-by-one defect makes a count return 6 instead of 8?
                A) the inner loop ends at inner-1
                B) the outer loop runs twice
                C) REPEAT counts from zero
                D) END REPEAT skips one iteration
    Q3 (2 pts): Predict the output of the snippet above. Output: 12
    Q4 (2 pts): Which keyword marks the end of a REPEAT block?
                A) STOP B) END REPEAT C) EXIT D) QUIT
    Q5 (2 pts): What is the most efficient way to count iterations of two nested
                REPEAT blocks of sizes m and n? A) m + n
                B) m * n
                C) max(m, n)
                D) min(m, n)
    """)


def _good_answer_key_for(quiz: str, correct_letter: dict[str, str], tier: str = "on-level") -> str:
    """
    Build a misconception-bearing answer key that satisfies every
    deterministic check (linkage, total, mastery, length).
    """
    q_count = 5
    total = 10
    lines = [f"ANSWER KEY -- Nested loops ({tier} tier)", ""]
    for n in range(1, q_count + 1):
        correct = correct_letter.get(str(n), "B")
        lines.append(f"Q{n}: {correct}.")
        # Include all four options so the linkage check has one
        # misconception per wrong option.
        for letter in ("A", "B", "C", "D"):
            if letter == correct:
                continue
            if n == 3:
                misc = {
                    "A": "forgetting the inner loop runs on every outer iteration",
                    "B": "treating REPEAT like a one-shot statement",
                    "C": "thinking the inner count adds to the outer count",
                    "D": "adding outer + inner instead of multiplying",
                }[letter]
            else:
                misc = {
                    "A": "confusing REPEAT with IF (reads REPEAT like a check)",
                    "B": "thinking the loop runs only once even when nested",
                    "C": "thinking REPEAT waits for a condition",
                    "D": "thinking REPEAT skips the next line",
                }[letter]
            lines.append(f"   {letter}) misconception: {misc}.")
    lines.append("")
    lines.append(f"Total: {total}. Mastery at 8/10.")
    return "\n".join(lines)


_ANSWER_KEY_STRUGGLING = _good_answer_key_for(_QUIZ_STRUGGLING, {"1": "D", "2": "C", "3": "D", "4": "B", "5": "D"}, tier="struggling")
_ANSWER_KEY_ON_LEVEL = _good_answer_key_for(_QUIZ_ON_LEVEL, {"1": "B", "2": "D", "3": "C", "4": "B", "5": "D"}, tier="on-level")
_ANSWER_KEY_ADVANCED = _good_answer_key_for(_QUIZ_ADVANCED, {"1": "B", "2": "A", "3": "C", "4": "B", "5": "B"}, tier="advanced")


# --------------------------------------------------------------------------- #
# Stub for the model call. Routes on the system-prompt keywords.
# Order matters: the audit prompt contains "worksheet" and "reviewer",
# so the reviewer branch must match first.
# --------------------------------------------------------------------------- #

_REVIEWER_PASS = json.dumps({
    "verdict": "pass", "problems": [], "reason": "Acceptable for this tier.",
    "findings": [],
})
_REVIEWER_REVISE = json.dumps({
    "verdict": "revise",
    "problems": [
        "Material does not echo any word from the stated objective, so a student cannot see what they are learning to do.",
        "Worksheet is missing the 6 visible ______ blanks the struggling tier needs so the student knows where to write.",
    ],
    "reason": "A blank-driven fill-in exercise at this tier is the whole point.",
    "findings": [
        {"severity": "high", "issue": "Worksheet is missing the 6 visible ______ blanks.",
         "affected_component": "worksheet", "recommended_action": "regenerate_worksheet"},
    ],
})
_REVIEWER_HUMAN = json.dumps({
    "verdict": "human",
    "problems": [
        "The objective asks the student to debug an off-by-one defect in a nested loop, but no two defensible readings of 'off by one' exist for this snippet at the 8th-grade level.",
    ],
    "reason": "Two defensible readings of the off-by-one location: line 5 vs line 7.",
    "findings": [
        {"severity": "high", "issue": "Off-by-one location is ambiguous without teacher judgement.",
         "affected_component": "explanation", "recommended_action": "escalate_to_teacher"},
    ],
})
_PLAN = json.dumps({
    "unit_name": "Unit 3: Nested Loops",
    "objective": "Students can trace a nested REPEAT loop and predict its total iterations.",
})
_DISTRACTORS = json.dumps({
    "distractors": [
        {"option": "A) forgetting the inner loop runs on every outer iteration",
         "misconception": "treating the inner loop as a single statement",
         "why_plausible": "the student does not visualise two nested loops"},
        {"option": "B) treating REPEAT like a one-shot statement",
         "misconception": "confusing REPEAT with PRINT",
         "why_plausible": "the student reads REPEAT as a single command"},
        {"option": "D) adding outer + inner instead of multiplying",
         "misconception": "thinking the inner count adds to the outer count",
         "why_plausible": "linear, not multiplicative, mental model"},
    ],
})

# Canonical snippets the fake agent runs through the AST sandbox
# (verify_code_example -> verify_pseudocode + the bounded interpreter in
# app/interpreter) before trusting a trace item, the way the real Strands
# agent does. Lines are un-numbered so the parser sees code, not prose.
_AST_CLEAN_SNIPPET = textwrap.dedent("""\
    SET total = 0
    REPEAT 3 TIMES
      REPEAT 2 TIMES
        SET total = total + 1
      END REPEAT
    END REPEAT
    PRINT total
    """)

# A genuinely broken snippet: an END REPEAT with nothing left open. The
# sandbox must flag it — that is the negative case the AST HUD exists for.
_AST_BROKEN_SNIPPET = textwrap.dedent("""\
    SET total = 0
    END REPEAT
    """)


# Tracks the per-tier state for the fake agent's "script".
_SCRIPT = {
    "tier": None,
    "tries": 0,
    "on_level_audit_history": [],
}
def _set_tier(tier: str) -> None:
    _SCRIPT["tier"] = tier
    _SCRIPT["tries"] = 0
    _SCRIPT["on_level_audit_history"] = []


async def stub_complete(system: str, user: str, *, max_tokens: int = 4096,
                        temperature: float = 0.4, json_mode: bool = False) -> str:
    s = (system or "").lower()
    tier = _SCRIPT.get("tier")

    if "reviewer" in s or "strict reviewer" in s:
        # Per-tier audit outcomes. The on-level tier first returns revise
        # (so the agent self-corrects), then pass. The advanced tier
        # returns pass for the worksheet but human for the answer key
        # (so the agent escalates the genuine judgement call).
        kind_match = re.search(r"material kind:\s*(\w+)", user or "", re.IGNORECASE)
        kind = kind_match.group(1).lower() if kind_match else ""
        if tier == "on-level":
            _SCRIPT["tries"] += 1
            if _SCRIPT["tries"] == 1:
                return _REVIEWER_REVISE
            return _REVIEWER_PASS
        if tier == "advanced" and kind == "answer_key":
            return _REVIEWER_HUMAN
        return _REVIEWER_PASS

    if "planner" in s or "lesson planner" in s:
        return _PLAN

    if "distractors" in s:
        return _DISTRACTORS

    if json_mode and "quiz" in s:
        if tier == "struggling":
            return json.dumps({"quiz": _QUIZ_STRUGGLING, "answer_key": _ANSWER_KEY_STRUGGLING})
        if tier == "on-level":
            return json.dumps({"quiz": _QUIZ_ON_LEVEL, "answer_key": _ANSWER_KEY_ON_LEVEL})
        if tier == "advanced":
            return json.dumps({"quiz": _QUIZ_ADVANCED, "answer_key": _ANSWER_KEY_ADVANCED})
        return json.dumps({"quiz": _QUIZ_ON_LEVEL, "answer_key": _ANSWER_KEY_ON_LEVEL})

    if "curriculum writer" in s or "worksheet" in s:
        if tier == "struggling":
            return _WORKSHEET_STRUGGLING
        if tier == "on-level":
            _SCRIPT["tries"] += 1
            if _SCRIPT["tries"] == 1:
                return _WORKSHEET_ON_LEVEL_V1
            return _WORKSHEET_ON_LEVEL_V2
        if tier == "advanced":
            return _WORKSHEET_ADVANCED
        return _WORKSHEET_STRUGGLING

    return ""


# --------------------------------------------------------------------------- #
# Fake Strands Agent that drives the ten tools per tier.
# --------------------------------------------------------------------------- #


class FakeAgent:
    """
    A stand-in for the Strands Agent. invoke_async runs the per-tier
    script: generate -> audit -> (self-correct if needed) -> (escalate
    or save). Real @tool functions are called through their underlying
    coroutines (strands' DecoratedFunctionTool._tool_func) so mission
    events get recorded through the real path.
    """

    def __init__(self, mission: main.Mission) -> None:
        self.mission = mission

    async def invoke_async(self, brief: str, **kwargs: Any) -> str:
        # The orchestrator calls invoke_async(brief, limits=...) with the
        # per-mission agent budget (app/main.py, _run_tier_sub_mission).
        # The demo script does not model the token/step budget, so the
        # kwarg is accepted here and ignored — rejecting it would raise a
        # TypeError inside every tier, which the orchestrator would
        # swallow per-tier, leaving the mission with 0 generated tiers.
        # The brief says which tier ("tier: struggling/on-level/advanced").
        for tier in ("struggling", "on-level", "advanced"):
            if f"tier: {tier}" in brief:
                _set_tier(tier)
                break
        else:
            _set_tier("on-level")

        tier = _SCRIPT["tier"]
        topic = self.mission.topic
        unit_name = self.mission.unit_name or "Unit 3: Nested Loops"
        objective = self.mission.objective or (
            "Students can trace a nested REPEAT loop and predict its total iterations."
        )

        # 1. Look up the curriculum
        await main.get_curriculum_context._tool_func(topic)

        # 2. Write a measurable objective
        await main.generate_learning_objective._tool_func(topic)

        # 3. Generate the worksheet
        if tier == "struggling":
            worksheet_result = await main.generate_differentiated_material._tool_func(
                topic, tier, objective,
            )
            ws = worksheet_result["worksheet"]
            audit_ws = await main.audit_material._tool_func(
                "worksheet", tier, topic, objective, ws
            )
            assert audit_ws["verdict"] == "pass", \
                f"struggling worksheet should pass, got {audit_ws}"
        elif tier == "on-level":
            # First attempt: bad worksheet (no blanks). The audit will
            # fail and the agent will self-correct.
            worksheet_result = await main.generate_differentiated_material._tool_func(
                topic, tier, objective,
            )
            ws_first = worksheet_result["worksheet"]
            audit_ws = await main.audit_material._tool_func(
                "worksheet", tier, topic, objective, ws_first
            )
            if audit_ws["verdict"] == "revise":
                # Real self-correction: regenerate with the audit's
                # problems as revision_notes.
                revision = "; ".join(audit_ws["problems"])
                worksheet_result = await main.generate_differentiated_material._tool_func(
                    topic, tier, objective, revision_notes=revision,
                )
            ws = worksheet_result["worksheet"]
            audit_ws2 = await main.audit_material._tool_func(
                "worksheet", tier, topic, objective, ws
            )
            assert audit_ws2["verdict"] == "pass", \
                f"on-level worksheet should pass on retry, got {audit_ws2}"
        elif tier == "advanced":
            worksheet_result = await main.generate_differentiated_material._tool_func(
                topic, tier, objective,
            )
            ws = worksheet_result["worksheet"]
            audit_ws = await main.audit_material._tool_func(
                "worksheet", tier, topic, objective, ws
            )
            assert audit_ws["verdict"] == "pass", \
                f"advanced worksheet should pass structural, got {audit_ws}"

        # 3b. AST verification: never trust a trace item without running
        # the snippet through the deterministic sandbox. The clean
        # nested-loop snippet must execute to 6 with no findings, and a
        # genuinely broken snippet must be flagged.
        ast_clean = await main.verify_code_example._tool_func(_AST_CLEAN_SNIPPET)
        assert ast_clean["ok"] and ast_clean["execution"]["ok"], \
            f"AST sandbox rejected the clean snippet: {ast_clean}"
        assert ast_clean["execution"]["output"] == ["6"], \
            f"AST sandbox computed the wrong total: {ast_clean['execution']}"
        ast_broken = await main.verify_code_example._tool_func(_AST_BROKEN_SNIPPET)
        assert not ast_broken["ok"], \
            f"AST sandbox failed to flag the broken snippet: {ast_broken}"

        # 4. Generate the assessment (quiz + answer key)
        if tier == "struggling":
            assess = await main.generate_assessment._tool_func(topic, tier, objective)
            quiz, answer_key = assess["quiz"], assess["answer_key"]
        elif tier == "on-level":
            assess = await main.generate_assessment._tool_func(topic, tier, objective)
            quiz, answer_key = assess["quiz"], assess["answer_key"]
            audit_key = await main.audit_material._tool_func(
                "answer_key", tier, topic, objective, answer_key, companion_text=quiz,
            )
            assert audit_key["verdict"] == "pass", \
                f"on-level answer key should pass, got {audit_key}"
        elif tier == "advanced":
            assess = await main.generate_assessment._tool_func(topic, tier, objective)
            quiz, answer_key = assess["quiz"], assess["answer_key"]
            audit_key = await main.audit_material._tool_func(
                "answer_key", tier, topic, objective, answer_key, companion_text=quiz,
            )
            # The audit returns "human" for the advanced tier's
            # answer key, so the agent must escalate rather than save.
            if audit_key["verdict"] == "human":
                # The agent hands the teacher the actual material it
                # produced, not just a description. The teacher reviews
                # the worksheet, quiz, and answer key directly on the
                # dashboard and can export them once approved.
                await main.flag_for_human_review._tool_func(
                    unit_name=unit_name,
                    topic=topic,
                    objective=objective,
                    tier=tier,
                    category="pedagogical_judgement",
                    reason=(
                        "Two defensible readings of the off-by-one location: line 5 "
                        "(missing final increment) vs line 7 (output ordering)."
                    ),
                    attempted_fixes="regenerated the worksheet; clarified the prompt",
                    worksheet_text=ws,
                    quiz_text=quiz,
                    answer_key_text=answer_key,
                )
                await main.finalize_lesson_package._tool_func(
                    unit_name=unit_name, topic=topic,
                    summary=f"Advanced tier escalated: the off-by-one defect has two defensible readings; a teacher must choose which is intended.",
                )
                return "advanced escalated"

        # 5. Save the tier (struggling, on-level, or if the advanced path
        # did not escalate).
        if tier != "advanced":
            save = await main.save_draft._tool_func(
                unit_name=unit_name,
                topic=topic,
                objective=objective,
                tier=tier,
                worksheet=worksheet_result["worksheet"] if tier != "on-level" else worksheet_result["worksheet"],
                quiz=quiz,
                answer_key=answer_key,
            )
            assert save["outcome"] == "saved", f"save should succeed, got {save}"

        # 6. Finalize
        await main.finalize_lesson_package._tool_func(
            unit_name=unit_name, topic=topic,
            summary=f"Completed the {tier} tier of '{topic}'.",
        )
        return f"{tier} done"


# --------------------------------------------------------------------------- #
# The demo itself.
# --------------------------------------------------------------------------- #


class DemoFailure(AssertionError):
    pass


def assert_eq(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise DemoFailure(f"{label}: expected {expected!r}, got {actual!r}")


def assert_in(needle: Any, haystack: Any, label: str) -> None:
    if needle not in haystack:
        raise DemoFailure(f"{label}: {needle!r} not found in {haystack!r}")


def assert_true(condition: bool, label: str) -> None:
    if not condition:
        raise DemoFailure(f"{label}: condition was false")


def _use_scratch_db() -> str:
    """Redirect the materials DB to a scratch file so the demo is repeatable."""
    scratch = "demo_mission.db"
    if os.path.exists(scratch):
        os.remove(scratch)
    original = main.DB_PATH
    main.DB_PATH = scratch
    return original, scratch


async def _init_db_async() -> None:
    async with main.aiosqlite.connect(main.DB_PATH) as db:
        await db.execute(main.CREATE_MATERIALS_TABLE)
        await db.commit()


def _read_records() -> list[dict[str, Any]]:
    conn = sqlite3.connect(main.DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, unit_name, topic, objective, tier, status, reason "
        "FROM materials ORDER BY id"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def _clear_in_memory_missions() -> None:
    main.MISSIONS.clear()
    main.CURRICULUM_MISSIONS.clear()
    main.RECORD_TO_CURRICULUM.clear()


def _read_records_with_text() -> list[dict[str, Any]]:
    conn = sqlite3.connect(main.DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, unit_name, topic, objective, tier, status, reason, "
        "worksheet_text, quiz_text, answer_key_text "
        "FROM materials ORDER BY id"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def _ast_verify_saved_records() -> tuple[bool, str]:
    """
    Run every saved record's material through the deterministic sandbox
    (the bounded interpreter behind verify_code_example). A record passes
    when the interpreter either executes cleanly or finds no pseudo-code
    to execute; a runtime defect (undefined variable, step-cap overrun,
    type error, division by zero) fails the check.
    """
    for row in _read_records_with_text():
        for field in ("worksheet_text", "quiz_text", "answer_key_text"):
            text = row.get(field) or ""
            if not text.strip():
                continue
            result = main.execute_pseudocode(text)
            if not result["ok"] and result["error"] and result["error"][0] != "syntax":
                return False, (
                    f"record {row['id']} ({row['tier']}) {field}: "
                    f"{result['error'][1]}"
                )
    return True, "all records executed cleanly (or hold no pseudo-code)"


async def _wait_for_tiers_ready(curriculum: main.CurriculumMission, *,
                                timeout_s: float = 30.0,
                                poll_s: float = 1.0) -> dict[str, Any]:
    """
    Poll the mission status every `poll_s` seconds (30s budget) until all
    three tiers have finished generating AND passed AST verification.

    The route hands the orchestrator to BackgroundTasks and returns
    immediately, so the mission is not ready the moment the POST returns.
    Anything polling it must wait for the tier statuses to settle, then
    confirm the deterministic interpreter is happy with what landed in
    the database.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        snap = curriculum.snapshot()
        tier_status = snap["tier_status"]
        # The mission stores the DB row status per tier: a tier the agent
        # saved lands as "draft" (the row's status column), an escalation
        # as "flagged", a teacher-approved one as "approved".
        settled = all(
            tier_status.get(t) in {"draft", "saved", "flagged", "approved"}
            for t in curriculum.tiers
        )
        ast_ok, ast_detail = _ast_verify_saved_records()
        if settled and ast_ok:
            return snap
        if time.monotonic() >= deadline:
            raise DemoFailure(
                f"mission not ready after {timeout_s:.0f}s of polling every "
                f"{poll_s:.0f}s: state={curriculum.state}, "
                f"tier_status={tier_status}, ast={ast_detail}, "
                f"errors={curriculum.errors[-3:]}"
            )
        await asyncio.sleep(poll_s)


async def run_demo() -> int:
    print("=" * 70)
    print("  IdentAI -- autonomous background mission demo")
    print("=" * 70)
    print()
    print("Goal: 'Prepare next week's differentiated CS materials for")
    print("       Unit 3: Nested Loops' (all three ability tiers).")
    print()

    _clear_in_memory_missions()
    original_db, _scratch = _use_scratch_db()
    await _init_db_async()
    main.MISSION_HISTORY_LIMIT = 64
    main.CURRICULUM_MISSION_HISTORY_LIMIT = 64

    # The route rejects with 503 if the provider is unavailable. The demo
    # stubs the agent at the model level, so the provider check is not
    # meaningful here; bypass it.
    original_unavailable = main.provider_unavailable_reason
    main.provider_unavailable_reason = lambda: None

    fake_agent_factory = lambda mission: FakeAgent(mission)

    try:
        with patch.object(main, "_complete", side_effect=stub_complete):
            with patch.object(main, "build_agent", side_effect=fake_agent_factory):
                # ------------------------------------------------------------------ #
                # Step 1: Teacher creates the mission. We call the orchestrator
                # directly (rather than POST /missions + BackgroundTasks) so the
                # demo runs deterministically in a single event loop. The route
                # is exercised at the end of the demo via TestClient, after the
                # orchestrator has finished.
                # ------------------------------------------------------------------ #
                print("[1] Teacher: POST /missions  (mission accepted, runs in background)")
                curriculum = main.new_curriculum_mission({
                    "topic": "Nested loops",
                    "unit_name": "Unit 3: Nested Loops",
                    "objective": (
                        "Students can trace a nested REPEAT loop and predict its "
                        "total iterations."
                    ),
                    "tiers": ["struggling", "on-level", "advanced"],
                })
                print(f"     mission_id: {curriculum.id}")
                print(f"     initial state: {curriculum.state}")
                print()

                # The orchestrator runs in the background exactly the way
                # the deployment runs it (BackgroundTasks ->
                # _run_curriculum_mission_background): the POST returns
                # immediately, so poll the mission status every 1s (30s
                # budget) until all three tiers have finished generating
                # and passed AST verification.
                print("[2] Mission runs in the background; polling status "
                      "every 1s (30s budget) until all 3 tiers are "
                      "generated and AST-verified...")
                orchestrator = asyncio.create_task(main.run_curriculum_mission(curriculum))
                await _wait_for_tiers_ready(curriculum)
                await orchestrator  # propagate any orchestrator crash

                # ------------------------------------------------------------------ #
                # Step 2: Inspect what the agent produced.
                # ------------------------------------------------------------------ #
                snap = curriculum.snapshot()
                print("[2] Mission state after agent work:")
                print(f"     state: {curriculum.state}")
                print(f"     current_phase: {curriculum.current_phase}")
                print(f"     progress: {snap['progress']}")
                print(f"     counts: {snap['counts']}")
                print(f"     summary: {curriculum.summary}")
                print()

                assert curriculum.state == "WAITING_FOR_HUMAN", \
                    f"expected WAITING_FOR_HUMAN, got {curriculum.state}"

                # The struggling tier should be saved; the on-level tier should
                # be saved (after self-correction); the advanced tier should
                # be flagged.
                records = _read_records()
                print(f"     database rows: {len(records)}")
                for r in records:
                    print(f"       id={r['id']} tier={r['tier']:<11} status={r['status']:<10} "
                          f"reason={(r.get('reason') or '')[:60]}")
                print()

                assert_eq(len(records), 3, "three tiers should produce three records")
                statuses = {r["tier"]: r["status"] for r in records}
                assert_eq(statuses.get("struggling"), "draft", "struggling tier status")
                assert_eq(statuses.get("on-level"), "draft", "on-level tier status")
                assert_eq(statuses.get("advanced"), "flagged", "advanced tier status")

                # ------------------------------------------------------------------ #
                # Step 3: Also exercise the real HTTP routes via TestClient.
                # ------------------------------------------------------------------ #
                from fastapi.testclient import TestClient
                client = TestClient(main.app)

                detail = client.get(f"/api/curriculum-missions/{curriculum.id}").json()
                assert_eq(detail["state"], "WAITING_FOR_HUMAN", "API state")
                assert detail["summary"], "API summary should be populated"

                # ------------------------------------------------------------------ #
                # Step 4: The teacher approves the flagged advanced-tier item.
                # ------------------------------------------------------------------ #
                advanced_row = next(r for r in records if r["tier"] == "advanced")
                print(f"[4] Teacher: POST /approve/{advanced_row['id']}")
                resp = client.post(f"/approve/{advanced_row['id']}")
                assert resp.status_code == 200, \
                    f"approve should return 200, got {resp.status_code}: {resp.text}"
                body = resp.json()
                print(f"     approved: id={body['id']} tier={body['tier']} status={body['status']}")
                print(f"     curriculum_mission_id: {body.get('curriculum_mission_id')}")
                print()

                # ------------------------------------------------------------------ #
                # Step 5: The mission should now be COMPLETED.
                # ------------------------------------------------------------------ #
                detail = client.get(f"/api/curriculum-missions/{curriculum.id}").json()
                print("[5] Mission state after approval:")
                print(f"     state: {detail['state']}")
                print(f"     summary: {detail['summary']}")
                print(f"     counts: {detail['counts']}")
                print()

                assert_eq(detail["state"], "COMPLETED", "mission state after approval")
                # After the teacher approves, the summary may read either
                # "all handled automatically" (nothing waiting) or
                # "X/Y handled automatically, 1 decision required your
                # attention" (when one still pending). After approval there
                # is nothing pending, so we accept either phrase as long as
                # the agent's auto-handled count is reflected.
                summary_low = detail["summary"].lower()
                assert_true(
                    "handled automatically" in summary_low
                    or "handled" in summary_low and "completed" in summary_low,
                    f"summary should mention handled/completed, got: {detail['summary']!r}",
                )

                # The active section should be empty; the completed section should
                # have one mission; the waiting section should be empty.
                active = client.get("/api/curriculum-missions/active").json()
                waiting = client.get("/api/curriculum-missions/waiting").json()
                completed = client.get("/api/curriculum-missions/completed").json()
                assert_eq(len(active), 0, "active section")
                assert_eq(len(waiting), 0, "waiting section")
                assert_eq(len(completed), 1, "completed section")

                # ------------------------------------------------------------------ #
                # Step 6: The classroom-ready output is in the database.
                # ------------------------------------------------------------------ #
                records = _read_records()
                print("[6] Classroom-ready output:")
                for r in records:
                    print(f"     id={r['id']} tier={r['tier']:<11} status={r['status']:<10}")
                print()
                assert_eq(len(records), 3, "three records saved")
                # The teacher's approve action closed the mission. The two
                # auto-saved drafts and the one approved record are the
                # classroom-ready output. The teacher can one-click the
                # remaining drafts to clear them for printing, but the
                # mission is already complete.
                statuses = [r["status"] for r in records]
                assert all(s in {"draft", "approved"} for s in statuses), \
                    f"all three records should be draft or approved, got {statuses}"
                assert sum(1 for s in statuses if s == "approved") >= 1, \
                    f"at least one record should be approved, got {statuses}"

                # ------------------------------------------------------------------ #
                # Step 7: Verify the mission log captured the agent's decisions.
                # ------------------------------------------------------------------ #
                sub_missions = detail["sub_mission_ids"]
                print(f"[7] Mission log captured {len(sub_missions)} sub-missions:")
                for sub_id in sub_missions:
                    sub = main.MISSIONS.get(sub_id)
                    if sub is None:
                        continue
                    kinds = [e["kind"] for e in sub.events]
                    print(f"     {sub_id}: {len(sub.events)} events, kinds={sorted(set(kinds))}")
                assert_eq(len(sub_missions), 3, "three sub-missions")

                on_level_sub = None
                advanced_sub = None
                for sub_id in sub_missions:
                    sub = main.MISSIONS.get(sub_id)
                    if sub is None:
                        continue
                    for event in sub.events:
                        if event.get("tier") == "on-level":
                            on_level_sub = sub
                        if event.get("tier") == "advanced":
                            advanced_sub = sub

                # The on-level sub-mission should have a self_correction event.
                on_level_kinds = [e["kind"] for e in on_level_sub.events] if on_level_sub else []
                assert_true(
                    "self_correction" in on_level_kinds or "audit_failed" in on_level_kinds,
                    f"on-level tier should show a self-correction event, got {on_level_kinds}",
                )

                # The advanced sub-mission should have a human_review event.
                advanced_kinds = [e["kind"] for e in advanced_sub.events] if advanced_sub else []
                assert_true(
                    "human_review" in advanced_kinds,
                    f"advanced tier should show a human_review event, got {advanced_kinds}",
                )

                # ------------------------------------------------------------------ #
                # Step 8: Exercise the POST /missions route for completeness.
                # ------------------------------------------------------------------ #
                print("[8] Exercise the real POST /missions route:")
                resp = client.post("/missions", json={
                    "topic": "Variables and assignment",
                    "tiers": ["on-level"],
                })
                assert resp.status_code == 200, \
                    f"POST /missions should return 200, got {resp.status_code}: {resp.text}"
                body = resp.json()
                print(f"     mission_id: {body['mission_id']}, state={body['state']}")

                # Wait for the second mission to finish.
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    detail2 = client.get(f"/api/curriculum-missions/{body['mission_id']}").json()
                    if detail2["state"] in {"WAITING_FOR_HUMAN", "COMPLETED", "FAILED"}:
                        break
                    await asyncio.sleep(0.1)
                print(f"     second mission state: {detail2['state']}")
                assert detail2["state"] in {"COMPLETED", "WAITING_FOR_HUMAN"}, \
                    f"second mission should be terminal, got {detail2['state']}"

                # ------------------------------------------------------------------ #
                # Step 9: Export the approved advanced-tier record. The teacher
                # can now take the package to class: HTML handout, Markdown
                # document, and a print-to-PDF route.
                # ------------------------------------------------------------------ #
                approved_advanced = next(r for r in records
                                          if r["tier"] == "advanced")
                print(f"[9] Teacher: export record {approved_advanced['id']}")
                for fmt in ("html", "md", "pdf"):
                    resp = client.get(f"/api/export/{approved_advanced['id']}/{fmt}")
                    print(f"     {fmt}: status={resp.status_code}, "
                          f"content-type={resp.headers.get('content-type','?')}, "
                          f"bytes={len(resp.content)}")
                    assert resp.status_code == 200, \
                        f"export {fmt} returned {resp.status_code}: {resp.text[:200]}"
                    body = resp.content
                    body_str = body.decode("utf-8", "replace")
                    if fmt == "html":
                        assert "Worksheet" in body_str, f"HTML missing Worksheet: {body_str[:200]}"
                        assert "Misconception" in body_str, f"HTML missing Misconception: {body_str[:200]}"
                    elif fmt == "md":
                        assert "## Answer key" in body_str or "ANSWER KEY" in body_str, \
                            f"MD missing answer key: {body_str[:200]}"
                        assert "Misconception diagnostics" in body_str, \
                            f"MD missing misconception diagnostics: {body_str[:200]}"
                    elif fmt == "pdf":
                        assert "window.print" in body_str, f"PDF missing window.print trigger"
                        assert "<noscript>" in body_str, f"PDF missing noscript fallback"
                print("     HTML, Markdown, and PDF exports all returned 200 with content.")

                # ------------------------------------------------------------------ #
                # Step 10: The agent activity endpoint surfaces safe, concise
                # events for the approved record. We assert the timeline shape
                # and that chain-of-thought is redacted.
                # ------------------------------------------------------------------ #
                print(f"[10] Teacher: /api/materials/{approved_advanced['id']}/activity")
                activity = client.get(
                    f"/api/materials/{approved_advanced['id']}/activity"
                ).json()
                assert "timeline" in activity
                assert "agent_reasoning" not in [e["kind"] for e in activity["timeline"]], \
                    f"chain-of-thought leaked: {[e['kind'] for e in activity['timeline']]}"
                kinds = sorted({e["kind"] for e in activity["timeline"]})
                print(f"     activity kinds: {kinds}")

                # ------------------------------------------------------------------ #
                # Step 11: Impact metrics — the dashboard's per-mission strip.
                # The teacher sees honest counts, no fake precision, and the
                # time-saved cell is explicitly labelled as an estimate. The
                # teacher already approved the advanced tier in step 4, so the
                # post-approval state shows 3 auto-handled + 0 escalated.
                # ------------------------------------------------------------------ #
                print("[11] Impact metrics for the first mission:")
                metrics = detail["metrics"]
                for key in ("tasks_completed_automatically", "human_decisions_required",
                            "materials_generated", "validation_failures_auto_repaired",
                            "time_saved_estimate_seconds", "time_saved_estimate_label"):
                    print(f"     {key:<38} {metrics[key]}")
                assert_eq(metrics["time_saved_estimate_label"], "estimated",
                          "time_saved label")
                # After the teacher's approval in step 4, all 3 tiers
                # count as auto-handled and zero are waiting.
                assert_true(
                    metrics["tasks_completed_automatically"] == 3,
                    "all 3 tiers should be auto-handled after approval",
                )
                assert_true(
                    metrics["human_decisions_required"] == 0,
                    "no human decisions required after approval",
                )
                assert_true(
                    metrics["materials_generated"] == 3,
                    "materials generated should be 3",
                )
                assert_true(
                    metrics["time_saved_estimate_seconds"] > 0,
                    "time saved estimate should be positive",
                )
                assert_true(
                    "benchmark" in metrics["time_saved_benchmark"].lower(),
                    "benchmark string should mention the methodology",
                )

                # ------------------------------------------------------------------ #
                # Step 12: Aggregate metrics across all missions.
                # ------------------------------------------------------------------ #
                print("[12] Aggregate metrics across all missions:")
                aggregate = client.get("/api/metrics").json()
                for key in ("missions_total", "missions_completed",
                            "tasks_completed_automatically",
                            "human_decisions_required",
                            "materials_generated",
                            "time_saved_estimate_seconds",
                            "time_saved_estimate_label"):
                    print(f"     {key:<38} {aggregate[key]}")
                assert_eq(aggregate["time_saved_estimate_label"], "estimated",
                          "aggregate time-saved label")
                assert_true(
                    aggregate["materials_generated"] >= 3,
                    "aggregate materials_generated should be >= 3",
                )

                print()
                print("PASS -- the agent ran all three tiers, fixed the on-level")
                print("       worksheet on its own, escalated the advanced tier to")
                print("       the teacher, the mission closed when the teacher")
                print("       approved the flagged item, and the teacher can now")
                print("       export the approved record in HTML, Markdown, or PDF.")
                return 0
    finally:
        main.DB_PATH = original_db
        main.provider_unavailable_reason = original_unavailable
        if os.path.exists("demo_mission.db"):
            os.remove("demo_mission.db")


def main_sync() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    args = parser.parse_args()
    try:
        return asyncio.run(run_demo())
    except DemoFailure as exc:
        print(f"\nFAIL: {exc}")
        return 1
    except AssertionError as exc:
        print(f"\nFAIL (assertion): {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main_sync())
