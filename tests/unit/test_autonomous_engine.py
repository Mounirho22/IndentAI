"""
Tests for the autonomous learning-agent runtime (app/interpreter/engine.py).

The critical property under test is the autonomy boundary: routine outcomes
must stay silent (state transitions + telemetry only), and exactly three
triggers may open a human decision — stuck_loop, hallucination_risk,
permission_gate. Everything runs in deterministic mode (MODEL_PROVIDER=mock),
which is the mode the runtime falls back to whenever no model is reachable.
"""
from __future__ import annotations

import asyncio
import os

import aiosqlite
import pytest

os.environ.setdefault("MODEL_PROVIDER", "mock")

from app.interpreter.engine import AutonomousAgentEngine  # noqa: E402
from app.interpreter.evaluator import (  # noqa: E402
    STUCK_ATTEMPT_THRESHOLD,
    MilestoneProfile,
    UnitCase,
    analyze_syntax,
    decide_autonomy,
    evaluate_milestone,
    run_unit_tests,
    assess_claim,
)

GOOD_CODE = "SET counter TO 0\nREPEAT 3 TIMES\nSET counter TO counter + 1\nEND REPEAT\nPRINT counter"
STUCK_CODE = "WHILE counter < 3\nPRINT counter\nEND WHILE"  # body never changes counter
BAD_SYNTAX = "FOR i = 1 TO 3\nPRINT i\nNEXT i"


async def _make_engine(tmp_path, **kwargs) -> AutonomousAgentEngine:
    # Tests drain the queue by hand, so the engine runs without its worker
    # task — that keeps every transition deterministic. The worker itself is
    # covered by test_background_loop_drains_without_prompts.
    engine = AutonomousAgentEngine(tmp_path / "lessons.db", poll_interval=0.02, **kwargs)
    await engine.start(run_worker=False)
    return engine


async def _submission_status(engine: AutonomousAgentEngine, submission_id: int) -> str:
    async with aiosqlite.connect(engine.db_path) as db:
        async with db.execute(
            "SELECT status FROM student_submissions WHERE id = ?", (submission_id,)
        ) as cursor:
            row = await cursor.fetchone()
    assert row is not None, "submission row vanished"
    return str(row[0])


async def _milestone(engine: AutonomousAgentEngine, student_id: str, milestone_id: str) -> dict:
    async with aiosqlite.connect(engine.db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM student_milestones WHERE student_id = ? AND milestone_id = ?",
            (student_id, milestone_id),
        ) as cursor:
            row = await cursor.fetchone()
    return dict(row) if row is not None else {}


def test_clean_pass_is_silent(tmp_path) -> None:
    """A routine correct submission commits its state; no human ever hears of it."""
    async def scenario() -> None:
        engine = await _make_engine(tmp_path)
        try:
            sid = await engine.ingest_submission(
                "s1", "loops", GOOD_CODE, expected_output=["3"],
            )
            await engine.drain_once()
            assert await _submission_status(engine, sid) == "auto_passed"
            assert await engine.open_decisions() == []
            milestone = await _milestone(engine, "s1", "loops")
            assert milestone["state"] == "completed"
            assert milestone["consecutive_failures"] == 0
            # The silence is accountable: the transition is on the telemetry trail.
            async with aiosqlite.connect(engine.db_path) as db:
                async with db.execute(
                    "SELECT COUNT(*) FROM agent_telemetry WHERE event = 'submission_auto_processed'"
                ) as cursor:
                    assert (await cursor.fetchone())[0] == 1
        finally:
            await engine.close()

    asyncio.run(scenario())


def test_syntax_failure_is_autonomous(tmp_path) -> None:
    """Untaught constructs fail the run quietly — below the gate, no escalation."""
    async def scenario() -> None:
        engine = await _make_engine(tmp_path)
        try:
            sid = await engine.ingest_submission("s1", "loops", BAD_SYNTAX)
            await engine.drain_once()
            assert await _submission_status(engine, sid) == "auto_failed"
            assert await engine.open_decisions() == []
        finally:
            await engine.close()

    asyncio.run(scenario())


def test_stuck_loop_escalates_only_past_threshold(tmp_path) -> None:
    """
    Failures 1..STUCK_ATTEMPT_THRESHOLD are autonomous (struggling band);
    the first failure beyond the threshold opens exactly one brief.
    """
    async def scenario() -> None:
        engine = await _make_engine(tmp_path)
        try:
            ids = []
            for _ in range(STUCK_ATTEMPT_THRESHOLD + 1):
                ids.append(await engine.ingest_submission("s1", "loops", STUCK_CODE))
                await engine.drain_once()
                open_categories = [d.category for d in await engine.open_decisions()]
                assert "stuck_loop" not in open_categories or len(ids) > STUCK_ATTEMPT_THRESHOLD

            decisions = await engine.open_decisions()
            assert [d.category for d in decisions] == ["stuck_loop"]
            assert decisions[0].submission_id == ids[-1]
            assert await _submission_status(engine, ids[-1]) == "escalated"
            # Work is frozen while paused.
            assert (await _milestone(engine, "s1", "loops"))["state"] == "blocked"
        finally:
            await engine.close()

    asyncio.run(scenario())


def test_advise_resolution_requeues_without_re_escalating(tmp_path) -> None:
    """
    advise injects the mentor's note and resumes autonomously. The guided
    re-run must not re-count the attempt nor open a fresh brief on identical
    evidence; a genuinely NEW failing submission re-arms the gate.
    """
    async def scenario() -> None:
        engine = await _make_engine(tmp_path)
        try:
            for _ in range(STUCK_ATTEMPT_THRESHOLD + 1):
                sid = await engine.ingest_submission("s1", "loops", STUCK_CODE)
                await engine.drain_once()
            (brief,) = await engine.open_decisions()

            resolved = await engine.resolve_decision(
                brief.decision_id, "Hint: the body must change counter.",
                resolution="advise",
            )
            assert resolved.status == "resolved"
            assert resolved.resolution == "advise"

            for _ in range(3):
                await engine.drain_once()
            # The gate stayed closed through the guided re-runs.
            assert await engine.open_decisions() == []
            row_status = await _submission_status(engine, sid)
            assert row_status == "auto_failed"
            milestone = await _milestone(engine, "s1", "loops")
            # The re-runs were not new attempts: counters unchanged.
            assert milestone["consecutive_failures"] == STUCK_ATTEMPT_THRESHOLD + 1
            assert milestone["total_attempts"] == STUCK_ATTEMPT_THRESHOLD + 1

            # But a genuinely new attempt re-arms the gate.
            await engine.ingest_submission("s1", "loops", STUCK_CODE)
            await engine.drain_once()
            fresh = await engine.open_decisions()
            assert len(fresh) == 1 and fresh[0].category == "stuck_loop"
        finally:
            await engine.close()

    asyncio.run(scenario())


def test_hallucination_risk_escalates_and_resolves(tmp_path) -> None:
    """A claimed output that contradicts execution is a judgement call, not effort."""
    async def scenario() -> None:
        engine = await _make_engine(tmp_path)
        try:
            sid = await engine.ingest_submission(
                "s1", "predict", "SET x TO 5\nPRINT x + 1",
                expected_output=["6"], claimed_output="7",
            )
            await engine.drain_once()
            (brief,) = await engine.open_decisions()
            assert brief.category == "hallucination_risk"
            assert brief.severity == "high"
            assert brief.evidence["claim"]["hallucination_risk"] is True

            resolved = await engine.resolve_decision(
                brief.decision_id, "Confirmed: the material's expected output is wrong.",
                resolution="reject",
            )
            assert resolved.status == "resolved"
            assert await _submission_status(engine, sid) == "auto_failed"
            assert await engine.open_decisions() == []
        finally:
            await engine.close()

    asyncio.run(scenario())


def test_permission_gate_and_unlock(tmp_path) -> None:
    """
    A passing student at a gated advanced module pauses for permission.
    The unlock is refused until a mentor approves, then granted.
    """
    async def scenario() -> None:
        engine = await _make_engine(tmp_path)
        try:
            sid = await engine.ingest_submission(
                "s1", "advanced-recursion", GOOD_CODE, expected_output=["3"],
                requires_permission=True,
            )
            await engine.drain_once()
            (brief,) = await engine.open_decisions()
            assert brief.category == "permission_gate"
            assert await _submission_status(engine, sid) == "escalated"

            # Below the human gate the unlock is impossible — even for the engine.
            assert await engine.has_approved_permission("s1", "advanced-recursion") is False
            refused = await engine.apply_unlock(student_id="s1", milestone_id="advanced-recursion")
            assert refused["unlocked"] is False

            await engine.resolve_decision(
                brief.decision_id, "Ready for the advanced module.", resolution="approve",
            )
            assert await engine.has_approved_permission("s1", "advanced-recursion") is True
            unlocked = await engine.apply_unlock(student_id="s1", milestone_id="advanced-recursion")
            assert unlocked["unlocked"] is True
            milestone = await _milestone(engine, "s1", "advanced-recursion")
            assert milestone["unlocked"] == 1
        finally:
            await engine.close()

    asyncio.run(scenario())


def test_resolve_unknown_decision_raises(tmp_path) -> None:
    async def scenario() -> None:
        engine = await _make_engine(tmp_path)
        try:
            with pytest.raises(LookupError):
                await engine.resolve_decision("nope", "x")
            await engine.ingest_submission("s1", "loops", GOOD_CODE, expected_output=["3"])
            await engine.drain_once()
            (brief,) = [d for d in await engine.open_decisions()] or [None]
            assert brief is None  # nothing escalated, nothing resolvable twice
        finally:
            await engine.close()

    asyncio.run(scenario())


def test_double_resolve_raises(tmp_path) -> None:
    async def scenario() -> None:
        engine = await _make_engine(tmp_path)
        try:
            sid = await engine.ingest_submission(
                "s1", "predict", "SET x TO 5\nPRINT x + 1",
                expected_output=["6"], claimed_output="7",
            )
            await engine.drain_once()
            (brief,) = await engine.open_decisions()
            await engine.resolve_decision(brief.decision_id, "ok", resolution="approve")
            with pytest.raises(LookupError):
                await engine.resolve_decision(brief.decision_id, "again")
            assert await _submission_status(engine, sid) == "auto_passed"
        finally:
            await engine.close()

    asyncio.run(scenario())


def test_background_loop_drains_without_prompts(tmp_path) -> None:
    """The spawned worker empties the queue on its own — no human, no manual drain."""
    async def scenario() -> None:
        engine = AutonomousAgentEngine(tmp_path / "lessons.db", poll_interval=0.02)
        await engine.start()  # with the worker: this is the production mode
        try:
            for _ in range(3):
                await engine.ingest_submission("s1", "loops", GOOD_CODE, expected_output=["3"])
            statuses: list[str] = []
            for _ in range(100):
                await asyncio.sleep(0.05)
                async with aiosqlite.connect(engine.db_path) as db:
                    async with db.execute(
                        "SELECT status FROM student_submissions"
                    ) as cursor:
                        statuses = [r[0] for r in await cursor.fetchall()]
                if statuses and all(s in ("auto_passed", "auto_failed") for s in statuses):
                    break
            assert statuses and all(s == "auto_passed" for s in statuses)
        finally:
            await engine.close()

    asyncio.run(scenario())


# --------------------------------------------------------------------------- #
# Deterministic assessment layer (evaluator.py)
# --------------------------------------------------------------------------- #


def test_milestone_threshold_is_strictly_beyond_three() -> None:
    struggling = evaluate_milestone(MilestoneProfile("s", "m", consecutive_failures=3))
    stuck = evaluate_milestone(MilestoneProfile("s", "m", consecutive_failures=4))
    assert struggling.state == "struggling" and struggling.escalate is False
    assert stuck.state == "stuck" and stuck.escalate is True


def test_permission_gate_verdict() -> None:
    gated = evaluate_milestone(MilestoneProfile(
        "s", "m", consecutive_failures=0, requires_permission=True,
    ))
    approved = evaluate_milestone(MilestoneProfile(
        "s", "m", consecutive_failures=0, requires_permission=True, permission_approved=True,
    ))
    assert gated.state == "advance_gate" and gated.escalate is True
    assert approved.escalate is False


def test_hallucination_outranks_stuck_in_autonomy_decision() -> None:
    syntax = analyze_syntax(STUCK_CODE)
    tests = run_unit_tests(STUCK_CODE, [UnitCase("t", {}, ("3",))])
    claim = assess_claim("1\n2\n3", ["", ""])  # mismatch on purpose
    stuck = evaluate_milestone(MilestoneProfile("s", "m", consecutive_failures=9))
    decision = decide_autonomy(syntax=syntax, tests=tests, claim=claim, milestone=stuck)
    assert decision.category == "hallucination_risk"


def test_analyze_syntax_flags_untaught_and_unclosed() -> None:
    report = analyze_syntax(BAD_SYNTAX)
    assert not report.ok
    assert any(f.code == "untaught_construct" for f in report.findings)

    unclosed = analyze_syntax("IF counter > 2\nPRINT counter")
    assert any(f.code == "unclosed_block" for f in unclosed.findings)

    clean = analyze_syntax(GOOD_CODE)
    assert clean.ok and clean.errors == ()


def test_run_unit_tests_matches_normalised_output() -> None:
    result = run_unit_tests(GOOD_CODE, [UnitCase("case-1", {}, ("3",))])
    assert result.ok and result.passed == 1
    # Trailing punctuation / case differences never fail a student.
    tolerant = run_unit_tests('PRINT "Hello World"', [UnitCase("case-1", {}, ("hello world.",))])
    assert tolerant.ok
