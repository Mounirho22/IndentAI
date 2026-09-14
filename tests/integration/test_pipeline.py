"""
Integration tests for the complete agent lifecycle: topic/standard input
parsing and prompt synthesis, the deterministic AST execution sandbox and
its loop bounds, 3-tier schema compliance, and SQLite mission persistence
and retrieval.

The model call is stubbed at the `_complete` level and the Strands agent
at the `build_agent` factory level (the same seam verify.py and the demos
use), so every tool coroutine, audit, state transition, and database
write below runs through the real production code paths.

Run from the repo root:

    pytest tests/integration/test_pipeline.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import app.main as main
from app.interpreter import execute_pseudocode
from app.main import verify_pseudocode
from demos import demo_mission as demo


# --------------------------------------------------------------------------- #
# Fixture: scratch database + stubbed model, real everything else.
# --------------------------------------------------------------------------- #


class _NoopLearningEngine:
    """Lifespan stand-in so tests never spawn the real background worker.

    The real engine binds to the developer's data/lessons.db at import time;
    letting TestClient lifespans start it here would race its poller against
    the dev server's engine for the same SQLite file.
    """

    async def start(self, **kwargs):
        return None

    async def close(self):
        return None


@pytest.fixture()
def scratch_env(tmp_path):
    """Point the app at a throwaway SQLite file and stub the model layer."""
    scratch = str(tmp_path / "integration.db")
    original_db = main.DB_PATH
    original_unavailable = main.provider_unavailable_reason
    original_engine = main.LEARNING_ENGINE
    main.DB_PATH = scratch
    main.provider_unavailable_reason = lambda: None
    main.LEARNING_ENGINE = _NoopLearningEngine()
    demo._clear_in_memory_missions()
    asyncio.run(_init_db(scratch))
    try:
        yield scratch
    finally:
        demo._clear_in_memory_missions()
        main.DB_PATH = original_db
        main.provider_unavailable_reason = original_unavailable
        main.LEARNING_ENGINE = original_engine


async def _init_db(path: str) -> None:
    async with main.aiosqlite.connect(path) as db:
        await db.execute(main.CREATE_MATERIALS_TABLE)
        await db.execute(main.CREATE_CURRICULUM_MISSIONS_TABLE)
        await db.commit()


def _run_mission_to_waiting(scratch_db: str) -> main.CurriculumMission:
    """
    Drive one full 3-tier curriculum mission through the real orchestrator
    with the demo's deterministic model stub, exactly as the background
    pipeline would, and return the settled mission.
    """
    async def _drive() -> main.CurriculumMission:
        curriculum = main.new_curriculum_mission({
            "topic": "Nested loops",
            "unit_name": "Unit 3: Nested Loops",
            "objective": (
                "Students can trace a nested REPEAT loop and predict its "
                "total iterations."
            ),
            "tiers": ["struggling", "on-level", "advanced"],
        })
        with patch.object(main, "_complete", side_effect=demo.stub_complete), \
             patch.object(main, "build_agent",
                          side_effect=lambda mission: demo.FakeAgent(mission)):
            await main.run_curriculum_mission(curriculum)
        return curriculum

    return asyncio.run(_drive())


# --------------------------------------------------------------------------- #
# 1. Topic / standard input parsing and prompt synthesis.
# --------------------------------------------------------------------------- #


class TestInputParsingAndPromptSynthesis:
    def test_request_model_parses_topic_standard_and_tiers(self):
        payload = main.CurriculumMissionRequest(
            topic="Nested loops",
            unit_name="Unit 3: Nested Loops",
            objective="Students can trace a nested REPEAT loop.",
            tiers=["struggling", "on-level", "advanced"],
        )
        assert payload.topic == "Nested loops"
        assert payload.unit_name == "Unit 3: Nested Loops"
        assert payload.objective.startswith("Students can")
        assert set(payload.tiers) == set(main.TIERS)

    def test_request_model_defaults_to_all_three_tiers(self):
        payload = main.CurriculumMissionRequest(topic="Variables")
        assert payload.tiers is None  # the orchestrator expands to main.TIERS
        assert payload.objective == ""

    def test_empty_topic_is_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            main.CurriculumMissionRequest(topic="")

    def test_curriculum_mission_normalises_goal(self):
        mission = main.new_curriculum_mission({
            "topic": "Nested loops",
            "tiers": ["struggling", "on-level", "advanced"],
        })
        assert mission.topic == "Nested loops"
        assert list(mission.tiers) == ["struggling", "on-level", "advanced"]
        assert mission.tier_status == {t: "pending" for t in main.TIERS}
        assert mission.state == "QUEUED"

    def test_tier_brief_synthesises_topic_tier_and_objective(self):
        curriculum = main.new_curriculum_mission({
            "topic": "Nested loops",
            "unit_name": "Unit 3: Nested Loops",
            "objective": "Students can trace a nested REPEAT loop.",
            "tiers": list(main.TIERS),
        })
        for tier in main.TIERS:
            brief = main._tier_brief(curriculum, tier)
            assert f"topic: Nested loops" in brief
            assert f"tier: {tier}" in brief
            assert "objective: Students can trace a nested REPEAT loop." in brief
            # The brief must state the per-tier acceptance criterion so the
            # prompt drives the agent toward save-or-escalate, not wandering.
            assert "save_draft" in brief or "save" in brief
            assert "flag_for_human_review" in brief

    def test_missions_route_parses_and_accepts_the_input(self, scratch_env):
        from fastapi.testclient import TestClient
        with TestClient(main.app) as client:
            resp = client.post("/missions", json={
                "topic": "Nested loops",
                "unit_name": "Unit 3: Nested Loops",
                "objective": "Students can trace a nested REPEAT loop.",
                "tiers": ["struggling", "on-level", "advanced"],
            })
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["state"] in {"QUEUED", "RUNNING"}
            assert body["mission_id"] in main.CURRICULUM_MISSIONS
            mission = main.CURRICULUM_MISSIONS[body["mission_id"]]
            assert mission.topic == "Nested loops"
            assert list(mission.tiers) == list(main.TIERS)


# --------------------------------------------------------------------------- #
# 2. Deterministic AST execution sandbox and loop bound checks.
# --------------------------------------------------------------------------- #


class TestAstExecutionSandbox:
    def test_repeats_execute_deterministically(self):
        result = execute_pseudocode(textwrap.dedent("""\
            SET total = 0
            REPEAT 3 TIMES
              REPEAT 2 TIMES
                SET total = total + 1
              END REPEAT
            END REPEAT
            PRINT total
            """))
        assert result.ok
        assert result.output == ["6"]
        assert result.steps > 0
        assert result.error is None

    def test_same_input_always_produces_the_same_trace(self):
        code = "SET x = 0\nREPEAT 4 TIMES\nSET x = x + 1\nEND REPEAT\nPRINT x"
        traces = [tuple(execute_pseudocode(code).output) for _ in range(5)]
        assert traces == [("4",)] * 5

    def test_loop_bound_step_cap_reports_infinite(self):
        result = execute_pseudocode(
            "SET x = 1\nWHILE x > 0\nSET x = x + 1\nEND WHILE",
            step_cap=200,
        )
        assert not result.ok
        assert result.error[0] == "infinite"
        assert result.steps >= 200

    def test_non_progressing_while_is_flagged_infinite_at_default_cap(self):
        result = execute_pseudocode(textwrap.dedent("""\
            SET x = 0
            WHILE x < 10
              SET x = x - 1
            END WHILE
            PRINT x
            """))
        assert not result.ok
        assert result.error[0] == "infinite"

    def test_division_by_zero_is_a_typed_error(self):
        result = execute_pseudocode("SET a = 10 / 0\nPRINT a")
        assert not result.ok
        assert result.error[0] == "divzero"

    def test_undefined_variable_is_a_typed_error(self):
        result = execute_pseudocode("PRINT never_set")
        assert not result.ok
        assert result.error[0] == "undef"

    def test_prose_only_input_reports_syntax_honestly(self):
        result = execute_pseudocode("This worksheet is about loops. Enjoy!")
        assert not result.ok
        assert result.error[0] == "syntax"

    def test_static_analysis_flags_broken_block_structure(self):
        report = verify_pseudocode("SET total = 0\nEND REPEAT\nPRINT total")
        assert not report["ok"]
        codes = {f["code"] for f in report["errors"]}
        assert "stray_end" in codes

    def test_static_analysis_flags_real_language_syntax(self):
        report = verify_pseudocode("print(total)\nSET x = 1")
        assert not report["ok"]
        codes = {f["code"] for f in report["errors"]}
        assert "real_syntax" in codes

    def test_verify_code_example_tool_wires_static_and_dynamic_checks(self):
        async def _call() -> dict[str, Any]:
            return await main.verify_code_example._tool_func(
                "SET total = 0\nREPEAT 2 TIMES\nSET total = total + 1\n"
                "END REPEAT\nPRINT total"
            )
        report = asyncio.run(_call())
        assert report["ok"] and report["execution"]["ok"]
        assert report["execution"]["output"] == ["2"]


# --------------------------------------------------------------------------- #
# 3. 3-tier schema compliance.
# --------------------------------------------------------------------------- #


class TestThreeTierSchemaCompliance:
    def test_tier_vocabulary_is_exact(self):
        assert main.TIERS == ("struggling", "on-level", "advanced")

    def test_full_mission_produces_all_three_tiers(self, scratch_env):
        curriculum = _run_mission_to_waiting(scratch_env)
        assert curriculum.state == "WAITING_FOR_HUMAN"
        # Every tier settled in the schema's vocabulary, none pending.
        assert set(curriculum.tier_status) == set(main.TIERS)
        assert all(
            status in {"draft", "saved", "flagged", "approved"}
            for status in curriculum.tier_status.values()
        )

    def test_database_rows_match_the_3_tier_schema(self, scratch_env):
        _run_mission_to_waiting(scratch_env)
        conn = sqlite3.connect(scratch_env)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT tier, status, worksheet_text, quiz_text, answer_key_text, "
            "objective FROM materials ORDER BY id"
        ).fetchall()
        conn.close()
        assert len(rows) == 3
        tiers = [row["tier"] for row in rows]
        assert tiers == ["struggling", "on-level", "advanced"]
        # The scripted run saves the first two tiers and escalates the third.
        statuses = {row["tier"]: row["status"] for row in rows}
        assert statuses["struggling"] in {"draft", "approved"}
        assert statuses["on-level"] in {"draft", "approved"}
        assert statuses["advanced"] == "flagged"
        for row in rows:
            assert (row["worksheet_text"] or "").strip()
            assert (row["quiz_text"] or "").strip()
            assert (row["answer_key_text"] or "").strip()
            # Schema compliance: the objective is measurable-verb based.
            assert any(verb in (row["objective"] or "").lower()
                       for verb in ("trace", "predict", "write", "identify"))

    def test_generated_materials_pass_the_ast_sandbox(self, scratch_env):
        _run_mission_to_waiting(scratch_env)
        conn = sqlite3.connect(scratch_env)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, tier, worksheet_text, quiz_text, answer_key_text "
            "FROM materials"
        ).fetchall()
        conn.close()
        assert rows
        for row in rows:
            for field in ("worksheet_text", "quiz_text", "answer_key_text"):
                result = execute_pseudocode(row[field] or "")
                # Syntax complaints are the honest "no code here" answer for
                # prose-heavy fields; runtime defects are the real failures.
                assert result.ok or (result.error and result.error[0] == "syntax"), (
                    f"row {row['id']} ({row['tier']}) {field}: {result.error}"
                )

    def test_approval_promotes_flagged_tier_and_completes_mission(self, scratch_env):
        curriculum = _run_mission_to_waiting(scratch_env)
        conn = sqlite3.connect(scratch_env)
        flagged_id = conn.execute(
            "SELECT id FROM materials WHERE tier = 'advanced' AND status = 'flagged'"
        ).fetchone()[0]
        conn.close()
        assert curriculum.approve_flagged(flagged_id) is True
        assert curriculum.state == "COMPLETED"
        assert not curriculum.has_pending_human_review


# --------------------------------------------------------------------------- #
# 4. SQLite mission persistence and retrieval.
# --------------------------------------------------------------------------- #


class TestSqliteMissionPersistence:
    def test_mission_row_is_persisted_with_full_state(self, scratch_env):
        curriculum = _run_mission_to_waiting(scratch_env)
        conn = sqlite3.connect(scratch_env)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM curriculum_missions WHERE id = ?", (curriculum.id,)
        ).fetchone()
        conn.close()
        assert row is not None, "mission was not persisted to SQLite"
        assert row["state"] == "WAITING_FOR_HUMAN"
        assert row["topic"] == "Nested loops"
        assert row["unit_name"] == "Unit 3: Nested Loops"
        tier_status = json.loads(row["tier_status_json"])
        assert set(tier_status) == set(main.TIERS)
        assert "1 decision" in (row["summary"] or "")

    def test_persisted_mission_is_retrievable_via_the_api(self, scratch_env):
        curriculum = _run_mission_to_waiting(scratch_env)
        from fastapi.testclient import TestClient
        with TestClient(main.app) as client:
            detail = client.get(
                f"/api/curriculum-missions/{curriculum.id}"
            ).json()
        assert detail["state"] == "WAITING_FOR_HUMAN"
        assert detail["topic"] == "Nested loops"
        assert set(detail["tier_status"]) == set(main.TIERS)
        assert len(detail["sub_mission_ids"]) == 3
        assert detail["counts"]["generated"] == 3

    def test_mission_restores_from_sqlite_after_a_restart(self, scratch_env):
        curriculum = _run_mission_to_waiting(scratch_env)
        # Simulate a process restart: the in-memory registry is gone.
        demo._clear_in_memory_missions()
        assert main.CURRICULUM_MISSIONS == {}
        asyncio.run(main.restore_curriculum_missions_from_db())
        restored = main.CURRICULUM_MISSIONS.get(curriculum.id)
        assert restored is not None, "mission did not survive the restart"
        assert restored.state == "WAITING_FOR_HUMAN"
        assert restored.topic == "Nested loops"
        assert set(restored.tier_status) == set(main.TIERS)
        # The reverse index came back too, so approve_material can still
        # find the owning curriculum mission for a flagged record.
        assert any(rid == curriculum.id
                   for rid in main.RECORD_TO_CURRICULUM.values())


# --------------------------------------------------------------------------- #
# N. Autonomous learning pipeline: submission -> resolution, HITL escalation,
#    and offline (mock) model modes.
#
# These run the real AutonomousAgentEngine against a scratch SQLite database
# with the same schema as data/lessons.db — no model provider needed.
# --------------------------------------------------------------------------- #


class TestAutonomousLearningPipeline:
    async def _engine(self, tmp_path):
        from app.interpreter.engine import AutonomousAgentEngine

        engine = AutonomousAgentEngine(tmp_path / "lessons.db", poll_interval=0.02)
        await engine.start(run_worker=False)
        return engine

    def test_full_pipeline_from_submission_to_automated_resolution(self, tmp_path):
        """A correct submission flows ingestion -> assessment -> ledger, silently."""
        import aiosqlite

        async def scenario():
            engine = await self._engine(tmp_path)
            try:
                sid = await engine.ingest_submission(
                    "student-1", "counter-loops",
                    "SET counter TO 0\nREPEAT 3 TIMES\n"
                    "SET counter TO counter + 1\nEND REPEAT\nPRINT counter",
                    expected_output=["3"],
                )
                await engine.drain_once()

                async with aiosqlite.connect(engine.db_path) as db:
                    db.row_factory = aiosqlite.Row
                    sub = dict(await (await db.execute(
                        "SELECT * FROM student_submissions WHERE id = ?", (sid,)
                    )).fetchone())
                    milestone = dict(await (await db.execute(
                        "SELECT * FROM student_milestones WHERE student_id = 'student-1'"
                    )).fetchone())
                assert sub["status"] == "auto_passed"
                assert sub["attempts"] == 1
                assert milestone["state"] == "completed"
                assert milestone["consecutive_failures"] == 0
                # Silent means silent: no human decision was ever opened.
                assert await engine.open_decisions() == []
            finally:
                await engine.close()

        asyncio.run(scenario())

    def test_hitl_escalation_triggers_when_threshold_is_met(self, tmp_path):
        """
        Three consecutive failures stay autonomous; the failure beyond the
        threshold pauses the work and opens exactly one decision brief, and
        the mentor's resolution moves the pipeline to its final state.
        """
        import aiosqlite

        async def scenario():
            engine = await self._engine(tmp_path)
            try:
                ids = []
                for _ in range(4):
                    ids.append(await engine.ingest_submission(
                        "student-2", "while-loops",
                        "WHILE n < 3\nPRINT n\nEND WHILE",
                    ))
                    await engine.drain_once()

                briefs = await engine.open_decisions()
                assert len(briefs) == 1
                brief = briefs[0]
                assert brief.category == "stuck_loop"
                assert brief.submission_id == ids[-1]
                assert brief.options == ("approve", "reject", "advise")

                # The paused submission and blocked milestone are durable.
                async with aiosqlite.connect(engine.db_path) as db:
                    async with db.execute(
                        "SELECT status FROM student_submissions WHERE id = ?",
                        (ids[-1],),
                    ) as cursor:
                        assert (await cursor.fetchone())[0] == "escalated"
                    async with db.execute(
                        "SELECT state FROM student_milestones WHERE student_id = 'student-2'"
                    ) as cursor:
                        assert (await cursor.fetchone())[0] == "blocked"

                resolved = await engine.resolve_decision(
                    brief.decision_id,
                    "Walk through the loop body line by line with them.",
                    resolution="advise",
                )
                assert resolved.status == "resolved"
                assert resolved.human_input.startswith("Walk through")
                # The gate closed: nothing remains open after resolution.
                assert await engine.open_decisions() == []
            finally:
                await engine.close()

        asyncio.run(scenario())

    def test_bedrock_mock_and_fallback_modes_for_offline_testing(self, tmp_path):
        """
        Offline modes, verified without any network:
          * MODEL_PROVIDER=mock -> the app reports generation unavailable and
            the engine composes escalation briefs deterministically;
          * the engine's model resolver degrades to None instead of raising,
            so the autonomous loop never depends on a credential;
          * briefs still open, carry full evidence, and are resolvable.
        """
        async def scenario():
            engine = await self._engine(tmp_path)
            try:
                assert main.MODEL_PROVIDER == "mock", "suite runs offline by default"
                assert main.provider_unavailable_reason() is not None
                assert engine.reasoning_enabled is False

                sid = await engine.ingest_submission(
                    "student-3", "predict-output",
                    "SET x TO 5\nPRINT x + 1",
                    expected_output=["6"],
                    claimed_output="7",  # contradiction -> hallucination_risk
                )
                await engine.drain_once()
                briefs = await engine.open_decisions()
                assert [b.category for b in briefs] == ["hallucination_risk"]
                brief = briefs[0]
                assert brief.evidence["claim"]["hallucination_risk"] is True
                assert brief.evidence["unit_tests"]["passed"] == 1

                resolved = await engine.resolve_decision(
                    brief.decision_id, "The material's expected output is wrong.",
                    resolution="reject",
                )
                assert resolved.status == "resolved"
                import aiosqlite
                async with aiosqlite.connect(engine.db_path) as db:
                    async with db.execute(
                        "SELECT status FROM student_submissions WHERE id = ?", (sid,)
                    ) as cursor:
                        assert (await cursor.fetchone())[0] == "auto_failed"
            finally:
                await engine.close()

        asyncio.run(scenario())
