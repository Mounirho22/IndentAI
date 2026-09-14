import unittest
from app.interpreter import (
    ExecutionResult,
    execute_pseudocode,
    interp_eval,
    normalise_output_line,
    outputs_match,
)


class TestInterpreterEvaluator(unittest.TestCase):
    def test_arithmetic(self):
        env = {"x": 10, "y": 2}
        self.assertEqual(interp_eval("x + y", env), 12)
        self.assertEqual(interp_eval("x - y", env), 8)
        self.assertEqual(interp_eval("x * y", env), 20)
        self.assertEqual(interp_eval("x / y", env), 5.0)

    def test_comparisons_and_booleans(self):
        env = {"a": 5, "b": 10}
        self.assertTrue(interp_eval("a < b", env))
        self.assertFalse(interp_eval("a > b", env))
        self.assertTrue(interp_eval("a <= 5", env))
        self.assertTrue(interp_eval("b >= 10", env))
        self.assertTrue(interp_eval("a = 5", env))
        self.assertTrue(interp_eval("a <> b", env))
        self.assertTrue(interp_eval("a < b AND b = 10", env))
        self.assertTrue(interp_eval("a > b OR b = 10", env))
        self.assertFalse(interp_eval("NOT (a < b)", env))

    def test_division_by_zero(self):
        with self.assertRaises(ZeroDivisionError):
            interp_eval("10 / 0", {})


class TestInterpreterExecution(unittest.TestCase):
    def test_repeat_loop_and_dict_access(self):
        code = """
        SET count TO 0
        REPEAT 3 TIMES
            SET count TO count + 1
            PRINT count
        END REPEAT
        """
        res = execute_pseudocode(code)
        self.assertTrue(res.ok)
        self.assertTrue(res["ok"])  # test dict subscript
        self.assertEqual(res.output, ["1", "2", "3"])
        self.assertEqual(res["output"], ["1", "2", "3"])
        self.assertIsNone(res.error)

    def test_while_loop(self):
        code = """
        SET x TO 3
        WHILE x > 0
            PRINT x
            SET x TO x - 1
        END WHILE
        """
        res = execute_pseudocode(code)
        self.assertTrue(res.ok)
        self.assertEqual(res.output, ["3", "2", "1"])

    def test_if_else_if_else(self):
        code = """
        SET score TO 75
        IF score >= 90 THEN
            PRINT "A"
        ELSE IF score >= 70 THEN
            PRINT "B"
        ELSE
            PRINT "C"
        END IF
        """
        res = execute_pseudocode(code)
        self.assertTrue(res.ok)
        self.assertEqual(res.output, ["B"])

    def test_infinite_loop_step_cap(self):
        code = """
        SET x TO 1
        WHILE x > 0
            SET x TO x + 1
        END WHILE
        """
        res = execute_pseudocode(code, step_cap=100)
        self.assertFalse(res.ok)
        self.assertEqual(res.error[0], "infinite")

    def test_normalise_and_matching(self):
        self.assertEqual(normalise_output_line(" Hello, World! . "), "hello, world!")
        self.assertTrue(outputs_match(["1", " 2."], ["1", "2"]))
        self.assertFalse(outputs_match(["1", "2"], ["1", "3"]))


# --------------------------------------------------------------------------- #
# Submission parsing and rule sets (background-agent intake surface)
# --------------------------------------------------------------------------- #

import asyncio
import tempfile
import time
from pathlib import Path

from app.interpreter import code_lines, leading_keyword
from app.interpreter.engine import (
    INTERPRETER_STEP_CAP,
    PSEUDO_KEYWORDS,
    AutonomousAgentEngine,
)


class TestSubmissionParsingAndRules(unittest.TestCase):
    def test_code_lines_extracts_pseudocode_from_prose(self):
        mixed = (
            "Question 2 — predict the output.\n"
            "SET total TO 0\n"
            "\n"
            "Read the loop carefully before answering.\n"
            "REPEAT 2 TIMES\n"
            "SET total TO total + 2\n"
            "END REPEAT\n"
            "PRINT total"
        )
        found = code_lines(mixed)
        self.assertEqual(
            [line for _, line in found],
            ["SET total TO 0", "REPEAT 2 TIMES", "SET total TO total + 2",
             "END REPEAT", "PRINT total"],
        )
        # Line numbers stay anchored to the original text for audits.
        self.assertEqual(found[0][0], 2)

    def test_leading_keyword_rule_set(self):
        self.assertEqual(leading_keyword("repeat 3 times"), "REPEAT")
        self.assertEqual(leading_keyword("ELSE IF x > 1 THEN"), "ELSE IF")
        self.assertEqual(leading_keyword("ENDIF"), "ENDIF")
        self.assertIsNone(leading_keyword("SETX 5"))       # no invented keywords
        self.assertIsNone(leading_keyword("just prose"))

    def test_taught_vocabulary_is_complete(self):
        for required in ("SET", "INPUT", "PRINT", "IF", "ELSE", "REPEAT", "WHILE", "END"):
            self.assertIn(required, PSEUDO_KEYWORDS)

    def test_prose_only_submission_is_a_typed_syntax_error(self):
        result = execute_pseudocode("This homework is about loops. Enjoy!")
        self.assertFalse(result.ok)
        self.assertEqual(result.error[0], "syntax")


class TestBackgroundExecutionWithoutBlocking(unittest.TestCase):
    def test_step_cap_bounds_every_run(self):
        # A non-terminating submission must return bounded, not hang.
        started = time.monotonic()
        result = execute_pseudocode("SET x TO 0\nWHILE true\nPRINT x\nEND WHILE")
        elapsed = time.monotonic() - started
        self.assertFalse(result.ok)
        self.assertEqual(result.error[0], "infinite")
        # The cap counts the boundary step that trips it, hence +1.
        self.assertEqual(result.steps, INTERPRETER_STEP_CAP + 1)
        self.assertLess(elapsed, 5.0)

    def test_ingest_returns_without_processing_inline(self):
        """Ingestion is fire-and-forget: the row lands pending, assessment is deferred."""
        async def scenario():
            with tempfile.TemporaryDirectory() as tmp:
                engine = AutonomousAgentEngine(Path(tmp) / "t.db")
                await engine.start(run_worker=False)
                try:
                    sid = await engine.ingest_submission(
                        "s1", "m1", "SET x TO 1\nPRINT x", expected_output=["1"],
                    )
                    status = await _status_of(engine, sid)
                    self.assertEqual(status, "pending")
                    processed = await engine.drain_once()
                    self.assertEqual(processed, 1)
                    self.assertEqual(await _status_of(engine, sid), "auto_passed")
                finally:
                    await engine.close()

        asyncio.run(scenario())

    def test_worker_loop_processes_while_caller_keeps_working(self):
        """The spawned worker empties the queue concurrently with other tasks."""
        async def scenario():
            with tempfile.TemporaryDirectory() as tmp:
                engine = AutonomousAgentEngine(Path(tmp) / "t.db", poll_interval=0.02)
                await engine.start()  # worker task running
                try:
                    ids = [
                        await engine.ingest_submission("s1", "m1", "SET x TO 1\nPRINT x")
                        for _ in range(3)
                    ]
                    # Meanwhile the "application" does its own unrelated work.
                    other = asyncio.create_task(asyncio.sleep(0.15))
                    deadline = time.monotonic() + 5.0
                    while time.monotonic() < deadline:
                        statuses = [await _status_of(engine, i) for i in ids]
                        if all(s == "auto_passed" for s in statuses):
                            break
                        await asyncio.sleep(0.05)
                    await other
                    statuses = [await _status_of(engine, i) for i in ids]
                    self.assertEqual(statuses, ["auto_passed"] * 3)
                finally:
                    await engine.close()

        asyncio.run(scenario())


async def _status_of(engine: AutonomousAgentEngine, submission_id: int) -> str:
    import aiosqlite

    async with aiosqlite.connect(engine.db_path) as db:
        async with db.execute(
            "SELECT status FROM student_submissions WHERE id = ?", (submission_id,)
        ) as cursor:
            row = await cursor.fetchone()
    return str(row[0]) if row else "missing"


class TestLessonsDbStateMutations(unittest.TestCase):
    """
    The agent's own writes: submissions, milestone counters, decisions and
    telemetry must all land in SQLite (exercised against a scratch database
    with the same schema as data/lessons.db).
    """

    def test_processing_mutates_submission_and_milestone_rows(self):
        async def scenario():
            with tempfile.TemporaryDirectory() as tmp:
                engine = AutonomousAgentEngine(Path(tmp) / "t.db")
                await engine.start(run_worker=False)
                try:
                    ok_id = await engine.ingest_submission(
                        "s1", "loops", "SET x TO 1\nPRINT x", expected_output=["1"],
                    )
                    bad_id = await engine.ingest_submission(
                        "s1", "loops", "WHILE n < 3\nPRINT n\nEND WHILE",
                    )
                    await engine.drain_once()

                    import aiosqlite

                    async with aiosqlite.connect(engine.db_path) as db:
                        db.row_factory = aiosqlite.Row
                        rows = {
                            r["id"]: r for r in (
                                await (await db.execute(
                                    "SELECT * FROM student_submissions")).fetchall())
                        }
                        milestone = dict(await (await db.execute(
                            "SELECT * FROM student_milestones WHERE student_id='s1'"
                        )).fetchone())
                        telemetry = (await (await db.execute(
                            "SELECT event FROM agent_telemetry")).fetchall())

                    self.assertEqual(rows[ok_id]["status"], "auto_passed")
                    self.assertEqual(rows[bad_id]["status"], "auto_failed")
                    self.assertEqual(rows[ok_id]["attempts"], 1)
                    self.assertEqual(milestone["total_attempts"], 2)
                    self.assertEqual(milestone["consecutive_failures"], 1)
                    events = {r["event"] for r in telemetry}
                    self.assertIn("submission_ingested", events)
                    self.assertIn("submission_auto_processed", events)
                finally:
                    await engine.close()

        asyncio.run(scenario())

    def test_escalation_and_resolution_persist_across_engine_instances(self):
        """A pause and its resolution are durable rows, not in-memory state."""
        async def scenario():
            with tempfile.TemporaryDirectory() as tmp:
                db_path = Path(tmp) / "t.db"
                engine = AutonomousAgentEngine(db_path)
                await engine.start(run_worker=False)
                for _ in range(4):
                    await engine.ingest_submission("s1", "m1", "WHILE n < 3\nPRINT n\nEND WHILE")
                    await engine.drain_once()
                (brief,) = await engine.open_decisions()
                await engine.resolve_decision(brief.decision_id, "hint", resolution="advise")
                await engine.close()

                # A brand-new engine instance reads the same durable state.
                engine2 = AutonomousAgentEngine(db_path)
                await engine2.start(run_worker=False)
                try:
                    resolved = await engine2.get_decision(brief.decision_id)
                    self.assertEqual(resolved.status, "resolved")
                    self.assertEqual(resolved.resolution, "advise")
                    self.assertEqual(resolved.human_input, "hint")
                    import aiosqlite

                    async with aiosqlite.connect(db_path) as db:
                        async with db.execute(
                            "SELECT advice FROM student_submissions WHERE id = ?",
                            (brief.submission_id,),
                        ) as cursor:
                            row = await cursor.fetchone()
                    self.assertIn("hint", row[0])
                finally:
                    await engine2.close()

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
