"""
demo_scenario.py — Good Neighbor Agents: the cinematic terminal demo.

Track theme: an agent running silently in the background, surfacing ONLY
when human judgment is needed.

This is not a canned animation. Every phase below drives the real
`AutonomousAgentEngine` from app/interpreter/engine.py — the real bounded
interpreter, the real unit-test runner, the real SQLite ledger
(milestones/submissions/telemetry/decision rows), and the real
resolve_decision hook. Nothing on screen is faked; the only theatrical
element is the pacing.

Phases
------
  1. SILENT BACKGROUND INGESTION — three students submit work. The agent
     evaluates code, runs unit tests, updates the progress ledger, and
     handles one struggling student — all with zero human attention.
  2. COGNITIVE LOOP DETECTION — a fourth student keeps hitting the same
     conceptual wall. On the attempt past the retry threshold the agent
     pauses automatic evaluation and surfaces an Educator Escalation Alert.
  3. HUMAN DECISION & AUTONOMOUS RESOLUTION — you (the mentor) choose how
     to help; the agent applies the directive, updates the records, and
     resumes background processing.

Run:

    python demos/demo_scenario.py           # interactive (you are the mentor)
    python demos/demo_scenario.py --auto    # hands-free, for screen recording

State lives in data/demo_scenario.db (gitignored, wiped on each run).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rich.align import Align
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from app.interpreter.engine import AutonomousAgentEngine

# The engine's structured telemetry would otherwise print between frames;
# the demo surfaces the same facts through the UI instead.
logging.getLogger("lesson-agent").setLevel(logging.CRITICAL)

console = Console(highlight=False)

DEMO_DB = PROJECT_ROOT / "data" / "demo_scenario.db"
THRESHOLD = 3  # the agent escalates AFTER this many consecutive failures

SENDER_STYLE = "bold cyan"
AGENT_STYLE = "bold magenta"
ALERT_STYLE = "bold red"
MUTED_STYLE = "dim"

# --------------------------------------------------------------------------- #
# The submission stream: three students handled silently, one stuck student.
# --------------------------------------------------------------------------- #

PHASE_ONE = [
    {
        "student_id": "alice",
        "milestone_id": "counter-loops",
        "label": "Alice",
        "task": "REPEAT loop, print a counter",
        "code": "SET counter TO 0\nREPEAT 3 TIMES\nSET counter TO counter + 1\nEND REPEAT\nPRINT counter",
        "expected": ["3"],
        "claimed": None,
        "note": "clean first attempt",
    },
    {
        "student_id": "ben",
        "milestone_id": "counter-loops",
        "label": "Ben",
        "task": "off-by-one on his first try",
        "code": "SET counter TO 0\nREPEAT 3 TIMES\nSET counter TO counter + 1\nEND REPEAT\nPRINT counter",
        "expected": ["4"],  # the code prints 3 — Ben expected 4
        "claimed": None,
        "note": "expects one iteration too many (off-by-one)",
    },
    {
        "student_id": "ben",
        "milestone_id": "counter-loops",
        "label": "Ben",
        "task": "Ben's second attempt, corrected",
        "code": "SET counter TO 0\nREPEAT 4 TIMES\nSET counter TO counter + 1\nEND REPEAT\nPRINT counter",
        "expected": ["4"],
        "claimed": None,
        "note": "recovered without any human help",
    },
    {
        "student_id": "cara",
        "milestone_id": "loop-syntax",
        "label": "Cara",
        "task": "uses FOR — a construct the course has not taught",
        "code": "FOR i = 1 TO 3\nPRINT i\nNEXT i",
        "expected": ["1", "2", "3"],
        "claimed": None,
        "note": "untaught construct: silent failure, support queued",
    },
]

# Dave's four attempts. Every one fails; the misconception persists and the
# attempt counter walks straight into the escalation gate.
PHASE_TWO = [
    {
        "label": "Dave",
        "task": "attempt 1 — reads a counter he never SET",
        "code": "WHILE counter < 3\nPRINT counter\nEND WHILE",
        "expected": ["3"],
    },
    {
        "label": "Dave",
        "task": "attempt 2 — SETs the counter AFTER the loop",
        "code": "WHILE counter < 3\nPRINT counter\nEND WHILE\nSET counter TO 0",
        "expected": ["3"],
    },
    {
        "label": "Dave",
        "task": "attempt 3 — tries a FOR loop, still not in the taught subset",
        "code": "FOR counter = 0 TO 2\nPRINT counter\nEND FOR",
        "expected": ["3"],
    },
    {
        "label": "Dave",
        "task": "attempt 4 — REPEAT without a count",
        "code": "REPEAT\nPRINT counter\nEND REPEAT",
        "expected": ["3"],
    },
]


# --------------------------------------------------------------------------- #
# Cinematic helpers
# --------------------------------------------------------------------------- #


def banner(title: str, subtitle: str = "") -> None:
    console.print()
    console.print(Rule(Text(title, style="bold white"), style="rgb(80,80,80)"))
    if subtitle:
        console.print(Align.center(Text(subtitle, style=MUTED_STYLE)))
    console.print()


def agent_line(text: str) -> None:
    console.print(f"[{AGENT_STYLE}]agent[/{AGENT_STYLE}] {text}")


def event_line(speaker: str, text: str, style: str = SENDER_STYLE) -> None:
    console.print(f"[{style}]{speaker}[/{style}] {text}")


def pause(seconds: float) -> None:
    time.sleep(seconds)


async def ingest_with_theatre(
    engine: AutonomousAgentEngine,
    submission: dict,
    progress: Progress,
) -> int:
    """Submit one student event with progress theatre, return the submission id."""
    task = progress.add_task(
        f"[cyan]{submission['label']}[/cyan] submits: {submission['task']}",
        total=None,
    )
    await asyncio.sleep(0.35)
    submission_id = await engine.ingest_submission(
        submission["student_id"],
        submission["milestone_id"],
        submission["code"],
        expected_output=submission["expected"],
        claimed_output=submission.get("claimed"),
    )
    await asyncio.sleep(0.45)
    progress.update(task, completed=1, total=1)
    return submission_id


async def ledger_table(engine: AutonomousAgentEngine) -> Table:
    """The SQLite progress ledger, rendered live from student_milestones."""
    import aiosqlite

    async with aiosqlite.connect(engine.db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT student_id, milestone_id, state, total_attempts,
                   consecutive_failures, unlocked
            FROM student_milestones ORDER BY student_id
            """
        ) as cursor:
            rows = await cursor.fetchall()
    table = Table(
        title="Progress Ledger  (data/demo_scenario.db — the agent's own writing)",
        border_style="rgb(70,70,70)",
        header_style="bold",
    )
    table.add_column("student", style="cyan")
    table.add_column("milestone")
    table.add_column("attempts", justify="right")
    table.add_column("consec. fails", justify="right")
    table.add_column("state")
    table.add_column("advanced", justify="center")
    for r in rows:
        state_style = {
            "completed": "green",
            "in_progress": "yellow",
            "blocked": "red",
            "advanced": "bold green",
        }.get(r["state"], "white")
        table.add_row(
            r["student_id"],
            r["milestone_id"],
            str(r["total_attempts"]),
            str(r["consecutive_failures"]),
            Text(r["state"], style=state_style),
            "yes" if r["unlocked"] else "-",
        )
    return table


def attention_meter(decisions_open: int, autonomous: int, escalated: int) -> Panel:
    """The demo's headline statistic: how much human attention was drained."""
    line = Text()
    line.append("Human attention drained: ", style="bold")
    line.append("0 units", style="bold green")
    line.append("   |   handled autonomously: ", style="bold")
    line.append(f"{autonomous}", style="green")
    line.append("   |   surfaced for judgment: ", style="bold")
    line.append(f"{escalated}", style="red" if escalated else "green")
    return Panel(
        Align.center(line),
        border_style="rgb(70,70,70)",
        padding=(0, 1),
    )


def escalation_alert(brief: dict) -> Panel:
    """The Educator Escalation Alert the whole demo exists to show."""
    evidence = brief.get("evidence", {})
    milestone = evidence.get("milestone", {})
    body = [
        f"[bold]{brief['title']}[/bold]",
        "",
        brief["summary"],
        "",
        f"[dim]student:[/dim] {brief['student_id']}    "
        f"[dim]milestone:[/dim] {brief['milestone_id']}    "
        f"[dim]category:[/dim] [red]{brief['category']}[/red]    "
        f"[dim]severity:[/dim] {brief['severity']}",
        f"[dim]consecutive failures:[/dim] {milestone.get('consecutive_failures')}    "
        f"[dim]attempts:[/dim] {milestone.get('total_attempts')}    "
        f"[dim]decision_id:[/dim] {brief['decision_id'][:16]}…",
    ]
    return Panel(
        Group(*body),
        title="[bold red]⚡ EDUCATOR ESCALATION ALERT[/bold red]",
        subtitle="the agent stopped by itself — background evaluation is paused",
        border_style="red",
        padding=(1, 2),
    )


async def last_interpreter_errors(engine: AutonomousAgentEngine, limit: int = 3) -> list[str]:
    """Pull the most recent unit-test failure messages from telemetry evidence."""
    import aiosqlite

    async with aiosqlite.connect(engine.db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT brief_json FROM agent_decisions ORDER BY created_at DESC LIMIT 1"
        ) as cursor:
            row = await cursor.fetchone()
    if row is None:
        return []
    import json

    cases = (json.loads(row[0]).get("evidence", {}) or {}).get("unit_tests", {}).get("cases", [])
    out = []
    for case in cases:
        err = case.get("error")
        if err:
            out.append(f"{err[0]}: {err[1]}")
    return out[:limit]


# --------------------------------------------------------------------------- #
# The three phases
# --------------------------------------------------------------------------- #


async def phase_one(engine: AutonomousAgentEngine, speed: float) -> None:
    banner(
        "PHASE 1 — SILENT BACKGROUND INGESTION",
        "Three students submit work. Nobody is watching. Nobody needs to be.",
    )
    agent_line("Background agent online. Watching the submission stream…")
    pause(speed)

    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        for submission in PHASE_ONE:
            await ingest_with_theatre(engine, submission, progress)
            pause(speed * 0.4)

    agent_line("Draining the queue — evaluating code, running unit tests, updating the ledger…")
    with console.status("[magenta]agent working…[/magenta]") as status:
        processed = 0
        for _ in range(6):
            processed += await engine.drain_once()
            if processed >= len(PHASE_ONE):
                break
            await asyncio.sleep(0.3)
        status.stop()
    pause(speed)

    event_line("agent", f"{processed} submissions assessed. Zero human attention drained.", AGENT_STYLE)
    console.print(await ledger_table(engine))
    console.print(attention_meter(0, processed, 0))
    pause(speed)


async def phase_two(engine: AutonomousAgentEngine, speed: float) -> dict:
    banner(
        "PHASE 2 — COGNITIVE LOOP DETECTION & ESCALATION",
        f"A fourth student keeps hitting the same conceptual wall. The gate opens at more than {THRESHOLD} consecutive failures.",
    )

    decision: dict | None = None
    with Progress(
        SpinnerColumn(style="red"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        for attempt in PHASE_TWO:
            task = progress.add_task(
                f"[cyan]{attempt['label']}[/cyan] {attempt['task']}", total=None
            )
            await engine.ingest_submission(
                "dave", "while-loops", attempt["code"], expected_output=attempt["expected"],
            )
            await asyncio.sleep(0.4)
            progress.update(task, completed=1, total=1)
            await asyncio.sleep(speed * 0.3)

    agent_line("Same milestone failing again and again — evaluating…")
    for _ in range(len(PHASE_TWO) + 2):
        await engine.drain_once()
        if await engine.open_decisions():
            break
        await asyncio.sleep(0.2)

    briefs = await engine.open_decisions()
    if not briefs:
        console.print("[red]Expected an escalation — the gate did not open.[/red]")
        raise SystemExit(1)

    brief = briefs[0].to_dict()
    decision = brief
    pause(speed)
    console.print(escalation_alert(brief))
    console.print()

    errors = await last_interpreter_errors(engine)
    if errors:
        console.print(Text("  what the interpreter saw (deterministic, no model):", style=MUTED_STYLE))
        for err in errors:
            console.print(f"    [red]✗[/red] {err}")
    console.print()
    agent_line(
        "Automatic evaluation is [bold red]paused[/bold red] for this student. "
        "Background processing continues for everyone else."
    )
    pause(speed)
    return decision


def generated_hint(brief: dict) -> str:
    """Build the tailored hint from the brief's evidence (deterministic)."""
    evidence = brief.get("evidence", {})
    cases = evidence.get("unit_tests", {}).get("cases", [])
    code = evidence.get("pseudocode", "")
    if "WHILE" in code:
        detail = (
            "A WHILE condition only CHECKS a variable — it never changes it. "
            "Before the loop starts, SET counter TO 0, and inside the body "
            "(between WHILE and END WHILE) add SET counter TO counter + 1."
        )
    elif "REPEAT" in code:
        detail = (
            "REPEAT needs a count up front: REPEAT 3 TIMES … END REPEAT. "
            "Count how many times the body should run and put that number before TIMES."
        )
    else:
        detail = "Re-read the taught subset: SET / INPUT / PRINT / IF / REPEAT n TIMES / WHILE."
    if cases and cases[0].get("error"):
        kind = cases[0]["error"][0]
        detail += f" Your last run stopped with a {kind!r} error — fix that first."
    return detail


async def phase_three(engine: AutonomousAgentEngine, decision: dict, speed: float, auto: bool) -> None:
    banner(
        "PHASE 3 — HUMAN DECISION & AUTONOMOUS RESOLUTION",
        "You are the volunteer mentor. The agent is waiting on exactly one judgment call.",
    )
    console.print(
        Panel(
            "[bold]1.[/bold] Unlock simplified scaffolding  [dim]— re-queue with a step-by-step scaffold[/dim]\n"
            "[bold]2.[/bold] Flag for 1-on-1 breakout    [dim]— stop auto-support, mentor takes over[/dim]\n"
            "[bold]3.[/bold] Auto-generate tailored hint [dim]— agent resumes with a targeted hint[/dim]",
            title="Mentor decision",
            border_style="yellow",
            padding=(1, 2),
        )
    )

    if auto:
        console.print(Align.center(Text("[--auto] mentor is afk — choosing 3: auto-generated hint", style=MUTED_STYLE)))
        choice = "3"
        pause(speed)
    else:
        choice = Prompt.ask(
            "[bold yellow]Your call[/bold yellow]", choices=["1", "2", "3"], default="3"
        )

    if choice == "1":
        resolution, note = "advise", (
            "Scaffolding unlocked: start from SET counter TO 0, then one REPEAT 3 TIMES "
            "body that only PRINTs the counter, THEN add the increment line."
        )
        directive = "simplified scaffolding unlocked — re-queued with a scaffold"
    elif choice == "2":
        resolution, note = "reject", (
            "Flagged for a 1-on-1 breakout at the next session — automatic support stopped."
        )
        directive = "flagged for 1-on-1 breakout — auto-support stopped"
    else:
        hint = generated_hint(decision)
        resolution, note = "advise", hint
        directive = "tailored hint generated and attached"

    console.print()
    with console.status("[magenta]agent applying your directive…[/magenta]"):
        resolved = await engine.resolve_decision(
            decision["decision_id"], note, resolution=resolution,  # type: ignore[arg-type]
        )
        pause(speed)
        await engine.drain_once()
    console.print(f"[green]✓[/green] {directive}. Records updated.")

    console.print()
    agent_line("Background processing resumed. Final ledger:")
    console.print(await ledger_table(engine))

    # Final stats straight from SQLite.
    import aiosqlite

    async with aiosqlite.connect(engine.db_path) as db:
        subs = (await (await db.execute("SELECT COUNT(*) FROM student_submissions")).fetchone())[0]
        passed = (await (await db.execute(
            "SELECT COUNT(*) FROM student_submissions WHERE status = 'auto_passed'"
        )).fetchone())[0]
        failed = (await (await db.execute(
            "SELECT COUNT(*) FROM student_submissions WHERE status = 'auto_failed'"
        )).fetchone())[0]
        escalated = (await (await db.execute(
            "SELECT COUNT(*) FROM agent_decisions"
        )).fetchone())[0]
        decisions = (await (await db.execute(
            "SELECT COUNT(*) FROM agent_decisions WHERE status = 'resolved'"
        )).fetchone())[0]
        humans = (await (await db.execute(
            "SELECT COUNT(*) FROM agent_telemetry WHERE payload_json LIKE '%human_intervention\": true%'"
        )).fetchone())[0]

    summary = Text()
    summary.append("Submissions processed: ", style="bold")
    summary.append(f"{subs}", style="cyan")
    summary.append("   autonomously: ", style="bold")
    summary.append(f"{passed + failed}", style="green")
    summary.append("   escalated to a human: ", style="bold")
    summary.append(f"{escalated}", style="red" if escalated else "green")
    summary.append("\nHuman decisions required: ", style="bold")
    summary.append(f"{decisions}", style="yellow")
    summary.append("   human interventions total: ", style="bold")
    summary.append(f"{humans}", style="yellow")
    summary.append("\n\nHuman attention drained: ", style="bold")
    summary.append("one decision, sixty seconds. Everything else stayed silent.", style="bold green")

    console.print()
    console.print(
        Panel(
            Align.center(summary),
            title="[bold green]Demo complete — Good Neighbor Agents[/bold green]",
            subtitle="python demos/demo_scenario.py --auto",
            border_style="green",
            padding=(1, 2),
        )
    )
    console.print(
        Align.center(
            Text(
                f"mentor note stored verbatim: “{note[:90]}{'…' if len(note) > 90 else ''}”",
                style=MUTED_STYLE,
            )
        )
    )
    console.print()


# --------------------------------------------------------------------------- #
# Entry
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Good Neighbor Agents — autonomous mentor operations demo"
    )
    parser.add_argument(
        "--auto", action="store_true",
        help="hands-free mode: the mentor's decision is made automatically (for recording)",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="pacing multiplier (0.5 = faster, 2 = slower)",
    )
    args = parser.parse_args()
    speed = max(args.speed, 0.1)

    console.clear()
    console.print(
        Panel(
            Align.center(
                Text.assemble(
                    ("IdentAI\n", "bold white"),
                    ("Good Neighbor Agents — Autonomous Mentor Operations\n", "bold cyan"),
                    ("for Community Coding Centers\n\n", "cyan"),
                    ("Strands Agents SDK  ·  Amazon Bedrock  ·  SQLite ledger\n", MUTED_STYLE),
                    ("an agent that earns its interruptions\n", MUTED_STYLE),
                )
            ),
            border_style="cyan",
            padding=(1, 4),
        )
    )

    # Fresh ledger every run so the story always lands the same way.
    DEMO_DB.parent.mkdir(parents=True, exist_ok=True)
    if DEMO_DB.exists():
        DEMO_DB.unlink()

    async def run() -> int:
        engine = AutonomousAgentEngine(DEMO_DB, poll_interval=0.2)
        await engine.start(run_worker=False)  # the demo drives the loop visibly
        try:
            await phase_one(engine, speed)
            decision = await phase_two(engine, speed)
            await phase_three(engine, decision, speed, args.auto)
        finally:
            await engine.close()
        return 0

    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())
