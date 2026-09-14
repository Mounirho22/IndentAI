"""
batch_test.py — simulates a teacher queueing a full week of CS lessons at once.
Run the FastAPI server first, then run this script.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import aiohttp

ENDPOINT = "http://127.0.0.1:8000/process-lesson"
REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=15)

LESSON_QUEUE: list[dict[str, str]] = [
    {
        "unit_name": "Unit 2: Foundations of Programming",
        "topic": "Variables and Data Types",
        "objective": (
            "Students can declare variables in pseudo-code and choose the correct data "
            "type (number, text, or true/false) for a given value."
        ),
    },
    {
        "unit_name": "Unit 2: Foundations of Programming",
        "topic": "Conditionals (IF/ELSE)",
        "objective": (
            "Students can trace an IF/ELSE statement to predict its output and write a "
            "two-branch conditional that solves a given scenario."
        ),
    },
    {
        "unit_name": "Unit 3: Data and Iteration",
        "topic": "Arrays and Lists",
        "objective": (
            "Students can access an array element by its index and explain why the first "
            "element is at index 0."
        ),
    },
    {
        "unit_name": "Unit 3: Data and Iteration",
        "topic": "Functions and Parameters",
        "objective": (
            "Students can define a function with one parameter and explain how the "
            "returned value is used by the code that called it."
        ),
    },
]

LINE = "=" * 78
THIN = "-" * 78


async def queue_lesson(
    session: aiohttp.ClientSession, index: int, lesson: dict[str, str]
) -> dict[str, Any]:
    """POST one lesson and capture its outcome without raising."""
    started = time.perf_counter()
    try:
        async with session.post(ENDPOINT, json=lesson) as response:
            # /process-lesson returns 202 immediately; the body confirms what was queued.
            body: Any = (
                await response.json()
                if "application/json" in response.headers.get("Content-Type", "")
                else await response.text()
            )
            return {
                "index": index,
                "topic": lesson["topic"],
                "status": response.status,
                "body": body,
                "elapsed": time.perf_counter() - started,
                "error": None,
            }
    except Exception as exc:  # network refused, timeout, malformed response
        return {
            "index": index,
            "topic": lesson["topic"],
            "status": None,
            "body": None,
            "elapsed": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


def print_header() -> None:
    print(f"\n{LINE}")
    print("  TEACHER QUEUE — batch submission of one week of CS lessons".upper())
    print(f"{LINE}")
    print(f"  Target endpoint : {ENDPOINT}")
    print(f"  Lessons queued  : {len(LESSON_QUEUE)}")
    print(f"  Expected records: {len(LESSON_QUEUE) * 3} (3 tiers per lesson)")
    print(f"{THIN}")
    for i, lesson in enumerate(LESSON_QUEUE, start=1):
        print(f"  [{i}] {lesson['topic']}")
        print(f"      unit      : {lesson['unit_name']}")
        print(f"      objective : {lesson['objective'][:64]}...")
    print(f"{THIN}")
    print("  Firing all requests concurrently...\n")


def print_results(results: list[dict[str, Any]], wall_clock: float) -> int:
    print(f"{LINE}")
    print("  SERVER RESPONSES")
    print(f"{LINE}")

    accepted = 0
    for result in sorted(results, key=lambda r: r["index"]):
        tag = f"[{result['index']}] {result['topic']}"
        if result["error"] is not None:
            print(f"  FAILED   {tag}")
            print(f"           {result['error']}")
        elif result["status"] == 202:
            accepted += 1
            message = ""
            if isinstance(result["body"], dict):
                message = result["body"].get("message", "")
            print(f"  ACCEPTED {tag}  (202 in {result['elapsed'] * 1000:.0f} ms)")
            if message:
                print(f"           {message}")
        else:
            print(f"  REJECTED {tag}  (HTTP {result['status']})")
            print(f"           {result['body']}")

    print(f"{THIN}")
    print(f"  Accepted {accepted}/{len(results)} in {wall_clock:.2f}s wall clock")
    print(f"{LINE}")
    return accepted


def print_next_steps(accepted: int) -> None:
    if accepted == 0:
        print("\n  No lessons were accepted. Is the server running?")
        print("  Start it with: uvicorn main:app --reload --host 127.0.0.1 --port 8000\n")
        return

    print("\n  BACKGROUND ORCHESTRATION IS NOW RUNNING")
    print(f"{THIN}")
    print("  The API has already replied — the agent is still working. Watch the Uvicorn")
    print("  terminal to see each lesson orchestrate itself in real time:")
    print()
    print("    'Agent run started'      -> agent picked up the lesson")
    print("    'Generated worksheet'    -> one tier's worksheet came back from Gemini")
    print("    'Generated quiz + key'   -> matching quiz and answer key parsed from JSON")
    print("    'Saved materials | id=N' -> row committed to lessons.db as status='draft'")
    print("    'HUMAN REVIEW REQUIRED'  -> objective was ambiguous, nothing was generated")
    print()
    print(f"  Expect up to {accepted * 3} saved rows once the queue drains.")
    print("  Then inspect the results with:")
    print()
    print("    curl http://127.0.0.1:8000/materials")
    print()
    print(f"{LINE}\n")


async def main() -> None:
    print_header()
    started = time.perf_counter()
    async with aiohttp.ClientSession(timeout=REQUEST_TIMEOUT) as session:
        tasks = [
            queue_lesson(session, i, lesson)
            for i, lesson in enumerate(LESSON_QUEUE, start=1)
        ]
        results = await asyncio.gather(*tasks)
    wall_clock = time.perf_counter() - started

    accepted = print_results(list(results), wall_clock)
    print_next_steps(accepted)


if __name__ == "__main__":
    asyncio.run(main())
