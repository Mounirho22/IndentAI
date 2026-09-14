# AGENT_SYNC.md

Lead Architect sync for **IdentAI**. Authoritative as of the current tree:
`main.py` (934 lines), `batch_test.py` (174 lines).

Read this before touching either file. Section 3 records one broken contract that
Phase 5 must close.

---

## 1. System Architecture Summary

Three layers, one process, one file.

| Layer | Technology | Where |
| --- | --- | --- |
| HTTP + server-rendered UI | FastAPI 0.x, `HTMLResponse` | `main.py:460-931` |
| Autonomous orchestration | `strands.Agent` + `LiteLLMModel` -> Gemini | `main.py:341-387` |
| Persistence | `aiosqlite` over `lessons.db` | `main.py:44-56`, `303-317` |

### Request path (write)

`POST /process-lesson` (`main.py:468`) validates a `LessonRequest`, hands the payload to
`BackgroundTasks`, and returns **202 Accepted** without waiting. The agent loop runs
after the response is flushed, in `run_lesson_agent` (`main.py:427`). Nothing about the
generation outcome is visible in that 202 body — only what was queued.

`build_agent()` (`main.py:376`) constructs a **fresh `Agent` per request** so no
conversation state leaks between lessons. The `LiteLLMModel` at `main.py:369` is
module-level and shared; it is instantiated at import time with `GEMINI_API_KEY`, so a
missing key produces a runtime failure inside the background task, not a startup error.

The agent owns four tools and the system prompt (`main.py:341`) forces a fixed loop —
worksheet, then quiz, then save — repeated once per tier in the order
`struggling, on-level, advanced`:

| Tool | Line | Returns | Side effect |
| --- | --- | --- | --- |
| `generate_tiered_worksheet` | `main.py:104` | worksheet text | none |
| `generate_quiz_and_key` | `main.py:196` | `{"quiz", "answer_key"}` | none |
| `save_materials` | `main.py:278` | confirmation + row id | **INSERT, status hard-coded `'draft'`** (`main.py:309`) |
| `flag_for_review` | `main.py:320` | confirmation string | **`print` + `logger.warning` only — no DB write** |

Both generation tools push the blocking `litellm.completion` call through
`asyncio.to_thread` (`main.py:142`, `main.py:237`), so concurrent tool calls and the
event loop are not stalled. `generate_quiz_and_key` defends against Gemini's JSON mode
in three stages: parse, strip ``` fences (`main.py:253`), then salvage the outermost
`{...}` (`main.py:260`) before failing the tool call.

### Read path

| Route | Line | Response |
| --- | --- | --- |
| `GET /` | `main.py:896` | Full review-desk HTML, records pre-injected |
| `GET /materials` | `main.py:485` | `list[MaterialRecord]`, `ORDER BY id DESC` |
| `POST /approve/{item_id}` | `main.py:909` | Updated row as JSON |
| `GET /health` | `main.py:928` | `{"status": "ok"}` |

`GET /` renders once with data embedded, then the page runs client-side from that
payload. Worksheet text is injected into a `<script type="application/json">` block with
`</` escaped to `<\/` (`main.py:905`), so generated content can never terminate the
script element early or be parsed as markup. Every value that reaches the DOM afterwards
goes through `esc()` (`main.py:703`), which sets `textContent` and reads back
`innerHTML`.

### Known architectural constraints

- **No status vocabulary enforcement.** `CREATE_MATERIALS_TABLE` (`main.py:44`) declares
  `status TEXT DEFAULT 'draft'` with no `CHECK`. `MaterialRecord.status` (`main.py:420`)
  is a bare `str`. Any string can land in the column.
- **Connection per operation.** Every handler and tool opens its own `aiosqlite.connect`.
  No pool, no WAL mode. Twelve inserts from four concurrent agent runs serialise on the
  default rollback journal.
- **No `created_at`, no uniqueness.** Re-queueing the same lesson silently produces a
  second set of three rows, indistinguishable from the first except by `id`.
- **Fire-and-forget failures.** `run_lesson_agent` swallows every exception into
  `logger.exception` (`main.py:442`). A failed run leaves no database trace at all, so
  the UI cannot distinguish "never queued" from "queued and crashed".

---

## 2. Phase 5 Task Checklist

> **Status correction.** All three named Phase 5 items are already implemented in the
> current tree. They are marked done below with their locations. The work that actually
> remains is listed under 2.4, which is the real Phase 5 scope.

### 2.1 Review dashboard UI layout — DONE

`REVIEW_PAGE_HTML` (`main.py:505-886`), served at `GET /` (`main.py:896`).

- Three fixed buckets driven by `SECTIONS` (`main.py:673`): Requires Review (Flagged),
  Drafts Ready for Approval, Approved Materials. Flagged sorts first by design.
- Live counts in the masthead (`main.py:657-659`), per-section tallies (`main.py:785`),
  and a per-section empty state (`main.py:782`).
- Cards show topic, a three-block tier meter (`meter()`, `main.py:720`), a status badge,
  record id, unit, objective, and up to three collapsible specimen panes for worksheet /
  quiz / answer key.
- Excerpts clamp at 240 chars on a word boundary (`main.py:727`) and fade with a CSS
  mask; "Show full text" expands in place and preserves keyboard focus via `refocus`
  (`main.py:807`).
- Accessibility: `aria-live="polite"` status region (`main.py:664`), visible
  `:focus-visible` outline (`main.py:569`), and `prefers-reduced-motion` honoured for the
  settle animation (`main.py:645`).

### 2.2 `POST /approve/{item_id}` endpoint — DONE

`approve_material` (`main.py:909-925`). Single `UPDATE ... WHERE id = ?`, `404` when
`cursor.rowcount == 0`, commit, re-`SELECT`, return the updated row.

### 2.3 Error boundaries — DONE (client side)

- `approve()` (`main.py:822`) wraps the fetch in `try/catch`, treats any non-`ok`
  response as a throw, re-enables the button, and writes a scoped message into that
  card's `.card-error` slot via `showCardError` (`main.py:817`). The record is explicitly
  described as unchanged, and the failure is announced to screen readers.
- `refresh()` (`main.py:846`) degrades to "showing the last loaded records" and restores
  its own button label in a `finally` block.
- `esc()` (`main.py:703`) is the XSS boundary for all rendered fields.

### 2.4 Remaining Phase 5 work

- [ ] **Make `'flagged'` reachable.** See section 3.3. This is the blocking item —
      without it the flagged bucket is permanently empty and Phase 5's headline feature
      is decorative.
- [ ] **Add a `reason` column** so a flagged record can explain itself in the UI.
- [ ] **Guard the transition** in `approve_material`: refuse to approve a row that is not
      `'draft'`. Verified today, an unguarded `UPDATE` happily flips `'flagged'` straight
      to `'approved'`.
- [ ] **Constrain the vocabulary**: `CHECK (status IN ('draft','approved','flagged'))` on
      the table and `Literal['draft','approved','flagged']` on `MaterialRecord.status`.
- [ ] **No server-side error boundary exists.** `GET /` has no `try/except`; if
      `lessons.db` is missing or locked, the teacher gets a bare 500 and no HTML.
- [ ] **Close the batch verification gap.** See section 4.

---

## 3. Contract Definitions

### 3.1 UI route contracts

**`GET /`** — teacher-facing review desk.

- `200 text/html` always, whatever the row count. Empty database renders three empty
  states, not an error.
- Serialises the same nine columns as `GET /materials`, newest first.
- Contract for content: generated text is **data, never markup**. Any future change to
  the injection at `main.py:905` must preserve both the `</` escape and `esc()`.

**`GET /materials`** — machine-readable mirror of the same query.

- `200 application/json`, an array ordered `id DESC`.
- Element shape is `MaterialRecord` (`main.py:411`): `id`, `unit_name`, `topic`,
  `objective`, `tier`, `worksheet_text?`, `quiz_text?`, `answer_key_text?`, `status`.
- The three text fields are nullable. Consumers must handle `null`; the UI's `specimen()`
  returns `""` for falsy text (`main.py:734`).

**`POST /approve/{item_id}`** — the only state-mutating UI route.

| Condition | Status | Body |
| --- | --- | --- |
| Row exists | `200` | `{id, unit_name, topic, tier, status}` |
| No such row | `404` | `{"detail": "No material with id=N."}` |

- Takes no request body. `item_id` is coerced to `int` by FastAPI; a non-integer path
  segment yields `422` before the handler runs.
- **Verified idempotent**, by direct SQLite experiment rather than inference: an `UPDATE`
  that assigns a column the value it already holds still reports `rowcount == 1`
  (checked on SQLite 3.37.2; a non-existent id reports `0`). Re-approving an
  already-approved row therefore returns `200`, never `404`. Note this is the opposite of
  MySQL's "rows changed" semantics — do not port the `rowcount == 0` 404 test to another
  engine without re-checking it.
- Returns a **projection, not the full record** — no `objective`, no text fields. The
  client merges only `status` back into its local copy (`main.py:833`).

### 3.2 Intended status transitions

```
                  save_materials()
                         |
                         v
                     [ draft ] ------ POST /approve/{id} -----> [ approved ]  (terminal)
                         |
                         '----------- (no writer today) -------> [ flagged ]
```

| From | To | Trigger | Guard |
| --- | --- | --- | --- |
| — | `draft` | `save_materials` (`main.py:309`) | none; hard-coded literal |
| `draft` | `approved` | `POST /approve/{item_id}` | **none — must be added** |
| `draft` | `flagged` | **unimplemented** | — |
| `approved` | * | none | terminal; UI renders a `✓ Approved` note, no buttons (`main.py:752`) |
| `flagged` | `approved` | reachable today, and should not be | **must be blocked** |

Unknown status strings are not an error state: `bucket()` (`main.py:713`) lowercases and
falls through to `"draft"` for anything it does not recognise. A typo'd status therefore
appears as an approvable draft.

### 3.3 Contract break: `'flagged'` is unreachable

The UI is built for a status the backend never writes.

- `flag_for_review` (`main.py:320`) is a `print` and a `logger.warning`. It does not
  touch `lessons.db`, and its signature (`item: str, reason: str`) carries no `unit_name`,
  `topic`, `objective`, or `tier` — so it could not insert a well-formed row even if it
  tried.
- `save_materials` (`main.py:309`) writes the `'draft'` literal and has no other branch.
- No route mutates status except `approve_material`, which only ever writes
  `'approved'`.

Therefore `status = 'flagged'` can never appear in the table, and the following are all
dead paths: the flagged section (`main.py:675`), the `count-flagged` stat
(`main.py:657`), `.badge-flagged` / `--amber` styling (`main.py:597`), the `"flagged"`
branch of `bucket()` (`main.py:715`), and the "the agent flagged it before writing
anything" message (`main.py:773`).

When the agent escalates, the teacher sees **nothing at all** in the UI. The information
exists only in the server's stdout.

Closing it requires, at minimum:

1. `flag_for_review` accepts `unit_name`, `topic`, `objective`, `tier` and inserts a row
   with `status='flagged'`, null text fields, and the `reason` persisted.
2. A `reason TEXT` column, surfaced on the flagged card.
3. `approve_material` rejects any row whose current status is not `'draft'` — `409
   Conflict` is the right code, with the current status in the detail.
4. The system prompt (`main.py:359`) already instructs the agent to flag *and stop*; it
   must also pass the lesson identifiers through so step 1 has data to write.

---

## 4. Acceptance Criteria — full-pipeline verification with `batch_test.py`

### 4.0 What `batch_test.py` does and does not prove

`batch_test.py` fires all four `LESSON_QUEUE` lessons concurrently at
`POST /process-lesson` (`batch_test.py:161-166`) and reports each HTTP result.

It asserts **only that the API accepted the work** — `status == 202` (`batch_test.py:115`).
It never polls `/materials`, so it cannot observe a single saved row. `print_next_steps`
(`batch_test.py:150`) hands verification back to the human with a `curl` suggestion.
Two consequences:

- A green `Accepted 4/4` run is compatible with **zero** rows being written. Every
  criterion in 4.2 currently requires manual checking.
- All four objectives in `LESSON_QUEUE` are well-formed, so **the escalation path is
  never exercised** by the batch test.

Criteria below are split accordingly.

### 4.1 Preconditions

Check every item below **on the host that will run Uvicorn**. Import probes taken from
any other machine or sandbox prove nothing about this project: `main.py` is only ever
executed by whatever serves `127.0.0.1:8000`.

- [ ] All third-party imports resolve in the *server's* interpreter:
      `python -c "import fastapi, aiosqlite, uvicorn, pydantic, litellm, strands"`,
      plus `aiohttp` for `batch_test.py`. Every one of these is imported at module
      scope (`main.py:17-24`), so a missing package is an **import-time crash**, not a
      degraded run.
- [ ] **Gap: there is no dependency manifest in the tree.** No `requirements.txt`, no
      `pyproject.toml`, no lockfile — only `main.py`, `batch_test.py`, `opencode.json`
      and `skills-lock.json`. "Dependencies are installed" is therefore unverifiable and
      unreproducible on a second machine. Pin the seven packages above before anyone
      calls this environment reproducible.
- [ ] `GEMINI_API_KEY` is set in the server's environment. Unset, `LiteLLMModel` is still
      constructed successfully at import time (`main.py:369`) — the failure surfaces only
      later, inside the background task, with nothing returned to the client.
- [ ] Server running on `127.0.0.1:8000` — `batch_test.py:14` hardcodes that origin.
- [ ] `GET /health` returns `{"status": "ok"}`.
- [ ] `lessons.db` exists with a `materials` table. Created by `lifespan` (`main.py:450`)
      on first boot; absent from the tree today.

### 4.2 Pipeline criteria (automatable now)

- [ ] All four lessons return `202` within the 15s client timeout
      (`batch_test.py:15`); `batch_test.py` prints `Accepted 4/4`.
- [ ] Each 202 body echoes the submitted `unit_name` and `topic`, and
      `tiers == ["struggling","on-level","advanced"]`.
- [ ] Wall clock for the batch stays well under the sum of the four runs — proves the
      202 returns before generation, not after.
- [ ] After the queue drains, `GET /materials` returns **exactly 12 rows**
      (4 lessons x 3 tiers), matching `batch_test.py:94`'s stated expectation.
- [ ] Every row has non-empty `worksheet_text`, `quiz_text`, `answer_key_text`.
- [ ] For each of the four topics, the three `tier` values are present exactly once.
- [ ] Every row's `status` is `'draft'`.
- [ ] No two rows share the same `id`; concurrent inserts produced no lock error in the
      server log.
- [ ] Server log shows, per lesson: `Agent run started`, three `Generated worksheet`,
      three `Generated quiz + key`, three `Saved materials | id=`, one
      `Agent run finished`. No `Agent run failed`.

### 4.3 Content-contract criteria

- [ ] No worksheet or quiz contains real language syntax — no `;`, `{`, `}`, `def `,
      `import `, `print(`. Pseudo-code keywords only (`main.py:68`).
- [ ] `struggling` worksheets contain an ASCII flowchart, a word bank, and a trace table;
      no blank-page authoring task (`main.py:80`).
- [ ] `advanced` worksheets contain a deliberately flawed snippet plus an efficiency
      comparison (`main.py:93`).
- [ ] Each quiz has five items labelled `Q1`-`Q5` totalling 10 points, and its answer key
      names a misconception per item and ends with the 8/10 mastery threshold
      (`main.py:222-230`).
- [ ] Both texts begin with their required headers: `WORKSHEET — <topic> (<tier> tier)`
      and `QUIZ — <topic> (<tier> tier)`.

### 4.4 Review-desk criteria

- [ ] `GET /` returns `200` and renders 12 draft cards under "Drafts Ready for Approval";
      masthead reads `0 flagged`, `12 drafts`, `0 approved`.
- [ ] "Show full text" expands all three panes; collapse restores the clamped excerpt and
      keyboard focus stays on the toggle.
- [ ] Approving one card moves it to Approved Materials, decrements drafts, increments
      approved, and announces the change on the live region — with no full page reload.
- [ ] Re-issuing `POST /approve/{id}` for that same id returns `200` (idempotent, per
      3.1) and the counts do not drift.
- [ ] `POST /approve/999999` returns `404`; driven through the UI the card shows a scoped
      `.card-error`, the button re-enables, and no other card is affected.
- [ ] With the server stopped, "Refresh from database" surfaces the degraded message and
      keeps the last-loaded records on screen.
- [ ] Inject a worksheet containing `</script><img src=x onerror=alert(1)>` and confirm
      `GET /` renders it as literal text with no script execution — this is the
      `main.py:905` + `esc()` boundary under test.

### 4.5 Escalation criteria (blocked on section 3.3)

Requires a fifth `LESSON_QUEUE` entry with a deliberately unusable objective, e.g.
`{"unit_name": "Unit 4: Loops", "topic": "Loops", "objective": "learn stuff"}`.

- [ ] The request still returns `202` — ambiguity is not a client error.
- [ ] Server log shows `HUMAN REVIEW REQUIRED` with a specific reason, and **no**
      `Saved materials` lines for that lesson.
- [ ] Row count rises by 12, not 15 — the flagged lesson saved nothing.
- [ ] **Blocked:** one row exists with `status='flagged'`, its `reason` populated, and
      null text fields.
- [ ] **Blocked:** it renders in "Requires Review (Flagged)" with the amber accent, the
      reason visible, and no Approve button.
- [ ] **Blocked:** `POST /approve/{flagged_id}` returns `409`, not `200`.

### 4.6 Definition of done for Phase 5

4.1 through 4.4 pass unattended, 4.5 passes with the section 3.3 work landed, and
`batch_test.py` polls `/materials` itself so 4.2 needs no `curl`. Until `batch_test.py`
verifies rows rather than acceptances, "the batch test passed" means only that FastAPI
answered the phone.

---

## Addendum B — Autonomous background missions

This section is the as-built record for the round that transformed IdentAI
from a teacher-initiated single-lesson generator into an autonomous background
agent. Read it alongside the original sections above; everything earlier in this
file still applies.

### B1. The conceptual model

Two distinct objects live in `main.py`:

* `Mission` is one Strands-agent run. It still exists for the legacy
  `/process-lesson` and `/generate` routes. It owns the per-run event trail
  (the mission log the dashboard polls) and the per-tier sub-events.
* `CurriculumMission` is the workflow object the teacher sees on the
  dashboard. It is what the teacher creates with one POST; it owns the
  state machine, the per-tier sub-mission ids, the per-tier outcomes, the
  count of items still needing a teacher's judgement, and the one-line
  summary at the bottom of each dashboard card.

The teacher never re-initiates a generation step. The orchestrator
(`main.run_curriculum_mission`) iterates the tiers, calls the existing per-tier
`run_mission` once per tier, observes the outcome, and updates the curriculum
mission. Within a tier the Strands agent still picks tools; the orchestrator
only owns the per-tier scheduling, the safety rails, the state machine, and
the human-review queue.

### B2. The state machine

`CurriculumMission.state` is one of seven values:

| State | Meaning |
| --- | --- |
| `QUEUED` | accepted, not yet started |
| `RUNNING` | orchestrator is iterating tiers |
| `AUDITING` | the agent is auditing one tier's output |
| `SELF_CORRECTING` | the agent regenerated one tier after a failed audit |
| `WAITING_FOR_HUMAN` | at least one item is flagged; orchestrator kept going through the rest |
| `COMPLETED` | all tiers accounted for, every flagged item approved |
| `FAILED` | unrecoverable: timeout, exception, no provider |

A same-state transition (e.g. `RUNNING -> RUNNING` with a new phase) does not log a
new transition entry. The very first `QUEUED` entry is always logged so the
dashboard can show "started in QUEUED at HH:MM:SS".

### B3. The new routes

* `POST /missions` — fire-and-forget. Returns 202 with the new
  `mission_id` immediately. The orchestrator runs in the background.
* `GET /api/curriculum-missions` — every mission, newest first. Optional
  `?state=active|waiting|completed|failed` filter.
* `GET /api/curriculum-missions/active` — the active dashboard section.
* `GET /api/curriculum-missions/waiting` — the requires-review section.
* `GET /api/curriculum-missions/completed` — the completed section.
* `GET /api/curriculum-missions/{id}` — full snapshot.
* `GET /api/curriculum-missions/{id}/summary` — the one-line teacher-facing
  summary.

The three dashboard sections are polled in parallel with the existing
`/api/materials` poll, at the same 6-second cadence. The mission log itself
keeps its existing shape (`/api/missions/{id}`) so older dashboards keep
working.

### B4. The review interrupt principle

The summary the teacher sees is computed in `CurriculumMission.compute_summary`:

  * All items handled, none waiting: "Mission completed — X/Y handled
    automatically."
  * Some items still waiting: "Mission completed — X/Y handled
    automatically. Z decision(s) require your attention."
  * In progress: "Mission in STATE — X/Y handled, Z waiting on you."

When the teacher approves a flagged item, `approve_material` (existing route)
finds the owning curriculum mission via the `RECORD_TO_CURRICULUM` reverse
index, calls `CurriculumMission.approve_flagged`, and the mission moves to
`COMPLETED` if that was the last flagged item. The mission-completion summary
is recomputed and exposed on the next dashboard poll.

### B5. The orchestrator's role vs the agent's role

What the orchestrator does:

  1. Iterate tiers in `goal.tiers` order.
  2. For each tier, spawn a sub-mission (a `Mission` object) and run it via
     `agent.invoke_async(tier_brief)`.
  3. Read the sub-mission's `saved_ids` / `flagged_ids` and record the
     outcome on the curriculum mission.
  4. Move the state machine.
  5. After all tiers, transition to `WAITING_FOR_HUMAN` (if any items are
     flagged) or `COMPLETED`.

What the agent does (unchanged from the previous round):

  1. Pick tools.
  2. Audit its own output.
  3. Self-correct on mechanical defects (with a finite budget).
  4. Escalate genuine judgement calls.

Python enforces safety rails only (the audit-rail in `save_draft`, the
correction budget, the per-tier timeout, the
CURRICULUM_TIER_TIMEOUT_SECONDS env var). The tool-selection decision lives
entirely in the Strands agent.

### B6. Acceptance-criteria demo

`demo_mission.py` is the acceptance-criteria demo. It uses the FastAPI
TestClient to exercise `POST /missions` and `POST /approve/{id}` against the
real routes, drives the orchestrator with a stubbed `_complete` and a
`FakeAgent` that exercises the real per-tool coroutines, and asserts on the
mission state at every step. The demo fails if any of:

  1. The mission does not reach `WAITING_FOR_HUMAN` after the agent runs.
  2. Any of the 3 tiers fails to produce a row.
  3. The teacher approve does not close the mission.
  4. The classroom-ready output is missing a record.

Run with `python demo_mission.py` from the project root; exits 0 on success,
1 on any failure.

### B7. Tests

`verify.py` grew to 133 checks. New groups:

  * `test_curriculum_mission_state_machine` — drives the state machine
    directly: initial state, transition logging, same-state no-op, unknown
    state rejection, tier-outcome recording, progress computation, summary
    formula, approve-flagged closure.
  * `test_curriculum_mission_routes_wired` — every new route is registered.
  * `test_record_to_curriculum_index` — the reverse index that
    `approve_material` uses to find the owning curriculum mission.

Run `python verify.py` to confirm 133 passing.

### B8. What is intentionally out of scope

  * Persistence of `CurriculumMission` across server restarts. The
    `Mission` class and the materials table are persistent; the curriculum
    mission is in memory, like the per-tier `Mission`. A production
    deployment would add a `curriculum_missions` table; the current design
    only loses the progress view, not the materials.
  * Resume after crash. The orchestrator is not idempotent across crashes;
    in practice the teacher would create a new mission.
  * Tier-level parallelism. Tiers run sequentially. A production deployment
    could spawn them concurrently; the current design is intentionally
    sequential so the dashboard shows clear progress.

---

## Addendum — Pedagogical self-auditing and misconception-driven assessment

This section is the as-built record for the round that delivered the project's
primary differentiator. Read it alongside the original sections above; everything
earlier in this file still applies, and the changes below extend it rather than
replace it.

### A1. Audit schema

The audit tool now returns a structured finding list on top of the human-readable
`problems`, in addition to the existing `verdict`:

    {
      "verdict": "pass" | "revise" | "human" | "unknown",
      "status":   "PASS" | "FAIL",
      "severity": "low" | "medium" | "high",
      "problems": [str, ...],
      "findings": [{"severity", "issue", "affected_component",
                    "recommended_action"}, ...],
      "auto_fixable": bool,
      "checked_layer": "structural" | "critique" | "input",
      "reason": str,
      "corrections_left": int,
      "next_step": str,
    }

Allowed `affected_component` values: `objective`, `worksheet`, `quiz`, `answer_key`,
`distractor`, `code_example`, `explanation`, `difficulty`, `prerequisite`. Unknown
components from the model are coerced to a safe default; the audit never trusts the
model's vocabulary.

Allowed `recommended_action` values: `regenerate_worksheet`, `regenerate_quiz`,
`regenerate_distractors`, `regenerate_objective`, `fix_explanation`,
`escalate_to_teacher`, `no_action`. The agent's corrective choice is driven by these.

### A2. Bounded pseudo-code interpreter

`_run_pseudocode` and the public `execute_pseudocode` cover the subset the curriculum
teaches: `SET ... TO/=`, `INPUT`, `PRINT`, `IF/ELSE IF/ELSE`, `REPEAT n TIMES`,
`WHILE`, and their `END` markers, with a 5000-step cap. Anything outside the subset is
reported as a `syntax` finding rather than a silent skip, and a non-progressing `WHILE`
is reported as `infinite`. The interpreter does not claim to be a general programming
language engine and says so when asked.

`verify_code_example` now also returns the interpreter's `execution` block:
`{ok, output, steps, error, line}`. Static errors and dynamic outputs are both
visible to the agent.

### A3. Misconception linkage

The answer-key audit now requires one named misconception per wrong option. The
parser `main._extract_answer_key_misconceptions` accepts several answer-key
conventions: a single `Q1: B. misconception: ...` line, a per-option list
`Q1: B. ...  A) misconception: ...  C) misconception: ...`, and free-response
items (`Q5: free response. misconception: ...`). The structured per-option
metadata is persisted on the mission's `mission.audits[digest]["distractor_metadata"]`
so the dashboard can render the diagnostic.

`check_misconception_linkage(quiz, answer_key)` is the deterministic guard: any
option without a named misconception, and any answer key that lists fewer
entries than the quiz has options, is reported as a `distractor` finding with
`recommended_action: regenerate_distractors`.

### A4. Mission event vocabulary

Two new safe event kinds have been added (the existing ones are unchanged):

  * `audit_failed` — emitted on every `verdict: revise` with the structured findings
  * `human_review_required` — emitted on every `verdict: human`

`agent_reasoning` is still deliberately absent. The reasoning handler is attached so
strands has somewhere to send model text, but it is a no-op for the live panel.

### A5. Demo scenario

`demo_scenario.py` is a deterministic end-to-end run on the topic "Nested loops"
(on-level tier). It uses the real Strands agent factory, the real ten tools, the
real audit, and the real save path. The first `generate_assessment` call returns a
quiz whose answer key is deliberately missing per-option misconceptions; the
deterministic linkage check catches it; the agent's `next_step` and
`recommended_action` direct the regeneration; the second audit passes; the tier
saves. Every event in the chain comes from the real tool path. Use:

    python demo_scenario.py

There is also a `--live` flag that drops the stub and uses the configured model
provider — slower and may hit provider rate limits.

### A6. Verification

`verify.py` (98 checks) now covers the new categories:

  * misconception metadata extraction and linkage on real answer keys
  * bounded pseudo-code interpreter (REPEAT, WHILE, infinite, prose-only)
  * answer-key-vs-execution consistency
  * pedagogical audit pass/fail on real audit paths with the structured schema
  * self-correction event on audit-revise
  * max-retry protection (`corrections_left` exhaustion, escalation `next_step`)
  * `human_review` event for judgement-call escalation

Run with `python verify.py` from the project root; exits 0 on success, 1 on any
failure. No existing test was weakened to make this work.

