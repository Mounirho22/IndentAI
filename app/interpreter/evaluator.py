"""
Deterministic assessment layer for the autonomous learning agent.

Two strictly separated concerns live in this module:

*Pure expression evaluation* — `interp_eval` is the pseudo-code expression
evaluator the bounded interpreter (engine.py) runs per line. It has no
knowledge of students, events, or humans.

*The silent assessment layer* — everything else. These functions are the
machinery the autonomous background loop calls between a student event and
its outcome. Their contract is that they never talk to a human: they take
artefacts (pseudo-code, claimed outputs, attempt histories), return typed
verdicts, and raise nothing on bad input that a verdict can express.

The autonomy boundary itself is decided in exactly one place:
`decide_autonomy`. The hackathon rule — the agent runs autonomously and only
surfaces when there is a real decision to make — is implemented as three
explicit escalation triggers (stuck-loop, hallucination risk, permission
gate); every other outcome is handled autonomously by the runtime.

All analysis primitives are deterministic and free (no model call), which is
what makes running them in the background for every submission affordable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence


# --------------------------------------------------------------------------- #
# Pure pseudo-code expression evaluation (unchanged public surface)
# --------------------------------------------------------------------------- #


def _top_level_scan(s: str, matches: Any) -> int | None:
    """
    Walk `s` left to right at paren depth 0, returning the index of the
    first character where `matches(index, s)` is true, or None.

    Operators inside parentheses belong to a sub-expression, so they are
    skipped here and handled when the sub-expression is evaluated.
    """
    depth = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and matches(i, s):
            return i
    return None


def interp_eval(expr: str, env: dict[str, Any]) -> Any:
    """
    Evaluate a simple pseudo-code expression against a variable environment.

    Supports:
      * string literals in "..." or '...'
      * integer and float literals
      * TRUE / FALSE
      * variable names (looked up in env)
      * + - * / ( ) with left-to-right evaluation within parens
      * comparisons =, <>, <, >, <=, >=
      * AND, OR, NOT

    Precedence, lowest first: OR, AND, NOT, comparisons, arithmetic — so
    "a < b AND b = 10" compares on both sides of the AND instead of
    splitting on the "=" inside "b = 10".

    Raises:
        KeyError: a variable name is not present in `env` (the interpreter
            maps this to the "undef" error kind).
        ValueError: an empty expression, or a type error while applying an
            arithmetic operator (mapped to the "type" error kind).
        ZeroDivisionError: division by zero (mapped to "divzero").
    """
    s = expr.strip()
    if not s:
        raise ValueError("empty expression")

    # Strip a trailing '.' some pseudo-code style uses.
    if s.endswith(".") and len(s) > 1:
        s = s[:-1].rstrip()

    upper = s.upper()
    if upper == "TRUE":
        return True
    if upper == "FALSE":
        return False

    # String literal.
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]

    # Pure integer / float literal.
    try:
        if any(ch in s for ch in ".eE") and any(ch.isdigit() for ch in s):
            return float(s)
        return int(s)
    except ValueError:
        pass

    # AND / OR first: they bind loosest. OR is split before AND, and the
    # right-hand side recurses, so "a OR b AND c" reads as "a OR (b AND c)".
    for connector in (" OR ", " AND "):
        idx = _top_level_scan(
            s, lambda i, text, _c=connector: text.upper().startswith(_c, i)
        )
        if idx is not None:
            left = interp_eval(s[:idx], env)
            right = interp_eval(s[idx + len(connector):], env)
            if connector == " OR ":
                return bool(left) or bool(right)
            return bool(left) and bool(right)

    # NOT binds tighter than AND / OR but looser than comparisons.
    if upper.startswith("NOT "):
        return not interp_eval(s[4:].strip(), env)

    # Comparisons, longest operator first so ">=" is never read as "=".
    # Two-character forms must be tested before their one-character
    # prefixes at the same position.
    i = _top_level_scan(
        s,
        lambda j, text: any(text.startswith(op, j) for op in COMPARISON_OPS),
    )
    if i is not None:
        op = next(op for op in COMPARISON_OPS if s.startswith(op, i))
        left = interp_eval(s[:i], env)
        right = interp_eval(s[i + len(op):], env)
        if op == "=":
            return left == right
        if op == "<>":
            return left != right
        if op == "<=":
            return left <= right
        if op == ">=":
            return left >= right
        if op == "<":
            return left < right
        return left > right

    # Arithmetic: + - * /. Walk left to right with paren grouping.
    for op in ("+", "-", "*", "/"):
        idx = _top_level_scan(
            s, lambda j, text, _op=op: text[j] == _op and not (_op == "-" and j == 0)
        )
        if idx is not None:
            left = interp_eval(s[:idx], env)
            right = interp_eval(s[idx + 1:], env)
            try:
                if op == "+":
                    return left + right
                if op == "-":
                    return left - right
                if op == "*":
                    return left * right
                if right == 0:
                    raise ZeroDivisionError("divzero")
                return left / right
            except TypeError as exc:
                raise ValueError(f"type:{exc}") from exc

    # Parenthesised single expression.
    if s.startswith("(") and s.endswith(")"):
        return interp_eval(s[1:-1], env)

    # Variable lookup.
    if s in env:
        return env[s]
    raise KeyError(s)


COMPARISON_OPS = ("<>", "<=", ">=", "=", "<", ">")


# --------------------------------------------------------------------------- #
# Silent assessment layer — typed verdicts only, never human-facing I/O
# --------------------------------------------------------------------------- #

FindingSeverity = Literal["error", "warning"]
EscalationCategory = Literal["stuck_loop", "hallucination_risk", "permission_gate"]
Severity = Literal["low", "medium", "high"]

#: A student is considered stuck only after MORE than this many consecutive
#: failed attempts (the hackathon brief: "repeatedly stuck on logic errors
#: > 3 attempts", i.e. the gate opens on the 4th consecutive failure).
STUCK_ATTEMPT_THRESHOLD = 3

#: Constructs the course has not taught yet. Their appearance is a hard
#: error for the taught subset, not a style warning — the bounded
#: interpreter cannot execute them either.
UNTAUGHT_CONSTRUCTS: dict[str, str] = {
    "FOR": "FOR loops are not taught yet; use REPEAT n TIMES or WHILE.",
    "FUNCTION": "FUNCTION definitions are not taught yet.",
    "RETURN": "RETURN statements are not taught yet.",
    "DEF": "function definitions are not taught yet.",
    "IMPORT": "module imports are not part of the pseudo-code subset.",
    "CLASS": "classes are not part of the pseudo-code subset.",
    "ARRAY": "arrays are not part of the taught subset yet.",
}

#: Closest first so a two-character opener inside a longer word is matched
#: deterministically. Mirrors engine.PSEUDO_KEYWORDS (single source of
#: truth stays in engine.py; this tuple only guards keyword *detection*
#: before the lazy import below resolves).
_OPENER_PROBES: tuple[str, ...] = (
    "ELSE IF", "END REPEAT", "END WHILE", "END IF", "ENDIF",
    "SET", "INPUT", "PRINT", "DISPLAY", "IF", "ELSE",
    "REPEAT", "WHILE", "STOP", "END",
)


@dataclass(frozen=True)
class SyntaxFinding:
    """One deterministic static-analysis finding for a pseudo-code snippet."""

    severity: FindingSeverity
    code: str
    message: str
    line: int | None = None


@dataclass(frozen=True)
class SyntaxReport:
    """Result of static syntax analysis over one submission."""

    ok: bool
    findings: tuple[SyntaxFinding, ...] = ()
    code_lines_found: int = 0

    @property
    def errors(self) -> tuple[SyntaxFinding, ...]:
        return tuple(f for f in self.findings if f.severity == "error")

    @property
    def warnings(self) -> tuple[SyntaxFinding, ...]:
        return tuple(f for f in self.findings if f.severity == "warning")


@dataclass(frozen=True)
class UnitCase:
    """One unit test: variable bindings plus the output lines we expect."""

    name: str
    inputs: dict[str, Any] = field(default_factory=dict)
    expected_output: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", dict(self.inputs))
        object.__setattr__(self, "expected_output", tuple(self.expected_output))


@dataclass(frozen=True)
class CaseResult:
    """Outcome of running one `UnitCase` through the bounded interpreter."""

    case: str
    passed: bool
    actual_output: tuple[str, ...] = ()
    error: tuple[str, str] | None = None  # (kind, message) as used by the interpreter
    steps_used: int = 0


@dataclass(frozen=True)
class TestRunResult:
    """Aggregate outcome of a unit-test run over one submission."""

    ok: bool
    passed: int
    failed: int
    cases: tuple[CaseResult, ...] = ()
    total_steps: int = 0

    @property
    def first_failure(self) -> CaseResult | None:
        return next((c for c in self.cases if not c.passed), None)


@dataclass(frozen=True)
class ClaimAssessment:
    """
    Compares an output somebody claims the code produces with what the
    interpreter actually produced.

    A mismatch is the hallucination-risk signal: either a student pasted
    output from another source, or an upstream generated "expected" answer
    disagrees with deterministic execution. Either way the disagreement is
    a judgement call, not effort — the escalation boundary.
    """

    hallucination_risk: bool
    claimed: tuple[str, ...] = ()
    actual: tuple[str, ...] = ()
    detail: str = ""


@dataclass(frozen=True)
class MilestoneProfile:
    """Everything the milestone evaluator needs to judge progress."""

    student_id: str
    milestone_id: str
    consecutive_failures: int = 0
    total_attempts: int = 0
    requires_permission: bool = False
    permission_approved: bool = False

    def __post_init__(self) -> None:
        if self.consecutive_failures < 0 or self.total_attempts < 0:
            raise ValueError("failure/attempt counters cannot be negative")


@dataclass(frozen=True)
class MilestoneVerdict:
    """Where the student stands on a milestone after the latest attempt."""

    state: Literal["on_track", "struggling", "stuck", "advance_gate", "completed"]
    escalate: bool
    reason: str


@dataclass(frozen=True)
class AutonomyDecision:
    """
    The single authority on autonomous-versus-escalate for one processed
    submission.

    mode == "autonomous": the runtime commits the state transition, writes
    telemetry, and never disturbs a human. mode == "escalate": the runtime
    pauses the affected work and opens a decision for a human.
    """

    mode: Literal["autonomous", "escalate"]
    outcome: Literal["auto_passed", "auto_failed", "escalated"]
    category: EscalationCategory | None = None
    severity: Severity = "low"
    reason: str = ""


def _opener_of(line: str) -> str | None:
    """The taught pseudo-code keyword a line opens with, or None."""
    upper = line.strip().upper()
    for probe in _OPENER_PROBES:
        if upper == probe or upper.startswith(probe + " "):
            return probe
    return None


def analyze_syntax(source: str) -> SyntaxReport:
    """
    Statically analyse student pseudo-code without executing it.

    Deterministic checks, cheapest first:
      * lines that open with an untaught real-language construct (FOR,
        FUNCTION, RETURN, ...) — hard errors for the taught subset;
      * REPEAT blocks whose header carries no "n TIMES" count — the
        interpreter refuses these at run time, so they are flagged up front;
      * block balance of IF/END IF, REPEAT/END REPEAT, WHILE/END WHILE,
        including stray closers and unclosed openers.

    This never runs the code and never talks to a human: dynamic problems
    (undefined variables, division by zero, non-termination) are the unit
    test run's job, not this function's.
    """
    # Lazy import: engine.py imports this module for interp_eval, so a
    # module-level import back into engine would close a cycle. The keyword
    # set and executor are resolved at call time — both are pure.
    from .engine import PSEUDO_KEYWORDS, code_lines  # noqa: PLC0415

    # Untaught constructs are scanned over EVERY non-empty line first: real
    # language leaking into a submission ("FOR i = 1 TO 3") does not open
    # with a taught keyword, so the taught-keyword filter used to extract
    # code lines below would silently hide it.
    all_lines: list[tuple[int, str]] = [
        (number, raw.strip())
        for number, raw in enumerate(source.splitlines(), start=1)
        if raw.strip()
    ]
    findings: list[SyntaxFinding] = []
    for number, raw in all_lines:
        upper = raw.upper()
        for construct, hint in UNTAUGHT_CONSTRUCTS.items():
            if upper == construct or upper.startswith(construct + " ") or upper.startswith(construct + "("):
                findings.append(SyntaxFinding(
                    "error", "untaught_construct",
                    f"{construct}: {hint}", line=number,
                ))

    lines = code_lines(source)
    if not lines:
        if findings:
            return SyntaxReport(ok=False, findings=tuple(findings), code_lines_found=0)
        return SyntaxReport(
            ok=False,
            findings=(SyntaxFinding(
                "error", "empty",
                "No pseudo-code found — every line must open with a taught keyword.",
            ),),
            code_lines_found=0,
        )

    # Block balance with nesting. IF chains close with END IF; ELSE /
    # ELSE IF belong to the open chain and do not change depth.
    balances: dict[str, tuple[int, int | None]] = {}  # kind -> (depth, first opener line)
    for keyword in ("IF", "REPEAT", "WHILE"):
        balances[keyword] = (0, None)

    closers = {"IF": {"END IF", "ENDIF"}, "REPEAT": {"END REPEAT"}, "WHILE": {"END WHILE"}}

    for number, raw in lines:
        stripped = raw.rstrip()
        upper = stripped.upper()
        opener = _opener_of(stripped)

        if opener is None:
            findings.append(SyntaxFinding(
                "error", "unknown_keyword",
                f"Line does not open with a taught keyword "
                f"({' '.join(PSEUDO_KEYWORDS)}).", line=number,
            ))
            continue

        if opener in balances:
            depth, _ = balances[opener]
            balances[opener] = (depth + 1, number)
        elif opener == "ELSE" or opener == "ELSE IF":
            if_depth, if_line = balances["IF"]
            if if_depth == 0:
                findings.append(SyntaxFinding(
                    "error", "stray_else",
                    "ELSE / ELSE IF without a matching open IF block.", line=number,
                ))
        elif opener in {"END IF", "ENDIF"}:
            depth, _ = balances["IF"]
            if depth == 0:
                findings.append(SyntaxFinding(
                    "error", "stray_end", "END IF without a matching IF.", line=number,
                ))
            else:
                balances["IF"] = (depth - 1, None)
        elif opener == "END REPEAT":
            depth, _ = balances["REPEAT"]
            if depth == 0:
                findings.append(SyntaxFinding(
                    "error", "stray_end", "END REPEAT without a matching REPEAT.", line=number,
                ))
            else:
                balances["REPEAT"] = (depth - 1, None)
        elif opener == "END WHILE":
            depth, _ = balances["WHILE"]
            if depth == 0:
                findings.append(SyntaxFinding(
                    "error", "stray_end", "END WHILE without a matching WHILE.", line=number,
                ))
            else:
                balances["WHILE"] = (depth - 1, None)
        elif opener == "REPEAT":
            tail = stripped[7:].strip().upper()
            if not tail.endswith(" TIMES") or len(tail) <= len(" TIMES"):
                findings.append(SyntaxFinding(
                    "error", "repeat_no_count",
                    'REPEAT needs a countable count, e.g. REPEAT 3 TIMES.', line=number,
                ))

    for kind, (depth, opener_line) in balances.items():
        if depth > 0 and opener_line is not None:
            findings.append(SyntaxFinding(
                "error", "unclosed_block",
                f"Unclosed {kind} block starting at line {opener_line}.",
                line=opener_line,
            ))

    return SyntaxReport(
        ok=not any(f.severity == "error" for f in findings),
        findings=tuple(findings),
        code_lines_found=len(lines),
    )


def run_unit_tests(
    source: str,
    cases: Sequence[UnitCase],
    *,
    step_cap: int | None = None,
) -> TestRunResult:
    """
    Execute `source` once per `UnitCase` in the bounded interpreter and
    compare captured output with the expected lines.

    Comparison reuses the interpreter's own output normalisation, so case
    and whitespace differences never fail a student. A case with no
    expected output still must run clean (ok, no interpreter error) to
    pass. The step cap bounds every run, so a student's non-terminating
    loop becomes a normal failed case, never a hung background worker.

    Raises nothing for bad student code — failures are CaseResults. A
    misconfigured *test* (cases=None) raises TypeError via the Sequence
    contract like any other API misuse.
    """
    # Lazy import keeps engine -> evaluator -> engine from closing at import
    # time; see analyze_syntax.
    from .engine import INTERPRETER_STEP_CAP, execute_pseudocode, outputs_match  # noqa: PLC0415

    cap = step_cap if step_cap is not None else INTERPRETER_STEP_CAP
    results: list[CaseResult] = []
    for case in cases:
        execution = execute_pseudocode(source, inputs=dict(case.inputs), step_cap=cap)
        passed = bool(execution.ok) and outputs_match(execution.output, list(case.expected_output))
        results.append(CaseResult(
            case=case.name,
            passed=passed,
            actual_output=tuple(execution.output),
            error=execution.error,
            steps_used=execution.steps,
        ))

    passed_count = sum(1 for r in results if r.passed)
    return TestRunResult(
        ok=bool(results) and passed_count == len(results),
        passed=passed_count,
        failed=len(results) - passed_count,
        cases=tuple(results),
        total_steps=sum(r.steps_used for r in results),
    )


def _normalise_claim(text: str) -> tuple[str, ...]:
    """Split a claimed-output blob into normalised lines."""
    from .engine import normalise_output_line  # noqa: PLC0415

    return tuple(
        normalise_output_line(line)
        for line in text.splitlines()
        if line.strip()
    )


def assess_claim(claimed_output: str | None, actual_output: Sequence[str]) -> ClaimAssessment:
    """
    Compare a claimed output against the interpreter's actual output.

    This is the hallucination tripwire: an "expected output" that a model,
    a worksheet, or a student asserted, checked against deterministic
    execution. No claim at all is the common (safe) case.
    """
    if claimed_output is None or not claimed_output.strip():
        return ClaimAssessment(hallucination_risk=False, claimed=(), actual=tuple(actual_output))

    from .engine import normalise_output_line  # noqa: PLC0415

    claimed = _normalise_claim(claimed_output)
    actual = tuple(
        line for line in (normalise_output_line(l) for l in actual_output) if line
    )
    if claimed == actual:
        return ClaimAssessment(hallucination_risk=False, claimed=claimed, actual=actual)

    detail = "claim disagrees with deterministic execution"
    if len(claimed) != len(actual):
        detail += f" ({len(claimed)} claimed line(s) vs {len(actual)} produced)"
    else:
        for index, (c, a) in enumerate(zip(claimed, actual), start=1):
            if c != a:
                detail += f" (first divergence at output line {index})"
                break
    return ClaimAssessment(hallucination_risk=True, claimed=claimed, actual=actual, detail=detail)


def evaluate_milestone(profile: MilestoneProfile) -> MilestoneVerdict:
    """
    Judge a student's standing on a milestone from attempt counters alone.

    Ordered rules — first match wins, because severity is not additive:
      1. The milestone fronts an advanced module that is locked and no
         human has approved the unlock -> "advance_gate". Progress is
         normal (that is exactly why the student is at the gate), so the
         only missing ingredient is a human permission — pure escalation.
      2. More than STUCK_ATTEMPT_THRESHOLD consecutive failures -> "stuck".
         More retries are effort, not progress; a mentor decision is.
      3. Any consecutive failures -> "struggling". Autonomously handled:
         the agent keeps supporting silently and keeps counting.
      4. Otherwise -> "on_track" (or "completed" when nothing is pending).

    Pure function: same profile in, same verdict out, no I/O anywhere.
    """
    if profile.requires_permission and not profile.permission_approved:
        return MilestoneVerdict(
            "advance_gate", True,
            "Milestone is an advanced module that stays locked until a "
            "mentor approves the unlock.",
        )
    if profile.consecutive_failures > STUCK_ATTEMPT_THRESHOLD:
        return MilestoneVerdict(
            "stuck", True,
            f"{profile.consecutive_failures} consecutive failed attempts "
            f"(gate opens above {STUCK_ATTEMPT_THRESHOLD}).",
        )
    if profile.consecutive_failures > 0:
        return MilestoneVerdict(
            "struggling", False,
            f"{profile.consecutive_failures} consecutive failure(s) — inside "
            f"the autonomous support band.",
        )
    return MilestoneVerdict("on_track", False, "Latest attempt passed.")


def decide_autonomy(
    *,
    syntax: SyntaxReport,
    tests: TestRunResult | None,
    claim: ClaimAssessment,
    milestone: MilestoneVerdict,
) -> AutonomyDecision:
    """
    THE autonomy boundary. Maps four deterministic verdicts onto exactly one
    decision: handle silently, or pause and surface to a human.

    Escalation triggers, in precedence order (a higher-priority trigger
    names the brief the human sees, lower ones survive as evidence):
      1. hallucination_risk — a claimed output disagrees with deterministic
         execution. Trust in the material is at stake, which is a judgement
         call, so it outranks the progress signals.
      2. stuck_loop — the student is past the consecutive-failure gate.
      3. permission_gate — the milestone verdict demands a human unlock.

    Everything else is autonomous by rule: pass -> commit "auto_passed";
    any syntax or unit-test failure below the stuck gate -> commit
    "auto_failed", log telemetry, adjust silently, never interrupt.
    """
    if claim.hallucination_risk:
        return AutonomyDecision(
            mode="escalate", outcome="escalated",
            category="hallucination_risk", severity="high",
            reason=claim.detail or "Claimed output disagrees with interpreter output.",
        )
    if milestone.escalate and milestone.state == "stuck":
        return AutonomyDecision(
            mode="escalate", outcome="escalated",
            category="stuck_loop", severity="high",
            reason=milestone.reason,
        )
    if milestone.escalate and milestone.state == "advance_gate":
        return AutonomyDecision(
            mode="escalate", outcome="escalated",
            category="permission_gate", severity="medium",
            reason=milestone.reason,
        )

    work_ok = syntax.ok and tests is not None and tests.ok
    if work_ok:
        return AutonomyDecision(
            mode="autonomous", outcome="auto_passed",
            reason="Syntax clean, all unit tests passed, no claim conflicts.",
        )
    return AutonomyDecision(
        mode="autonomous", outcome="auto_failed",
        reason=(
            f"Static/dynamic checks failed autonomously: "
            f"{len(syntax.errors)} syntax error(s), "
            f"{(tests.failed if tests else 0)} failed unit case(s)."
        ),
    )
