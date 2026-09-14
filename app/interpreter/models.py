from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping


@dataclass(frozen=True)
class ExecutionResult(Mapping[str, Any]):
    """
    Immutable execution outcome of a pseudo-code snippet.

    Supports attribute access (e.g. `result.ok`, `result.output`) and dictionary
    subscription (e.g. `result["ok"]`, `result["output"]`) for 100% backwards compatibility.
    """
    ok: bool
    output: list[str] = field(default_factory=list)
    steps: int = 0
    error: tuple[str, str] | None = None
    line: int | None = None

    def __getitem__(self, key: str) -> Any:
        if key == "ok":
            return self.ok
        if key == "output":
            return self.output
        if key == "steps":
            return self.steps
        if key == "error":
            return self.error
        if key == "line":
            return self.line
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        yield from ("ok", "output", "steps", "error", "line")

    def __len__(self) -> int:
        return 5

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "output": list(self.output),
            "steps": self.steps,
            "error": self.error,
            "line": self.line,
        }


@dataclass
class _IfFrame:
    start_pc: int
    condition_met: bool


@dataclass
class _RepeatFrame:
    start_pc: int
    target: int
    done: int = 0


@dataclass
class _WhileFrame:
    start_pc: int


_StackFrame = _IfFrame | _RepeatFrame | _WhileFrame

