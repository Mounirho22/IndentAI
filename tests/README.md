# tests

- `unit/` - self-contained unit tests.
  - `test_interpreter.py` - the bounded pseudo-code interpreter.
- The primary end-to-end verification suite lives at the repo root as
  `verify.py` (230 checks, run via `python verify.py`). It is kept at the root
  because `python verify.py` is part of the project's workflow.
- `integration/` and `fixtures/` are reserved for future files.

> Note: `tests/unit/test_interpreter.py` currently has a couple of
> pre-existing failures that reflect long-standing gaps between the interpreter
> and its own unit test (`interp_eval` for numeric `<`/`>=` and trailing-space
> whitespace normalisation). These are unrelated to the repository
> re-organization and are tracked as structural debt. `verify.py` is the
> authoritative behaviour contract and passes.
