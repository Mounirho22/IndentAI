"""IdentAI application package.

The canonical runtime lives in :mod:`app.main`. The repository-root ``main.py``
is a thin compatibility shim that re-exports ``app.main`` so legacy entry points
(``uvicorn main:app``, ``python main.py``, ``agentcore/entrypoint.py``) keep
working unchanged, while ``verify.py`` and the demos import the canonical
module directly so their in-process monkeypatching of module globals works.
"""