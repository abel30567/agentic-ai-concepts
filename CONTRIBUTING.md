# Contributing

## How to contribute

1. Fork, branch, and keep changes focused on one concept or one eval.
2. Standard library only — no new dependencies.
3. Add or update a test in `tests/` for any behaviour change. Run:
   `python -m unittest discover -s tests -v`
4. Run the demo to make sure the happy path still works:
   `python examples/demo.py`

## What makes a good PR

- A new tool in `examples/` with a strict schema and a helpful error
  path.
- A golden eval case in `tests/` built from a real failure you saw.
- A concept module that stays independently importable — no import
  of `agent.py` from the concept modules.
- Docs that explain the *why*, not just the *what*.
