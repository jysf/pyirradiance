Read AGENTS.md in this repo, then do gaps 1-5 in order. Keep it small and fast:
- Build with maturin, run `tests/smoke.py` on the Canon DNG after each change.
- Gap 1 first: write a failing hostile test (forged huge width/height must raise
  ValueError, not allocate/abort), then fix it.
- Don't touch the irradiance repo. Commit after each gap. Report what passed, what is
  unverified, and anything in irradiance's API that blocked you (e.g. a pre-validation
  size helper) as a note, not a change.
Stop after gap 5; skip the README/publish work.
