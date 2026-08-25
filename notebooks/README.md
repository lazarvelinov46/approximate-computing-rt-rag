# Notebooks
- `00_ac_smoke_test` — Phase 0: check GPU, load small model, embed 3 passages, one retrieval+generate.
- `01_ac_baseline` — Phase 1: precise baseline end-to-end, short regime. Also derives `generation.batch_size` (real-prompt batch ladder) and selects the short-regime system prompt on a dev sample.
- `02_ac_explain_prompt_selection` — Phase 1: explain-regime prompt selection over four variants on a 200-question disjoint dev sample.
- `03_ac_explain_baseline` — Phase 1: precise baseline end-to-end, explain regime.

(Notebooks are added as each phase begins; develop logic in `src/`, call it from thin notebooks.)