# LakeRCM Prompt Eval & Tuning

Trace-driven prompt versioning loop. The MLflow Experiment + Prompt Registry UI
*is* the dashboard — bookmark the links below.

## The loop

```
production traces → curated eval dataset → versioned eval → promotion gate → champion alias
                                                       ↑
                                                 GEPA optimizer (optional)
```

Every agent turn is tagged with the prompt name + version + alias_used (champion
or candidate), so any of these surfaces can slice by prompt version.

## CLI cheat-sheet

```bash
# One-time, on first deploy
python -m eval.dataset init

# Weekly (also runs as the lakercm-eval-curation job)
python -m eval.dataset curate --days 7
python -m eval.dataset stats

# Score a specific prompt version against the dataset
python -m eval.run_eval --prompt-version 7

# Compare two versions head-to-head (exits non-zero if gate fails)
python -m eval.compare --baseline-version 6 --candidate-version 7

# Promote candidate to champion (or block, with reason)
python -m eval.promote --candidate-version 7
python -m eval.promote --candidate-version 7 --dry-run

# GEPA: produce a candidate from the current champion
python -m eval.optimize
```

## Bookmarks

Workspace: `https://<your-workspace-host>`

- **Production experiment** (live traces, scorer pass rates):
  `/mlflow/experiments/?experimentName=/Shared/lakercm/agent-traces`
- **Eval experiment** (GEPA + holdout/gate rollouts):
  `/mlflow/experiments/?experimentName=/Shared/lakercm/agent-eval`
- **Prompt Registry**: `<your-catalog>.lakercm.agent_system_prompt`
- **Eval dataset**: `<your-catalog>.lakercm.agent_eval_v1`

### Useful trace filters (paste into MLflow Trace UI)

- All champion turns: `tag.prompt_alias_used = 'champion'`
- All candidate turns: `tag.prompt_alias_used = 'candidate'`
- Specific version: `prompt = '<your-catalog>.lakercm.agent_system_prompt/7'`
- Failed Safety: `feedback.safety = 'no'`
- Thumbs-down: `feedback.user_satisfaction = 'no'`

## Promotion gate

Codified in `eval/composite.py`:

| Check | Rule | Why |
|---|---|---|
| Safety hard gate | `candidate.safety ≥ champion.safety` | Medical-document context — no Safety regression |
| Correctness floor | `candidate.correctness ≥ champion.correctness − 0.01` | Allow ≤1pp regression iff overall improves |
| Composite delta | `candidate.composite ≥ champion.composite + 0.02` | Avoid promoting noise |

Composite weights: `correctness 0.40 / retrieval_groundedness 0.20 / safety 0.20 / relevance_to_query 0.10 / tool_call_correctness 0.10`.

## Champion / candidate aliases

- `agent_system_prompt@champion` — what production serves.
- `agent_system_prompt@candidate` — staged version under evaluation. Optional.
- `LAKERCM_CANDIDATE_TRAFFIC_PCT=0` (default) — manual ramp; bump to 5/10/50 to A/B against real traffic.

The agent app caches both prompts at boot and routes per-request via a
deterministic hash of `user_email + thread_id`. The trace tag
`prompt_alias_used` records which way each turn went.

## Files

| File | Purpose |
|---|---|
| `scorer_set.py` | Shared scorers (production scheduler + offline evaluator) |
| `composite.py` | Composite score + promotion gate logic |
| `dataset.py` | UC-managed eval dataset; trace curation + smoke-set seeding |
| `run_eval.py` | `mlflow.genai.evaluate(predict_fn=...)` for a pinned prompt version |
| `compare.py` | Head-to-head delta + gate verdict |
| `promote.py` | Re-runs gate, swaps `champion` alias on pass |
| `optimize.py` | GEPA optimizer — train/holdout split, candidate alias |
| `scorers.py` | Production scheduled scorers (registers from `scorer_set.py`) |
