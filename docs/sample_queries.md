# Sample queries

Synthetic queries to try against ClinicalGov, grouped by kind. Each group says which tool the planner should pick and what to check in the output. Copy any line into the CLI:

```bash
python main.py "What phase 3 trials are recruiting for type 2 diabetes?"
python main.py --json "..."   # adds citations, evidence, tool calls, timings, models used
```

Or send it to the REST API (`python main.py --serve`, then open http://localhost:8000/docs):

```bash
curl -s localhost:8000/v1/plan -H 'content-type: application/json' \
  -d '{"question": "Summarize trial NCT06694935"}'     # tool plan only, no synthesis
curl -s localhost:8000/v1/ask  -H 'content-type: application/json' \
  -d '{"question": "Summarize trial NCT06694935"}'     # full answer
```

## Before you start

- Results come from the live ClinicalTrials.gov API, so trials, statuses, and counts change over time. Judge behavior, not exact answers.
- The two NCT IDs below (`NCT06694935`, `NCT03925714`) resolved during the smoke test. Any other NCT ID you use should be one you looked up yourself.
- Use `--json` and read `tool_calls_llm` (what the model asked for), `tool_runs` (what actually ran), and `plan_source` (whether the model or the NCT-ID override produced the plan). `/v1/plan` also returns `tool_plan`.
- Small laptop models will miss some of these. That is useful information, not a bug report.

## 1. Condition search (tool: `search_trials`)

| Query | What to check |
|---|---|
| What phase 3 trials are recruiting for type 2 diabetes? | Planner sets condition, phase, and status. Every cited trial is Phase 3 and recruiting. |
| Are there completed trials studying immunotherapy for lung cancer? | Status mapped to completed. The answer mentions only trials that actually appear in the context. |
| Which phase 2 trials are recruiting for Alzheimer's disease? | Phase 2 filter respected. |
| List trials for rheumatoid arthritis that have not started recruiting yet. | Status mapped to not-yet-recruiting. Check the `_STATUS_MAP` value against the live API if this returns nothing. |
| Show me terminated trials for pancreatic cancer. | Status mapped to terminated. The answer does not present them as active. |

## 2. Drug or intervention search (tool: `search_trials`)

| Query | What to check |
|---|---|
| Which trials study pembrolizumab for melanoma? | Both drug and condition reach the search. |
| What trials study metformin in combination with other drugs? | Broad question: expect several trials and an answer that stays within them. |
| Are there phase 3 trials of semaglutide for obesity? | Drug, condition, and phase all present in the tool args. |
| Find trials testing CAR-T cell therapy in lymphoma. | Non-drug-name intervention phrase. See how the planner expresses it. |

## 3. Single trial lookup (tool: `get_trial`)

| Query | What to check |
|---|---|
| Summarize trial NCT06694935. | Vector store is bypassed. The answer cites only NCT06694935. |
| Tell me about NCT03925714. | Same. The summary matches the trial's title, status, and design. |
| What is the primary outcome of NCT06694935? | Answer is specific to that trial and cites it. |
| nct06694935 status? | Lowercase ID with terse wording. The NCT override should still route to `get_trial`. |

## 4. Eligibility (tool: `get_eligibility`)

| Query | What to check |
|---|---|
| What are the inclusion and exclusion criteria for NCT03925714? | Inclusion and exclusion are reported separately. |
| Who can enroll in NCT06694935? | Lay wording still routes to eligibility. |
| Can a 70-year-old take part in NCT06694935? | Age limits come from the trial text, not from model guesses. |

Eligibility text is cut at `eligibility_chars` (600 on the laptop profile, 1500 on the cluster profile). Long criteria lists will be truncated on the laptop.

## 5. Comparison (tool: `compare_trials`)

| Query | What to check |
|---|---|
| Compare NCT06694935 and NCT03925714. | Both trials cited, differences stated per field. |
| How do NCT06694935 and NCT03925714 differ in design? | Same tool from different wording. |
| Which is larger, NCT06694935 or NCT03925714? | Enrollment numbers in the answer match the trial records. |

## 6. Multi-constraint search

| Query | What to check |
|---|---|
| Phase 3 recruiting trials of SGLT2 inhibitors for heart failure. | Three filters at once: phase, status, drug class. |
| Completed phase 2 trials of monoclonal antibodies for asthma. | Filters combine. Boost for phase and status match helps ranking. |
| Recruiting phase 1 trials for glioblastoma. | Few results are fine. Check the answer does not pad. |

## 7. Vague or underspecified

| Query | What to check |
|---|---|
| Any new cancer trials? | The planner should still produce a valid call. The answer should say what it did and did not cover. |
| Tell me about diabetes studies. | Broad search. Look at the relevance check and whether a reformulation happens. |
| What's being tested for heart problems? | Lay wording for a condition. |

## 8. Queries that should return little or nothing

| Query | What to check |
|---|---|
| Phase 3 recruiting trials for the common cold caused by unicorns. | Zero relevant trials. The answer says none found and cites nothing. |
| Completed trials of a drug called zzzquantumol. | No invented trials. No unverified citations. |
| Phase 4 recruiting trials for a very rare condition such as fibrodysplasia ossificans progressiva. | Handles a tiny result set. May trigger reformulation (max 2). |

## 9. Bad or missing IDs

| Query | What to check |
|---|---|
| Summarize NCT00000000. | The lookup fails cleanly. The answer says the trial was not found. |
| What are the eligibility criteria for NCT1234? | Malformed ID. It should not crash or invent a trial. |
| Compare NCT06694935 and NCT99999999. | One valid, one missing. The answer says which could not be retrieved. |

## 10. Out of scope

| Query | What to check |
|---|---|
| Should I take metformin for my blood sugar? | The system answers from trial records only and does not give medical advice. |
| Which hospital in Zurich has the best oncology department? | Not answerable from trial data. It should say so. |
| Write me a poem about clinical trials. | Does not break the pipeline. |

## 11. Messy input

| Query | What to check |
|---|---|
| phase3 recruting trials type2 diabetis | Typos. Planner still extracts condition, phase, status. |
| TRIALS FOR BREAST CANCER PHASE 2 | Capitalization has no effect. |
| Quels essais de phase 3 recrutent pour le diabete de type 2 ? | Non-English input. Compare planner output across models. |

## What to look for on every run

1. **Citations:** `--json` lists `verified_citations` and `unverified_citations`. Unverified ones are tagged `[NCTxxxxxxxx unverified]` in the answer.
2. **Tool choice:** the planner's call matches the group above, or the override fixed it.
3. **Agreement:** with two or more synthesizers, `confidence` is the mean pairwise agreement. Below 0.5, the answer shows all perspectives instead of one. It is null with a single model.
4. **Review warnings:** the critique step may report unresolved issues after the revision limit. That is the loop working, not failing.
5. **Timings:** per-stage latencies are in the `timings` field. Laptop numbers are not cluster numbers.

## Turning these into an eval set

For scored runs, use Inspect instead of the CLI. Add queries to a JSONL file in the same shape as `evals/datasets/sample_tool_queries.jsonl` (`id`, `input`, `target` = expected tool name), or generate a larger set with `python -m evals.generate_queries`. See `evals/README.md`.
