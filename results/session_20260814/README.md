# Session 2026-08-13/14 — partial results, runs paused

Both GPU runs were **paused, not completed**. Files here are the state at the
pause point, pulled off the shared persistent filesystem. Instance A's
`/lambda/nfs` mount hung at ~03:55 UTC, so A's file was read and copied via
instance B (same shared filesystem) rather than from A itself.

## `bestofn_n3_14b_partial.json` — elicitation control (D-47), INCOMPLETE

`SStarPolyglotBestOfN`, N=3, `Qwen2.5-Coder-14B-Instruct-AWQ`,
`toolcheck_agent_src` (`tool_choice="required"`), 16k context.

**42 of 59 tasks. 2 resolved (oracle-blind), 3 pass@3.**

The gap between those two is the point of the arm: one task had a passing
candidate that consensus selection did not pick, i.e. measured selection
headroom. Do not quote 2/42 as a rate comparable to the 4/59 arms — the task
sets differ, and the run stopped mid-list rather than sampling randomly.

## `evo1_seed_metadata.json` — evolutionary run seed, COMPLETE

The seed agent's own evaluation over all 59 tasks, 32k context (needed for
HGM's diagnosis prompt; see D-48). This one is a finished, usable result.

**59 submitted · 4 resolved · 40 unresolved · 1 empty patch.**
The remaining 14 are `error`/wrong-artifact rows (D-46), counted as
unresolved, so the seed scores **4/59**.

Resolved: `go__counter`, `java__mazy-mice`, `java__react`, `rust__gigasecond`.

Compare the `14B × required` S0 arm (16k), which also scored 4/59 but on
`go__counter`, `java__bowling`, `java__mazy-mice`, `rust__gigasecond`. Same
aggregate, three of four the same tasks, one swap — evidence that the
aggregate is reproducible while its composition is partly resampling noise.
This also shows the 16k/32k deployment difference did not move the aggregate,
which weakens the comparability caveat recorded in D-48.

## `evo1_proxy_summary.json` / `evo1_proxy_events.json` — interception log

9.02 hours, 12,804 model calls (12,663 ok, 141 failed = 1.1%), 202.9M prompt
tokens. `hgm_exit_code: -15` is the SIGTERM from the intentional pause.

**Known discrepancy, unresolved:** the summary reports
`calls_with_tool_calls: 12645` but `trace_op_counts.tool_call: 28`. These are
computed by two different code paths — a direct check of
`response.choices[0].message.tool_calls` versus
`fork_bridge.reconstruct_trace_from_events`. At least one is wrong, and the
tagging path is the one the paper would rely on. **Do not use
`trace_op_counts` from this run until this is diagnosed.** The raw events are
preserved here precisely so it can be re-derived.

## What was lost

- Instance A: any tasks completing after 03:55 UTC, when its mount hung.
- Instance B: one in-flight `sample_child` (started 05:26). **No children were
  archived** — `output_evo1/hgm_metadata.jsonl` was never written, so the
  evolutionary run produced no scaffold variants. The seed evaluation is the
  only usable output from B.

## Resuming

- **A**: rerun the same command; it skips instance IDs already in the results
  file. Losslessly resumable.
- **B**: requires `--continue_from`, plus deleting `<output_dir>/initial`
  first so `cp -r` recreates it rather than nesting inside it (D-48 defect 4).
  Not yet verified in practice.
- Both need re-provisioning if the instances are terminated: model weights and
  Docker images live on local disk and do not survive; `hgm_B`, `hgm_venv`,
  and these outputs are on the persistent filesystem and do.
