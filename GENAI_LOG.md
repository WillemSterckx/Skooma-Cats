# GenAI log

Running log for the GenAI Disclosure at the end of the project (see the course GenAI policy: functional scope, prose, 3 to 5 real prompts, verification).

| Date | Who | Tool | Prompt (short) | What the AI did | What we checked / changed |
|---|---|---|---|---|---|
| 2026-10-05 | Aarya | Claude Code | "do a pass over all the code i have made so far in the mushroom project and make sure everything is according to exam standards" | Reviewed notebooks 1, 2 and 2.5 against the assignment rules, re-ran them and re-tested the claims in the markdown | Found 2 wrong numbers in earlier AI-written markdown: "12 tests" in notebook 1 (really 9, so the Bonferroni cut-off is 0.006) and "17 values filled" in notebook 2 (really 18, the counter ran before the stemless rule chained). Found 3 claims without code behind them (no duplicates, missing values not linked to class, dropping duplicates changes nothing) |
| 2026-10-05 | Aarya | Claude Code | "okay fix these issues, for notebook 1 azaam made the first half and i made the second" | Fixed the numbers, added code cells for the 3 claims, added markdown above code cells, cleaned up comments, fixed the class plot colors and the printed list/warning, pinned requirements | Re-ran all notebooks: no errors, `mushroom_clean.csv` byte-identical to before (md5 27a020f0...). The new tests gave the same results as the old claims |

Earlier sessions (2026-10-03, EDA extension and cleaning notebooks with Claude Code) still need to be added here.
