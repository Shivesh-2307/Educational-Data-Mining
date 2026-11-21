# Essential Research Guide

## What the project asks

Do students' actual prompts during AI-supported math practice help predict their later **unassisted exam score**? Do the prompt and reply patterns differ between GPT Base and GPT Tutor?

`vanilla` means GPT Base. `aug` means GPT Tutor.

## What happens

1. Start with the existing 28,664 cleaned student prompts and 518-student feature table. The original preprocessing is preserved.
2. Measure visible wording clues, such as requests for explanations or hints, attempts, checking, prompt length, and math terms. Keep the original messages and extract these clues before grouping them.
3. Summarize at message, student, session, problem, and conversation levels. Actual assistant turns come from the separate full-conversation export.
4. Join score data using exact student and session IDs. All 518 students match; 1,735 of 1,820 score sessions have same-session chat.
5. Compare text patterns and interaction profiles, then test score prediction while holding out whole classrooms.

The notebooks follow those steps: [01 text data](../notebooks/01_text_features_and_data.ipynb), [02 profiles and treatment](../notebooks/02_interaction_profiles_and_treatment.ipynb), and [03 learning outcomes](../notebooks/03_learning_outcomes.ipynb).

## What the current results say

For the later unassisted exam (`Part3Tot`), adding the new text cues did not improve Ridge prediction: RMSE changed from **0.21970** with context alone to **0.22009** with new text cues. The estimated improvement interval spans zero (**?0.00252 to +0.00183**). In this sample, these hand-built text clues did not add reliable predictive value beyond basic context.

The adjusted GPT Tutor minus GPT Base estimate was **+0.022** for the later exam (95% interval **?0.017 to +0.061**, *p* = .256). For assisted practice, it was **+0.226** (95% interval **+0.170 to +0.282**). Practice scores were earned with AI available, so that result is not proof of lasting learning.

## What it does and does not establish

This is a local, score-linked reanalysis. It adds an exact-session connection between the local conversation features and published scores, tests whether the new cues help predict scores, and reports GPT Tutor/Base comparisons for this matched sample. It does not prove that a prompt style causes learning or directly measure understanding. The matched local transcript cohort has no no-AI control group. The source paper contains the broader three-arm experiment.

## Run from the repository root

```powershell
./.venv/Scripts/python.exe Research_Analysis/run_research_analysis.py
./.venv/Scripts/python.exe Research_Analysis/run_outcome_analysis.py
```

All generated files live under `outputs/`. See [the output catalog](OUTPUT_CATALOG.md) and [the project audit](PROJECT_AUDIT.md).
