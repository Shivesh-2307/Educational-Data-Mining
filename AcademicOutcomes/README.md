# Matched Academic Outcomes

`matched_session_outcomes.csv` is a derived subset of the study authors' public `main_regressions/final_data.csv`, matched to this project's existing `username` values by exact numeric ID equality (`username` maps to `Student_ID`). It retains only the 518 students with local conversation data and the score/context columns needed here.

Source: [obastani/GenAICanHarmLearning](https://github.com/obastani/GenAICanHarmLearning), commit `2f63dae1a01d51453826fe07ef5cf6678e339588`. Cite Bastani et al. (2025), PNAS, DOI [10.1073/pnas.2422633122](https://doi.org/10.1073/pnas.2422633122).

The file has 1,820 unique student-session rows. `Part2Tot` is the assisted-practice score; `Part3Tot` is the later unassisted-exam score. Both score columns are complete. `gpa_prev` is missing in 36 rows; `Grader` is missing once. The cohort has 27 classes (14 GPT Tutor, 13 GPT Base) and no control-arm students. Treatment mapping is `GPTBase=1` / `Treatment arm=vanilla` / local `treatment=vanilla`; `GPTTutor=1` / `Treatment arm=augmented` / local `treatment=aug`.

The exact outcome join uses (`username`, `session_id`) = (`Student_ID`, `Session`). Of 1,820 score sessions, 1,735 have chat features for that same session; 85 do not. Thirty-six local chat sessions have no outcome row. No IDs are guessed, and the original processed datasets are not replaced or modified.
