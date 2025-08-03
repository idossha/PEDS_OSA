


LMM: Linear Mixed Model
=======================

### we are asking: Do kids with more spindles show a greater within-subject change i nthe performacence over time?

- Main effect of time: do kids generally show a greater within-subject change in performance over time?
- Main effect of spindle density: do kids with higher spindle density perform better overall?
- Time x spindle density interaction: do kids with more spindle density show a greater within-subject change in performance over time?

hypothesis: kids with higher spindle density show a greater within-subject change in performance over time.
Meaning: sleep related change depends on spindle density.

### Does AHI predict spijndle density? and how is it compared to the impact of age?

- main effect of age: are older kids generally higher spindles density?
- main effect of AHI: are kids with higher AHI generally have lower spindles density?
- interaction of age and AHI: is AHI severity impact natural development of spindles density?

age = age_years + age_months where if age_months is 6 or more, age_years is age_years + 1
AHI = log(NREM_AHI + 1)
