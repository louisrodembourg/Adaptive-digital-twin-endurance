# Contribution conventions

## What this project does

A digital twin for endurance training, built on a public dataset
(GoldenCheetah OpenData). It ingests sessions, derives a training load and a
performance measure, fits a fitness-fatigue model, then **evaluates honestly
what that model predicts, for whom, and with what certainty**.

The measured answer: load predicts performance **once it is matched to the
discipline being measured** — bike load against bike power beats "predict the
mean" by +0.339 [+0.055 · +0.524] on the reference athlete, and the effect
replicates across 61 athletes (sign test p = 0.020). It is heterogeneous:
detected in 23, inconclusive in 33, harmful in 5, with a between-athlete
dispersion established at 0.205 [+0.131 · +0.278]. Persistence remains
unbeaten, and the repository says how little power that comparison has. The
planner is built on this result: it measures the verdict for the athlete at
hand and never promises more than that verdict allows.

The evaluation is the core of the project, not a formality. The repository is
organised to make every claim — positive or null — verifiable rather than to
dress it up.

## Non-negotiable rules

**No invented data may pass as measured.** Every session carries an explicit
`DataSource`; `add_session` has no default value; writing synthetic data
requires `allow_synthetic=True`. Data generators live in `tests/fixtures/`,
never in `src/`. This rule exists because the dashboard long displayed 84
generated sessions under the `fit_file` label.

**No metric counts rows.** Series are not guaranteed to be 1 Hz without gaps.
Everything goes through `src/layer_a_signal/timebase.py` and is expressed in
real seconds.

**No physiological constant in code.** They live in `config/params.yaml` and
are read through `src/config.py`. Athlete-specific values are `null` until
`cli calibrate` has run: an explicit error is better than a number founded on a
guess.

**No improvement is claimed without out-of-sample validation.** An in-sample
R² proves nothing. Every variant goes through `src/evaluation/`, against the
baselines declared in `baselines.py`. Splits respect time order.

**Nor without an interval when the gap is thin.** Two variants evaluated under
the same split predict the same points: subtracting their R² does not say
whether the gap exceeds noise. `report.paired_table` performs a paired block
bootstrap — by folds, not by days, since a fold's errors are autocorrelated.
Chronos-2 comes first by 0.003 of R² over persistence; the interval says it is
nothing, and the interval is what is published.

**A variant that ran nothing publishes nothing.** `cross_validate` swallows a
predictor's exceptions: a broken variant comes out as `folds = 0` with NaNs
everywhere, which looks like a result. `cli evaluate` exits with an error in
that case, and warns when two variants did not cover the same number of folds
— a model whose head is trained on a short horizon fails on long folds and its
row then covers fewer points.

**No advertised capability is taken on trust.** TTM's model card advertises
exogenous variable integration; in zero-shot its backbone is
channel-independent and the forecast does not move by a single bit when the
covariates are replaced. A test measures it, and `SPECS` in `foundation.py`
records what each model actually does — not what its card promises.

**Load is only set against a target of its own discipline.** The
most-cited null result of the repository set a load that was two-thirds
cycling against a running speed. Correctly matched — bike load against bike
power — the load model beats "predict the mean" by +0.339 with an interval
[+0.055 · +0.524]. Variants carry their discipline in their name, and running
load serves as a negative control on the bike target: if it predicted anything
there, it would be a leak to look for, not a physiological transfer.

**A null result comes with its floor.** Asserting the absence of an effect
without saying what effect the protocol could see confuses "we found nothing"
with "there is nothing". `sensitivity.py` injects into the real envelope an
effect of known amplitude, of the shape the model postulates, and returns the
smallest one the paired interval distinguishes from chance: 2 sigma against
persistence on the running target, 1 sigma on the bike target. This floor is
OPTIMISTIC, the injected signal having exactly the fitted shape.

**A maximum is not taken twice.** `perf_index` took a maximum over durations
before the envelope took another over 42 days, hence the 0.97 autocorrelation
that makes persistence unbeatable by construction. Durations are published one
by one, and a duration with fewer than forty observations is dropped AND
reported as dropped.

**A result on one athlete is an anecdote.** `cli cohort` replays the
experiment on the athletes in the cache and returns the DISTRIBUTION of
results, never their mean — it would mix very unequal histories. On sixty-one
athletes, the load effect measured in the reference athlete replicates at
p = 0.020: 23 intervals exclude zero in its favour, 33 do not conclude, and 5
AGAINST — load degrades the forecast there. Persistence's dominance replicates
at p < 0.0001.

**A conclusion can be right and underpowered.** At thirteen athletes this
repository concluded that the effect did not generalise (sign test at 1.000);
at sixty-one it replicates. At fourteen athletes the between-athlete dispersion
came out at 0.212 [0.000 · 0.393] and heterogeneity was declared not
established; at sixty-two it is 0.205 [+0.131 · +0.278]. The means did not
move, the intervals tightened. A negative conclusion is therefore stated with
the number of subjects carrying it, and re-read when that number changes.

**Pooling cannot see an athlete who goes against the population.** Shrinkage
towards the mean is what identifies the time constants, and it is also what
prevents the hierarchical fit from producing a single negative gain — not even
for the five athletes where the out-of-sample protocol shows one. The two
protocols agree in rank (correlation +0.406) and not in sign. An individual
verdict is read out of sample, never in the pooled posterior.

**Identified is not validated.** Shared across sixty-two histories, tau1 and
tau2 come out tight (70.3 and 29.7 days) and incompatible with the literature
(42 and 12), tau2 pushing 3.5 standard deviations against its prior. A
parameter that moves away from its prior AS DATA ARE ADDED signals a
specification defect, not a discovery. A tightening posterior is not
validation.

**A parameter is challenged by moving the protocol, not by commenting on it.**
The hypothesis was that these constants learn the envelope's smoothing window.
Measured at window 21 instead of 42: tau1 loses 14.4 % and tau2 18.9 %, an
elasticity of 0.22 and 0.30 — zero for pure physiology, one for pure
window-learning. BOTH hypotheses are therefore false, and about a quarter of
these constants is protocol. Three traps on that path. `min_periods` applies to
the window, so shortening it drops athletes: a third CONTROL fit, at standard
window on the cohort minus the same athletes, is what clears composition
(70.3 -> 71.7). The comparison bears on a target that changed: its lag-1
autocorrelation drops from +0.539 to +0.343, `describe` publishes it, and
without that number one would attribute to physiology a contraction of ALL
parameters by 8 to 19 %. Finally an unpublished setting writes suffixed
artifacts (`_f21`, `_n60`), because an experiment that questions a measurement
must not be able to overwrite it.

**"Who responds?" is asked out of sample.** Searching among twelve predictors
for the one most correlated with the gain and publishing it means testing
twelve hypotheses and declaring one. `cli responders` leaves one athlete out
and asks the others to classify them, preprocessing recomputed WITHIN each fold
— computing it once on the whole sample would leak the target through
centring. Result: no family beats the majority class. The two nulls are not
equal, though, and the floor separates them: the family known ON DAY ONE
detects a shift of 0.05 sigma, so its null is well powered and the cold-start
question is CLOSED; the history-derived one does not reach its floor at
0.40 sigma, so its null is not established. Two defects came out of this
measurement: a CONSTANT predictor (`sexe_feminin`, zero for all 61 athletes)
came out significant because the permuted p of a NaN is 0.0000, and six
athletes declared a NEGATIVE age that pushed the age correlation from +0.132 to
+0.230. A dropped predictor is reported as dropped, and the Bonferroni price is
paid on the predictors actually tested.

**A plan says what it does not optimise.** The planner looks for the MINIMUM
load that delivers the requested stimulus, not the maximum the constraints
tolerate: under an unknown dose-response — it exists in 23 athletes out of 61,
and is harmful in 5 — the cost of unnecessary load is real where its benefit
is not. `dose_response.py` measures the verdict for the athlete at hand and
distinguishes "not detected" from "undeterminable", because presenting an
absence of measurement as an absence of effect is the mistake this repository
works not to make. None of the verdicts authorises maximising a predicted
performance.

**A safety bound is measured, and three confounders await it.**
`cli validate-plan` confronts the planner's bounds with 6,768 real weeks from
66 athletes: compliance versus violation WITHIN each athlete, then a sign test.
The retention floor (ACWR >= 0.8) comes out SUPPORTED — +0.051 sigma, 37/55,
p = 0.0145, detected without injection. The ACWR ceiling and the
consecutive-days ceiling come out UNDETERMINABLE, floor 0.20 sigma. Three traps
were stacked on the way, and the naive protocol claimed that respecting the
ACWR ceiling COST 0.333 sigma at p < 0.0001: the envelope, a rolling maximum,
has a median change of zero over 28 days; a backward-looking rolling maximum
includes the week being judged, so that week's hard effort enters its own
outcome; and a maximum over more draws is larger, high-ACWR weeks being
followed by 5.2 more rides. Every rule is therefore judged on TWO statistics —
best effort and MEDIAN effort, insensitive to the number of draws — and on TWO
regimes, one of which requires regular training BEFORE the judged week: a high
ACWR is most often a return after a break, whose reference window is depressed.
A rule that holds on only one of the four rows does not hold.

**An isolated case is judged against a base rate.** Five athletes out of
sixty-one see load degrade their forecast. No artefact was established there:
data gaps, level breaks and scale breaks were compared with the DISTRIBUTION of
the same statistic among the 23 detected, and four of the five stay within the
range. Without a comparison group, any value looks suspicious. The most natural
mechanistic hypothesis — a trending load makes the model extrapolate upward —
is REFUTED: median bias is −0.045 sigma, 27 athletes out of 61 are
overestimated, and rho(bias, gain) = −0.018 on 61 subjects. What governs the
score is the magnitude of the error, not its sign.

**A degraded plan does not read as an optimal plan.** `optimize_week` silently
fell back on the heuristic as soon as a constraint made the problem infeasible.
`WeeklyPlan.notes` now records what was not applied — floor lowered, stimulus
dropped, acute/chronic bound inactive for lack of history — and `xai_report`
publishes it. A silently inactive bound is worse than an absent one.

**Load must compete with the passage of time for the signal.** A first
hierarchical fit let tau1 escape to 127 days, 4.6 standard deviations beyond
its prior: at that constant, cumulative load IS a trend, and the gain measured
one while believing it measured the other. Every load model therefore carries a
competing trend term — not to improve the fit but to contest it.

**A threshold is not copied.** `readiness.py` announces when each capability
becomes available — 28 days for the acute/chronic bound, three measurements for
the envelope, eight folds for the verdict. Each is the one a function actually
imposes, and the counts go through those functions (`horizon_splits`,
`rolling_envelope`) rather than reimplementing them. A copied threshold drifts
the day the function changes, and the ladder becomes a reassuring lie: a test
fails if the two diverge. Same rule for the population prior, read from the
cohort artifact and never hard-coded.

**Tests never write to `data/`.** An autouse fixture checks it after every test
and fails otherwise. The sensitivity module is no exception: the perturbed
envelope exists only in memory, and the interface returns only statistics —
never a series that could be written and later mistaken for a measurement.

## Layout

```
README.md                project presentation
docs/results.md          full log of measurements
docs/method.md           protocols, what is missing, safeguards
docs/architecture.md     modules, data flow, getting started

src/config.py            single configuration, cached
src/ingestion/           access to the public dataset (cache-first, md5)
src/layer_a_signal/      cleaning, features, timebase, performance
src/layer_b_twin/        load model, Bayesian model,
                         hierarchical pooling of athletes
src/layer_c_planner/     safety constraints, optimiser, LLM parsing,
                         per-athlete dose-response verdict
src/pipeline/            orchestrator and provenance
src/evaluation/          metrics, baselines, validation, report,
                         foundation models, sensitivity, cohort, readiness,
                         plan rules against data,
                         predictors of the response to load
app/                     Streamlit dashboard
```

## Commands

```bash
pip install -e ".[dev]"
python -m src.cli ingest        # build the artifacts
python -m src.cli calibrate     # write estimates to params.yaml
python -m src.cli evaluate      # five protocols, floors, provenance
python -m src.cli cohort        # replication on the athlete cache
python -m src.cli fetch-athletes --critere velo|course   # grow the cache
python -m src.cli plan          # proposed week, and what it does not promise
python -m src.cli validate-plan # what the plan's bounds are worth against data
python -m src.cli responders    # who responds, tested out of sample
python -m src.cli pool          # hierarchical fit (slow: hours)
python -m src.cli pool --decrire               # the target only, in minutes
python -m src.cli pool --fenetre 21            # challenge the time constants
python -m src.cli pool --exclure UUID,UUID     # control cohort composition
pytest                          # offline by default

pip install -e ".[tsfm]"        # torch + Chronos-2 + TTM, optional
python -m src.cli evaluate --models chronos2,ttm
ruff check src app tests
streamlit run app/main_dashboard.py
```

## Documentation

Documentation is written in English. `README.md` is the project presentation,
for a reader discovering it: concise, result first, and no number that a
command cannot reproduce. The full log of measurements lives in
`docs/results.md`, protocols, gaps and safeguards in `docs/method.md`, the
architecture in `docs/architecture.md`. A new result is written in `docs/`, and
only moves up to the README if it changes what the project concludes. A
limitation is stated as the data or condition that would lift it.

## Style

Code comments and docstrings stay in French, as do CLI and dashboard output,
explaining **why** rather than **what**. A comment that paraphrases the
following code is useless; a comment that explains why a threshold is 42 rather
than 30, or why a guard exists, prevents it from being undone six months later.

Commit messages in the imperative, with an explanatory body when the change is
not obvious. No standardised sections or status emojis.
