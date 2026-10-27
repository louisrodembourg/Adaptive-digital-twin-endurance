# Method

How the numbers are produced, what is real, what is missing, and the rules
the code enforces so that a result cannot look better than it is.

Short presentation: [README](../README.md).
The measurements themselves: [results](results.md).

---

## What is real, and what is not

**Real.** 917 sessions from one GoldenCheetah OpenData athlete, August 2014 to
January 2020: 648 cycling, 172 running, 49 without discipline, 23 indoor
trainer, 21 swimming. No personal data, no GPS coordinates. The cohort
experiments add 61 to 66 further athletes from the same public dataset.

**Validated against an external implementation.** GoldenCheetah publishes its
own TRIMP. Regressing ours against it on 863 sessions: **slope 0.934,
intercept +1.05, r = 0.982, residual sd 15.6**. The slope matters more than the
correlation — it says our scale is 7 % lower. A rank coefficient would have
given 0.97 while completely hiding this gap, whereas TRIMP is precisely a
scale quantity feeding a linear convolution.

**Estimated, not assumed.** Max HR 194 bpm and resting HR 62 bpm, derived from
the history. The previous version assumed 205 and 65, annotated "to
calibrate", while the whole TRIMP depends on them.

**Absent, and not simulated.** Neither this dataset nor the `.fit` files from
a Coros watch contain RR intervals. DFA α₁ therefore returns `NaN`, and two of
the safety rules have no source. An earlier version fabricated these intervals
with a fixed-seed AR(1) process and used them to block high-intensity
sessions; the generator now lives in `tests/fixtures/`, under a name that says
what it produces.

**Extrapolated, and flagged as such.** Race-time predictions beyond 45 minutes
extrapolate a Riegel law fitted on 5- to 45-minute efforts (exponent 0.0771,
R² 0.987). No half-marathon appears in this athlete's data.

---

## What's missing, and what each gap costs

The protocol is in place and has produced a replicated result: bike load
predicts bike power in 23 athletes out of 61 (sign test p = 0.020), with an
established between-athlete heterogeneity. Each limitation below is stated as
**the data or condition that would lift it**, and the result it would unlock.

### Data the public dataset does not carry

**Session types.** Public summaries do not say whether a session was
intervals, tempo or endurance. Consequence: the planner's per-type ceilings
cannot be tested (`validate-plan` does not reach them), and the
consecutive-days rule cannot be restated by intensity. *Unlocks:* validating
the half of the planner that is still unmeasured. A personal export
(GoldenCheetah, Strava, Garmin) carries them.

**Declared maximal efforts.** The bike target `cp20_w` is the best
twenty-minute power of a ride, without the heart-rate guard `performance.py`
applies to running. A rolling envelope takes a maximum, so an easy ride cannot
lower the target — but nothing guarantees that the 417 observations are tests,
and the +0.279 should be read with this caveat. *Unlocks:* a target that
measures capacity rather than the ride. Tests and race results, tagged as
such, would do.

**HRV, sleep, perceived exertion.** Fatigue is today inferred from load alone.
The HRV threshold rule exists in the code and is wired, but no source feeds it:
the dataset has no RR intervals. *Unlocks:* a fatigue state measured rather
than modelled, and the two safety rules that currently have no input.

**Equipment.** One of the five athletes for whom load hurts shows a scale break
compatible with a smart-trainer change; public summaries do not record the
device. *Unlocks:* settling whether some negative cases are measurement
artefacts.

### Subjects

**More athletes.** Two results are explicitly underpowered at 61 athletes:
history-derived responder predictors do not reach their detectability floor
even at 0.40 σ, and the two gaps that set the five negative athletes apart
(load slope, absolute error, p = 0.074 and 0.070) sit about 20 % below what the
protocol can see. The repository has already seen two conclusions flip when
the cohort grew (sign test from p = 1.000 at 13 athletes to 0.020 at 61;
heterogeneity from "not established" at 14 to established at 62). *Unlocks:*
an answer to "who responds?" from history, and a verdict on the negative
cases.

**Running histories across several athletes.** The replication covers cycling
power only, the only comparable quantity available for enough subjects in the
cache. Everything the repository says about *running* still rests on one
athlete. *Unlocks:* a cohort result for the second discipline.

### Conditions of use

**Planned, not realised, future load.** Foundation models receive the future
load on purpose — the question is "does load carry information about
performance?", and refusing it would make a null result unusable. But it means
Chronos-2's +0.657 is not a planner's figure: in real use, the coming load is
not known. *Unlocks:* an honest planner score, measured on the load as it was
planned.

**Stability of the individual verdict.** The planner proposes a week under
constraints; it does not forecast what it will produce, and the dose-response
verdict it shows is measured on the athlete's past. Nothing guarantees a
responder stays one. *Unlocks:* repeated verdicts over time for the same
athlete.

**Over 400 days of history for an individual verdict.** See
[the maturity ladder](#the-maturity-ladder): until then the system serves the
cohort prior.

### Model limitations, measured

**The single-athlete model is not identified.** k₁ and k₂ are correlated at
0.986 in the posterior, and the time constants tighten by only 41 % relative to
their prior. At *fixed* time constants, collinearity looks moderate — the
least-squares matrix has condition number 5 — but as soon as τ is sampled, τ
and k compensate each other and the degeneracy appears. Only the posterior
reveals it. The sign of k₂ is not constrained in the model: forcing it would
hide precisely this result. Pooling across 62 athletes identifies the time
constants, but at values (70.3 and 29.7 days) incompatible with the literature
(42 and 12), about a quarter of which is measured to come from the smoothing
window.

**The envelope is autocorrelated by construction**, hence easy to predict. The
+0.318 of persistence measures inertia, not an understanding of physiology —
and the detectability floor puts a number on what that costs: a 2 σ load effect
would be needed to beat it demonstrably on the running target, 1 σ on the bike
target.

**The sport-transition confounder remains open on the running target.** The
athlete moves from cycling to running in 2018, and the whole evaluation window
of the running target is this transition: progress due to motor learning is
indistinguishable from progress due to load. A competing time trend absorbs
nothing (−0.268), which rules out the simplest form of the confounder, not the
confounder itself.

**A null result only holds on these data.** On the running target: eighteen
folds, one athlete, a majority discipline. The paired interval of
[−0.121 · +0.430] says this protocol cannot rule out a real covariate gain of
up to +0.43 — only that it does not detect one. Both readings matter.

---

## The maturity ladder

The dose-response verdict requires 180 days of context then eight 28-day
folds, i.e. more than four hundred days of history. Before that
`assess_dose_response` returns "undeterminable" — accurate, and useless: a new
user hears nothing for over a year without knowing why, or for how much
longer.

Yet the system's capabilities do not all appear at once.
`src/evaluation/readiness.py` returns the ladder, and for each missing rung
what is missing, in days or measurements:

| capability | threshold | imposed by |
|---|---|---|
| load (fitness, fatigue) | 1 day; stabilised after ~τ₁ = 42 d | `compute_load_curves` |
| acute/chronic bound | 28 full days | `optimizer._chronic_daily_load` |
| own freshness thresholds | 30 points | `load_model.form_thresholds` |
| performance envelope | 3 measurements within 42 days | `rolling_envelope` |
| evaluable forecast | 1 fold | `horizon_splits` |
| dose-response verdict | 8 folds | `assess_dose_response` |

**No threshold is copied.** Each is the one a function actually imposes, and
the counts go through those functions — `horizon_splits` for folds,
`rolling_envelope` for the envelope. A copied threshold would drift the day
the function changes, and the ladder would become a reassuring lie. A test
fails if either moves.

### What the system can say on day one

Nothing about the athlete — but something measured on the population:

```
$ python -m src.cli state      # excerpt, athlete on day ten

Maturite : 1/6 capacites disponibles

  [oui] charge (fitness, fatigue)          10 jour(s) de calendrier, pas encore stabilisee (t1 = 42 j)
  [non] borne aigu/chronique               exige 28 jours pleins de charge chronique
                                          manque : 18 jour(s) de calendrier
  [non] seuils de fraicheur propres        exige 30 points de fraicheur
                                          manque : 20 jour(s) de calendrier
  [non] enveloppe de performance           0 mesure(s)
                                          manque : 3 mesure(s) dans une fenetre de 42 jours
  [non] prevision evaluable                0 pli(s) d'horizon 28 j
                                          manque : 208 jour(s) d'enveloppe definie
  [non] verdict de dose-reponse            exige 8 plis, 0 disponible(s)
                                          manque : 224 jour(s) (8 pli(s))

  En attendant son propre verdict :
    Chez 23 athlete(s) sur 61 de la cohorte, la charge explique la performance avec un intervalle qui exclut zero ;
    chez 38, non. Rien dans les donnees disponibles a l'inscription ne dit lequel vous etes.
```

(The CLI speaks French. Each `[non]` line says what is missing, in calendar
days or measurements; the closing lines say: "for 23 athletes out of 61 in the
cohort, load explains performance with an interval excluding zero; for 38, it
does not. Nothing available at sign-up says which one you are.")

This prior is read from `cohort_replication.parquet`, never copied: a
hard-coded population figure becomes wrong as soon as the cohort grows, and
nobody notices. It is only served while the athlete has no verdict of their
own — giving it afterwards would blur what concerns them.

---

## Repository safeguards

These rules are enforced by code and tests, not just documented. Details in
[`AGENTS.md`](../AGENTS.md).

- **Mandatory provenance.** `DataSource` has no default value; writing
  synthetic data requires `allow_synthetic=True`. The dashboard once displayed
  84 generated sessions under the `fit_file` label.
- **No metric counts rows.** Everything goes through `timebase.py` and is
  expressed in real seconds; series are not guaranteed to be 1 Hz.
- **No physiological constant in code.** They live in `config/params.yaml`,
  and are `null` until calibration has run — an explicit error rather than an
  invented number.
- **No improvement claimed without out-of-sample validation**, nor without an
  interval when two variants are a few thousandths of R² apart. A ranking is
  not a measurement: `report.paired_table` compares variants point by point,
  by paired block bootstrap over folds.
- **A model that ran nothing publishes nothing.** `cross_validate` swallows a
  predictor's exceptions; a broken variant would come out as `folds = 0` with
  NaNs, which looks like a measured result. `cli evaluate` exits with an error
  in that case, and warns when two variants did not cover the same number of
  folds.
- **No advertised capability is taken on trust.** TTM's "exogenous variable
  integration" is inert in zero-shot; a test measures it by replacing the
  covariates and checking the output does not move.
- **A null result comes with its floor.** Asserting the absence of an effect
  without saying what effect the protocol could see confuses "we found
  nothing" with "there is nothing". `sensitivity.py` injects an effect of known
  amplitude into the real envelope and returns the smallest one the paired
  interval distinguishes from chance.
- **Load is only set against a target of its own discipline.** Setting a load
  that was two-thirds cycling against a running speed produced the most-cited
  null result of this repository. Variants now carry their discipline in their
  name, and running load serves as a negative control on the bike target: if
  it predicted anything there, a leak would have to be found before speaking of
  transfer.
- **A result on one athlete is an anecdote.** `cli cohort` replays the
  experiment on the cache and returns the **distribution** of results, never
  their mean: how many athletes see the effect, for how many the interval
  excludes zero, and in which direction the others err. This is what showed
  the reference athlete's effect replicates (23 of 61, p = 0.020) — and that
  it is not typical.
- **A plan says what it does not optimise.** The planner looks for the minimum
  load that delivers the stimulus, never the maximum tolerated: under an
  unknown dose-response, the cost of unnecessary load is real where its benefit
  is not. And `WeeklyPlan.notes` publishes what the solver had to drop — a
  degraded plan that does not announce itself reads as an optimal plan.
- **A safety bound is measured too.** `cli validate-plan` confronts the
  planner's bounds with 6,768 real weeks from 66 athletes. The retention floor
  comes out **supported**, the ACWR ceiling and the consecutive-days ceiling
  **undeterminable** within 0.20 σ. Every bound is judged on two statistics —
  best effort and **median** effort, insensitive to the number of draws — and
  on two regimes, one of which requires regular training before the judged
  week. Without these four rows, the naive protocol claimed that respecting the
  ACWR ceiling cost 0.333 σ at p < 0.0001: a sampling artefact compounded by
  selection bias.
- **An isolated case is judged against a base rate.** The five athletes for
  whom load degrades the forecast were investigated one by one, and each
  artefact statistic compared with its distribution among the 23 detected
  athletes. Without a comparison group, any value looks suspicious — four of
  the five stay within the range of the detected.
- **A predictor searched among twelve is tested out of sample.**
  `cli responders` leaves one athlete out and asks the others to classify
  them, preprocessing recomputed within each fold. No family beats the
  majority class — and the two nulls are not equal: the one for what is known
  on day one is well powered (floor 0.05 σ), the one for history is not (floor
  not reached at 0.40 σ). A **constant** predictor is dropped and reported as
  dropped: the permuted p of a NaN was 0.0000, and this degenerate predictor
  was the only one declared retained by the first run.
- **A parameter is challenged by moving the protocol.** The shared time
  constants come out tight and incompatible with the literature; the
  hypothesis was that they learn the envelope's smoothing window.
  `cli pool --fenetre 21` puts it to the test, and two safeguards make it
  readable. First a **composition control**: `min_periods` applies to the
  window, so shortening it drops two athletes, and `--exclure` replays the fit
  at standard window on the reduced cohort — without it, the shift in τ would
  have two mixed causes. Then the **characterisation of the target**, which
  `describe` publishes with every fit: its lag-1 autocorrelation drops from
  +0.539 to +0.343, and without that number one would attribute to physiology
  a contraction of **all** parameters by 8 to 19 %. Verdict: elasticity 0.22
  for τ₁ and 0.30 for τ₂ — neither pure physiology nor pure window-learning. An
  unpublished setting writes suffixed artefacts, because an experiment that
  questions a measurement must not be able to overwrite it.
- **Tests never write to `data/`**; a fixture checks it after every test. The
  sensitivity module is no exception: the perturbed envelope exists only in
  memory and the interface returns only statistics, never a series that could
  later be mistaken for a measurement.

The Bayesian model's convolution kernel is tested against the recursive filter
in `load_model.py` to within 1e-10: a single source of truth for load
dynamics. And a **parameter-recovery** test simulates from known values then
checks that the posterior recovers them — all five fall within their 94 %
credible interval.
