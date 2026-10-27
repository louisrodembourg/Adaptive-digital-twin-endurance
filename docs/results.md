# Measured results

The full log of measurements. Every number is reproduced by a command
(`python -m src.cli evaluate`, `cohort`, `pool`, `validate-plan`,
`responders`) on a public dataset — which is what makes them verifiable rather
than declarative.

The first two sections give the result and the starting point it came from.
Everything after them follows the order in which the measurements were made,
including the conclusions that later measurements corrected.

Short presentation: [README](../README.md).
Protocols and safeguards: [method](method.md).

---

## The headline result

**Training load predicts performance once it is matched to the discipline
being measured** — bike load against bike power. On the reference athlete,
over 417 observations and 38 time-ordered folds:

| comparison | skill gain | 95 % CI | P(gain) |
|---|---:|---|---:|
| bike load **vs. running load** (negative control) | +0.631 | [+0.407 · +0.742] | 1.000 |
| bike load **vs. predict the mean** | **+0.339** | **[+0.055 · +0.524]** | 0.986 |

It replicates across the cohort, and the cohort is what tells us how far it
reaches:

| on 61 athletes | value |
|---|---|
| intervals excluding zero in favour of load | **23 / 61** |
| inconclusive | 33 / 61 |
| intervals excluding zero **against** load | 5 / 61 |
| sign test, load vs. mean | **p = 0.020** |
| discipline pairing (35 athletes doing both) | +0.481 median, 14 in favour, 0 against, p = 0.0001 |
| between-athlete dispersion of the effect (hierarchical model, 62 athletes) | **0.205 [+0.131 · +0.278]** |

Read together: **the dose-response effect is real, it is heterogeneous, and
it can be measured athlete by athlete** — which is what the planner uses.
What is still missing is the ability to predict, on day one, which group a new
athlete belongs to; [the last section](#who-responds-the-open-question-asked-out-of-sample)
shows why this needs more athletes rather than more predictors.

Details: [matching load to target](#matching-load-to-target-and-measuring-what-the-protocol-can-see),
[the 61-athlete replication](#what-sixty-one-athletes-say),
[pooling](#what-pooling-the-athletes-adds).

---

## Starting point: a mismatched protocol

The first protocol asked the classic question — does the Banister
fitness-fatigue model predict performance out of sample? — on a **running**
target. On 917 real sessions, in time-ordered cross-validation:

| variant | in-sample R² | out-of-sample R² |
|---|---:|---:|
| **persistence** (repeat the last value) | −0.023 | **+0.318** |
| exponential moving average | −0.215 | +0.081 |
| time trend | +0.271 | +0.042 |
| predict the mean | 0.000 | −0.014 |
| fitness only | +0.066 | −0.060 |
| **Banister (fitness + fatigue)** | **+0.107** | **−0.090** |

Three things show in it.

**Banister has the best fit and the worst out-of-sample score.** It memorises,
it does not learn. This is the very signature of overfitting, and it
replicates the criticism published in [*Scientific Reports* (2025)](https://www.nature.com/articles/s41598-025-88153-7)
on the ill-conditioning of the fitness-fatigue model.

**The trend drops from +0.271 to +0.042.** A good fit is worth nothing on its
own — which is why this repository never publishes an in-sample R² alone.

**What predicts is inertia.** Persistence scores +0.318, and no load model
beats it. It replicates: across 61 athletes, three beat persistence, sign test
p < 0.0001.

Two things later turned out to limit what this table can say. The load was
68.5 % cycling while the target was a running speed — the
[mismatch fixed below](#matching-load-to-target-and-measuring-what-the-protocol-can-see).
And beating persistence on a target autocorrelated at 0.97 would take a load
effect **twice as dispersed as the whole envelope**: the comparison has
almost no power, and this is measured — see
[the detectability floor](#the-detectability-floor-and-what-it-does-to-the-conclusion).

---

## What makes this result possible

The project initially measured performance by the **best effort of each
session**. This is a poor estimand: it describes what the session *was* —
intervals or recovery — not the athlete's capacity.

| observable | autocorrelation (lag 1) |
|---|---:|
| best effort per session, running | −0.12 |
| best effort per session, 20-min power | −0.00 |
| **42-day rolling envelope, running** | **+0.79** |
| **42-day rolling envelope, cycling** | **+0.93** |

Noise around a level: no model could extract anything from it. The envelope
answers a different question — "what is the best this athlete has been
capable of recently?" — and becomes predictable.

The trade-off is accepted: an envelope is autocorrelated **by construction**,
hence easy to predict by mere inertia. This is exactly why persistence is
among the mandatory baselines; without it, the +0.318 would be told as a
success of the model.

---

## What the posterior says

The Bayesian model is fitted by MCMC on 1,987 days of load and 85 envelope
points. Sampling converges cleanly — `r_hat` at 1.0000 on every parameter, no
divergence, minimum ESS of 746 — which makes its diagnostics interpretable.

| parameter | mean | sd | 94 % interval |
|---|---:|---:|---|
| τ₁ (fitness, days) | 29.3 | 11.7 | [9.1 · 51.1] |
| τ₂ (fatigue, days) | 13.6 | 4.8 | [4.5 · 22.1] |
| k₁ (fitness gain) | 0.050 | 0.032 | [0.007 · 0.112] |
| k₂ (fatigue gain) | 0.050 | 0.040 | [−0.004 · 0.127] |
| **k₁ − k₂ (net effect)** | **−0.000** | **0.010** | **[−0.020 · 0.017]** |

**On this running target, the net effect of training load on performance is
indistinguishable from zero.** It is not an imprecise estimate for lack of
data: the interval is narrow, centred on zero, and comes from converged
sampling.

**k₁ and k₂ are correlated at 0.986.** The data cannot tell a high fitness gain
offset by a high fatigue penalty from a pair of modest gains — only their
difference is constrained. The posterior of τ₁ tightens by only 41 % relative
to its prior: the time constants are not identified either.

**The 95 % prediction interval covers only 34 % of observations.** The model
is massively overconfident.

A flat posterior is a result, not a failure. It is the only honest use of
Bayesian inference here: showing what the data actually constrain, where a
point fit would have presented k₁ = 0.050 as an estimated value.

> **Engineering note.** This geometry is expensive: with parameters correlated
> at 0.99, NUTS only walks the ridge in tiny steps, and the first fit took
> **22 hours**. Sampling the net effect rather than each gain — same model,
> different parametrisation — brings it down to **26 minutes**, a factor of 51.

---

## What the foundation models say

The result above rests on **a single parametric model**. One objection
therefore remains open: perhaps load does predict performance, but not in the
form of an exponential convolution with two time constants.

Time-series foundation models let us test this without assuming a functional
form. [Chronos-2](https://huggingface.co/amazon/chronos-2) accepts past and
known-future covariates: we give it the envelope as target and the load as
covariate, then compare it **to the same model deprived of the covariate**.
The gap between the two rows is a non-parametric measure of the information
load carries about performance.

The covariates are not just raw TRIMP: they include `fitness` and `fatigue`,
computed by the same function that feeds the Bayesian model. The model thus
receives **exactly Banister's regressors**, free to combine them differently.
A null result cannot be explained by insufficient feature shaping.

### The result

28-day calendar-horizon protocol, 18 rolling-origin folds, 545 days of
defined envelope:

| variant | in-sample R² | out-of-sample R² | folds |
|---|---:|---:|---:|
| **Chronos-2 + load** | — | **+0.657** | 18 |
| **persistence** | −0.008 | **+0.654** | 18 |
| Chronos-2 without load | — | +0.621 | 18 |
| TTM without load | — | +0.514 | 17 |
| exponential moving average | −0.001 | +0.503 | 18 |
| time trend | +0.281 | +0.122 | 18 |
| predict the mean | +0.000 | +0.090 | 18 |
| fitness only (τ 60) | +0.099 | +0.050 | 18 |
| Banister (τ 42/12) | +0.100 | −0.017 | 18 |

Chronos-2 fed with load comes first. **By +0.003.** A ranking is not a
measurement, and a gap of this size means nothing until it is bounded. Both
variants predict exactly the same points under the same split, which allows a
paired bootstrap — by **blocks**, since errors within a fold are strongly
autocorrelated and drawing days independently would assume 545 independent
observations where there are 18:

| comparison | gain | 95 % interval | P(gain) |
|---|---:|---|---:|
| Chronos-2 + load **vs.** Chronos-2 without load | +0.096 | [−0.121 · +0.430] | 0.70 |
| Chronos-2 + load **vs.** persistence | +0.008 | [−0.172 · +0.255] | 0.55 |

**Neither gap survives.** The gain brought by load is compatible with zero,
and Chronos-2's first place against persistence is a coin toss. A
120-million-parameter model, trained on series corpora this project does not
have, given past *and future* load — hence more information than any planner
would have — extracts nothing more than "repeat the last value" on this
running target.

It is the same finding as the Bayesian posterior, reached by a route that
shares neither functional assumption, nor inference method, nor code with it.
That is what makes it interesting.

### Three things the measurement learned along the way

**A zero-shot model has no in-sample R².** The column is empty for the three
foundation rows, and that is a result: these models fit nothing, their weights
are frozen. There is no fit / out-of-sample gap, hence **no overfitting to
measure** — exactly the opposite of Banister, whose best fit came with the
worst out-of-sample score.

**TTM's advertised exogenous support is inert in zero-shot.** The
[granite-timeseries-ttm](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r3)
model card advertises "exogenous variable integration". Measured: replacing
the covariate channels entirely leaves the forecast of the target channel
**bit-for-bit identical** — the backbone is channel-independent and
`enable_forecast_channel_mixing` is `False`. Its exogenous support requires
fine-tuning. TTM is therefore registered univariate only, and `foundation.py`
refuses `with_covariates=True` for it rather than publish two rows known to be
equal. A test checks it.

**TTM covered only 17 folds out of 18.** Its head is trained for a 96-step
horizon; the first fold asks for 152, because of a 135-day measurement gap for
this athlete. Its row therefore does not cover the same points as the others.
This is why the `folds` column exists: without it the comparison would look
like a level playing field, and `cli evaluate` warns on stderr when fold
counts differ.

### The two protocols are not comparable with each other

Persistence goes from **+0.318** (block protocol, above) to **+0.654** here.
Nothing improved: the target changed grid.

The envelope as the repository reported it only exists on days where a
measurement exists — 85 points spaced 1 to 140 days apart. A test block then
covers a highly variable span, and the aggregate score mixes one-week and
five-month forecasts; as a measure of a forecasting horizon, it means nothing.
Foundation models, for their part, almost all assume a constant sampling step.

Hence the daily grid. This is **not** interpolation: a 42-day rolling maximum
is defined for every calendar day from the real measurements in its window
alone, and the code simply never asked for it. A test enforces it — the daily
output, restricted to observation dates, is identical to the old one. Days
where the envelope is undefined carry `NaN`, never the last known value:
carrying forward would have fabricated data and, worse, made the target
artificially easy to predict by persistence.

This grid is denser and more autocorrelated (0.974 at lag 1, versus 0.79 for
the 85 points), so every score rises. The relative ranking is unchanged: no
load model beats inertia.

### Models retained, and those that were not

| model | size | weights licence | zero-shot covariates | run |
|---|---:|---|---|---|
| [Chronos-2](https://huggingface.co/amazon/chronos-2) | 120 M | Apache-2.0 | past **and known future** | **yes** |
| [TTM](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r3) (IBM granite) | 855 k | Apache-2.0 | none — measured | **yes, univariate** |
| [TabPFN-TS](https://github.com/PriorLabs/tabpfn-time-series) | 11 M | Apache-2.0, **weights behind account** | yes | no |
| [TimesFM 2.5](https://huggingface.co/google/timesfm-2.5-200m-pytorch) | 200 M | Apache-2.0 | none | no |
| TimesFM 3.0 | 330 M | **non-commercial** | yes | no |
| [Moirai-2](https://huggingface.co/Salesforce/moirai-2.0-R-small) | 11 M | **CC BY-NC 4.0** | — | no |

Four exclusions, four distinct reasons, none related to performance.

**Moirai-2 and TimesFM 3.0 forbid commercial use of their weights.** They are
two of the models benchmarks rank highest. This repository is MIT; citing them
without running them is the consistent choice.

**TabPFN weights cannot be downloaded without an account.** The code is
Apache-2.0, but `tabpfn` 9.x requires interactive licence acceptance and a
`TABPFN_TOKEN` key obtained from Prior Labs. The adapter is written and
functional; no number is published until it has run.

**TimesFM 2.5 accepts no covariate**, which makes it useless for the only
question asked here.

A version detail that matters: the checkpoint actually loaded for TTM is
**R2, not R3**. `granite-tsfm` 0.3.10, the latest version on PyPI, cannot read
R3 checkpoints — half the weights, including the multi-quantile head, are
randomly re-initialised and the output is not finite. The variant key is
called `ttm` rather than `ttm-r3` for that reason, and a test asserts it.

### On this machine

Latencies measured on a MacBook Pro M3 Pro, 18 GB, **CPU only**, weight
loading excluded from the timer:

| model | per forecast | weights |
|---|---:|---:|
| Chronos-2, with covariates | 138 ms | 120 M |
| Chronos-2, univariate | 155 ms | 120 M |
| TTM | **3 ms** | 855 k |

TTM's 3 ms reproduce IBM's published figure exactly. No GPU is needed: the
first Chronos-2 load takes about a hundred seconds, then the pipeline is
cached and the 18 folds run in a few seconds.

```bash
pip install -e ".[tsfm]"          # torch, chronos-forecasting, granite-tsfm
python -m src.cli evaluate --models chronos2,ttm
```

The command writes `data/processed/evaluation_results.parquet` and a
provenance JSON carrying package versions, weight revisions, protocol
parameters and latencies. It outputs **both** protocols: the six rows of the
[starting-point table](#starting-point-a-mismatched-protocol), which used to
be copied from the dashboard, now come out of it to the digit. That is also
what protects them — any drift on these six rows signals a regression of the
target or the split, not a result.

---

## Matching load to target, and measuring what the protocol can see

The imbalance between disciplines was already listed among the repository's
known limitations. What was missing was the experiment: over the evaluation
window,

| discipline | share of TRIMP |
|---|---:|
| Bike + VirtualRide | **68.5 %** |
| Run | 30.2 % |
| other (swimming, walking, kayak) | 1.3 % |

while the performance index is a **running speed ratio**. A single-coefficient
model must therefore find a `k₁` that holds both for a kilojoule of cycling
and for a kilometre of running. The bike → run transfer is partial and has a
different time constant, while fatigue accumulates fully: the coefficient
averages two incomparable transfers and comes out at zero.

`build_daily_load` now splits load into `trimp_run`, `trimp_bike` and
`trimp_other`, which sum exactly to `trimp` — the reference column is left
intact and the six rows published above reproduce to the digit.

### On the running target: the right direction, without the proof

| variant | out-of-sample R² (28-day horizon) | in-sample R² |
|---|---:|---:|
| persistence | **+0.654** | −0.008 |
| **banister run only** t60/15 | **+0.173** | +0.216 |
| banister run only t42/12 | +0.124 | +0.139 |
| time trend | +0.122 | +0.281 |
| predict the mean | +0.090 | 0.000 |
| banister t42/12 (total load) | −0.017 | +0.100 |
| banister per sport t42/12 | −0.155 | +0.305 |
| banister + trend t42/12 | −0.268 | +0.398 |

Restricting load to running moves the model from −0.017 to +0.124, ahead of
"predict the mean". But the interval refuses to conclude:

| variant | vs. | gain | 95 % CI | P(gain) |
|---|---|---:|---|---:|
| run only | total load | +0.139 | [−0.117 · +0.395] | 0.852 |
| run only | predict the mean | +0.038 | [−0.144 · +0.189] | 0.684 |
| run only | **persistence** | **−1.533** | [−6.279 · +0.048] | 0.032 |

Two variants fail outright, and it is instructive. **One pair of coefficients
per discipline** gets the best in-sample R² of the table (+0.305) and one of
the worst out of sample (−0.155): seven parameters on 85 autocorrelated
points, hand-made overfitting. **A time trend** put in competition does not
absorb load, it adds variance (−0.268) — the athlete moves from cycling to
running in 2018 and the whole window is a sport transition, but this
confounder is not captured by a linear term. It remains open.

### Persistence collapses as the effort gets longer

`perf_index` took a maximum **over durations** before the envelope took
another over 42 days: a double ratchet, hence the 0.97 autocorrelation that
makes persistence unbeatable by construction. A 5-minute ceiling depends
mostly on anaerobic power, a 45-minute ceiling on aerobic endurance.
Per-duration observations already existed in `mean_max_speed`; they are now
published one by one.

| duration | obs. | days | persistence | mean | run only | total load |
|---|---:|---:|---:|---:|---:|---:|
| 300 s | 90 | 545 | +0.637 | +0.095 | +0.068 | +0.007 |
| 600 s | 90 | 545 | **+0.697** | +0.122 | +0.204 | +0.192 |
| 1200 s | 89 | 545 | +0.506 | +0.077 | +0.104 | −0.046 |
| 1800 s | 82 | 520 | **+0.170** | +0.028 | +0.081 | −0.279 |
| 2700 s | 59 | 377 | +0.384 | +0.053 | +0.115 | −0.196 |
| 3600 s | 9 | — | dropped: nine observations do not make an R² |

Persistence falls from +0.697 at 600 s to +0.170 at 1800 s, and it is the only
place where the gap with the load model stops excluding zero — at 300 s the
interval is [−6.73 · **−0.35**] and P(gain) 0.000, at 1800 s it is
[−2.05 · +0.43] and P(gain) 0.398. The two models become indistinguishable
not because load improves, but because persistence collapses.

### The bike target: the pairing the repository had never made

Cycling critical power has been measured since 2016 — **417 observations of
`cp20_w`, two days apart at the median, over three and a half years** — and
had never served as a target, because `build_perf_observations` filters on
`is_run`. Yet it is the only pairing where the discipline carrying the load is
the one whose performance is measured, with 4.6 times more observations than
running and twice as many folds.

| variant | out-of-sample R² (38 folds) | in-sample R² |
|---|---:|---:|
| persistence | **+0.523** | −2.457 |
| exponential moving average | +0.436 | −2.174 |
| **banister bike only** t42/12 | **+0.279** | +0.433 |
| banister t42/12 (total load) | +0.218 | +0.388 |
| fitness only t42 | +0.212 | +0.316 |
| predict the mean | −0.090 | 0.000 |
| banister **run** only t42/12 | **−0.956** | +0.050 |

And this time the intervals decide:

| variant | vs. | gain | 95 % CI | P(gain) |
|---|---|---:|---|---:|
| bike only | **running load** | **+0.631** | **[+0.407 · +0.742]** | **1.000** |
| bike only | **predict the mean** | **+0.339** | **[+0.055 · +0.524]** | **0.986** |
| bike only | total load | +0.078 | [−0.143 · +0.248] | 0.761 |
| bike only | persistence | −0.513 | [−2.161 · +0.137] | 0.080 |

Three things show in it, and they are the first load results in this
repository that survive their own interval.

**Matching the load's discipline to the target's is worth +0.631 of skill,
with P(gain) = 1.000.** It is also a discriminant-validity check: running load
does *not* predict cycling power (−0.956), which is the right answer. Had it
done so, one would have had to look for a leak in date alignment before
claiming a physiological transfer.

**The load model beats "predict the mean" by +0.339, interval
[+0.055 · +0.524].** On the running target it had never done so demonstrably.
There is therefore a measurable load effect in this athlete — where load and
performance speak of the same discipline.

**It is still beaten by persistence**, but no longer decisively: P(gain) goes
from 0.032 on the running target to 0.080 here, and the interval contains
zero.

### The detectability floor, and what it does to the conclusion

All these intervals are wide, which raises the question the repository had
never asked: **what effect could this protocol see?** Without that number,
"we found nothing" and "there is nothing" are indistinguishable, and the
second does not follow from the first.

`src/evaluation/sensitivity.py` adds to the **real** envelope a **synthetic**
load effect of known amplitude — `fitness − fatigue` on the load of the tested
discipline, exactly the form the model postulates — then reruns the unchanged
protocol. `δ` is expressed in envelope standard deviations.

| | vs. "predict the mean" | vs. persistence |
|---|---|---|
| **running** target (18 folds) | floor **0.50 σ** | floor **2.00 σ** |
| **bike** target (38 folds) | **effect already seen without injection** | floor **1.00 σ** |

Four readings, the last of which corrects the way this repository used to
present its own result.

On the running target, the protocol rules out any Banister-shaped effect above
**0.5 σ** relative to the mean — a falsifiable bound, where "found nothing" is
not one.

To beat persistence on the same target, an effect **twice as dispersed as the
whole envelope** would be needed. In other words: "no load model beats
persistence" is true, but this comparison has almost no power. Persistence is
near-unbeatable on a target with 0.97 autocorrelation, and the injected effect
is itself smooth — two exponential moving averages —, so persistence tracks
it. The finding holds; what it demonstrates is much thinner than it seemed.

On the bike target the floor halves, to **1.00 σ**, thanks to the 38 folds and
417 observations. And against the mean there is no floor left to measure: the
effect is detected **without any injection**, in the real data.

The floor is moreover **optimistic**: the injected signal has precisely the
shape the predictor fits, the most favourable case imaginable. A real effect
of a different shape would have to be larger still to be seen.

### What this changes

The repository's null result is not cancelled, it is **circumscribed**. It
concerned a running target set against a load that was two-thirds cycling,
measured by a protocol now known to be unable to decide. Correctly matched,
the load model shows an effect that survives its interval.

What remains is whether this effect is a fact or a quirk of this athlete: the
next section confronts it with sixty others.

```bash
python -m src.cli evaluate       # the five protocols and the four floors
```

---

## What sixty-one athletes say

The reference athlete's result rested on 38 folds from one person.
`cli fetch-athletes` brought the cache from 62 to **183 archives**; 61 athletes
pass the cohort criteria and carry the experiment.

Everything comes from the archives' `<uuid>.json` files: date, discipline,
twenty-minute critical power, TRIMP. The trade-off is declared and checked —
the load used is GoldenCheetah's `trimp_points`, correlated at **0.982** with
the repository's TRIMP for the reference athlete (0.984 on cycling), with a
median ratio of 0.938. Enough to know whether a relationship replicates; not
enough to publish a coefficient.

### Three results, and the number of subjects changes the conclusions

| comparison | athletes | median gain | Q1 · Q3 | gain > 0 | CI > 0 | CI < 0 | sign p |
|---|---:|---:|---|---:|---:|---:|---:|
| bike load / **mean** | 61 | +0.130 | [−0.058 · +0.366] | 40/61 | **23** | **5** | **0.020** |
| bike load / **persistence** | 61 | −1.153 | [−1.707 · −0.592] | 3/61 | 0 | **32** | **<0.0001** |
| bike load / **running load** | 35 | +0.481 | [+0.160 · +0.731] | 29/35 | **14** | 0 | **0.0001** |

**Discipline pairing is the most solid result in the repository.** Among the
35 athletes who do both, setting cycling power against bike load rather than
running load is worth +0.481 of median skill, with fourteen intervals
excluding zero and **none in the opposite direction**, sign test at
p = 0.0001. Running load serves here as a negative control: if it predicted
cycling power, one would have to look for a leak in date alignment before
speaking of physiological transfer.

**The load effect replicates as a population fact**: 40 of 61 athletes have a
positive gain against the mean, 23 with an interval excluding zero, sign test
p = 0.020.

**Persistence's dominance is overwhelming.** Three athletes out of 61 beat it,
thirty-two have an interval excluding it in its favour, p < 0.0001. It holds
for the population — and the detectability floor says how little power that
comparison has against a smooth effect.

### What thirteen athletes could not see

On the first thirteen athletes, the sign test against the mean gave
**p = 1.000** and I concluded that the effect did not replicate as a
population fact. On 61, it gives **p = 0.020**.

That conclusion was not wrong, it was **underpowered** — exactly what the
detectability floor built above is meant to quantify. The number of subjects
was the limiting factor, and measurably so.

Two things also appear that thirteen subjects hid. First, **five athletes
whose interval excludes zero *against* load**: for them, feeding the model
with load degrades the forecast relative to predicting the mean.

| athlete | days | powers | folds | R² load | R² mean | gain | 95 % CI |
|---|---:|---:|---:|---:|---:|---:|---|
| a49a2a94 | 500 | 275 | 12 | −0.777 | −0.044 | −0.703 | [−1.404 · −0.210] |
| 0031326c | 456 | 263 | 11 | −0.359 | +0.048 | −0.428 | [−1.150 · −0.018] |
| 2d6239bd | 617 | 178 | 18 | −0.486 | −0.047 | −0.419 | [−0.849 · −0.048] |
| 7b8034de | 648 | 1090 | 17 | −0.362 | −0.089 | −0.250 | [−0.500 · −0.042] |
| ae4b2ab7 | 1245 | 276 | 47 | −0.126 | −0.056 | −0.066 | [−0.216 · −0.012] |

Heterogeneity is therefore **bidirectional**, not a mere presence or absence
of effect. A digital twin that assumes a dose-response does not merely add
nothing for a majority: for a minority, it actively predicts worse than the
mean.

### Real heterogeneity, or a measurement defect?

| group | athletes | median gain | gain > 0 | CI width | median powers | median folds |
|---|---:|---:|---:|---:|---:|---:|
| detected | 23 | **+0.442** | 23/23 | 0.397 | 408 | 25 |
| not detected | 38 | **−0.025** | 17/38 | 0.688 | 294 | 22 |

The contrast holds at 61 subjects as it did at 13, and it is sharper. The
detected are better measured — intervals 1.7 times tighter — but **their gain
also differs**, and that is what decides: the non-detected are centred on
−0.025, with 17 positive estimates out of 38, the split one expects around
zero. They are not positive-but-drowned, they are null. The median number of
folds is moreover nearly identical between the groups (25 vs. 22), so this is
not a difference in power.

**Between-athlete heterogeneity is real.** And no variable available at
sign-up separates the two groups — see
[who responds](#who-responds-the-open-question-asked-out-of-sample).

---

## The five athletes for whom load hurts

It is the most unexpected result of the replication: for five athletes out of
sixty-one, the paired interval excludes zero **against** load — the Banister
model predicts worse than "predict the mean". An uninvestigated artefact at
this point would distort the reading of everything else, so the five cases
were opened one by one.

| athlete | gain | CI | powers | days | folds |
|---|---:|---|---:|---:|---:|
| `a49a2a94` | −0.703 | [−1.404 · −0.210] | 275 | 501 | 12 |
| `0031326c` | −0.428 | [−1.150 · −0.018] | 263 | 490 | 11 |
| `2d6239bd` | −0.419 | [−0.849 · −0.048] | 178 | 725 | 18 |
| `7b8034de` | −0.250 | [−0.500 · −0.042] | 1090 | 653 | 17 |
| `ae4b2ab7` | −0.066 | [−0.216 · −0.012] | 276 | 1662 | 47 |

### No artefact signature

Three artefact hypotheses were tested against the **base rate** of the 23
detected athletes — without a comparison group, any value looks suspicious.

**Data gap:** median spacing between observations of 1 to 3 days, maximum gap
of 17 to 127 days, against 1 day and 56 days at the median among the detected.
Nothing abnormal.

**Sensor change, as a level break.** We look for the cut that maximises the
mean difference between the two sides of the power series. Four of the five
stay below the detected group's 9th decile (1.28 σ); `ae4b2ab7` comes out at
1.54 σ, above that decile but **below the detected maximum** (1.68 σ).
Suspicious, not conclusive.

**Scale change, as the ratio of standard deviations on either side of the
cut.** `2d6239bd` comes out at 0.55, just below the detected minimum (0.59):
its power loses half its dispersion mid-series, which is the signature of a
smart-trainer change. Public summaries do not carry equipment, so this remains
an unverifiable hypothesis from these data.

### The mechanistic hypothesis I had, and its refutation

The five negatives stand out clearly on one point: **their load rises much
more steeply** — +18.6 TRIMP/year against +4.2 among the detected. Hence a
plausible hypothesis: if cumulative load trends strongly, the convolution
trends too, the model extrapolates performance upward out of sample, and
overshooting the target is worse than predicting the mean.

It is false. Measured on the 61 athletes, **the model does not overestimate**:
median bias −0.045 σ, 27 athletes overestimated out of 61 — the error is
symmetric. And bias says nothing about the score: ρ(bias, gain) = **−0.018**
on 61 subjects, a well-powered null, not a lack of power.

What governs the score is the **magnitude** of the error, not its sign:
ρ(absolute error, gain) = −0.720. The five negatives do not err in direction,
they err more — 1.045 σ of absolute error against 0.631 among the detected.
Good news for the planner: a model whose errors are symmetric does not
systematically over-promise.

### The two remaining gaps sit just below the floor

The two observed differences — load slope and absolute error — point in the
direction of the explanation, and neither is demonstrable at five subjects.
Permutation test, 20,000 draws, median of the five against that of the other
56:

| quantity | observed gap | p | gap detectable at 5 % |
|---|---:|---:|---:|
| load slope (TRIMP/year) | +14.0 | 0.074 | **+16.9** |
| absolute error (σ) | +0.249 | 0.070 | **+0.267** |
| signed bias (σ) | −0.086 | 0.761 | +0.577 |

The first two gaps are real and **smaller than what this protocol can see**.
About 20 % more amplitude was needed, or more subjects. This is exactly the
form of result the detectability floor exists to produce — and it is the third
time in this repository that a negative finding is in fact an underpowered
one.

### What can be said

**No artefact established, no mechanism established.** One case (`2d6239bd`)
carries a scale break compatible with an equipment change, another
(`ae4b2ab7`) a level break at the edge of the base rate; the other three look
like the detected athletes on everything measurable. Bidirectional
heterogeneity is therefore not an identifiable measurement defect, and the
planner is right to promise nothing to an athlete whose dose-response is not
detected — including when it is detected in the wrong direction.

---

## What pooling the athletes adds

Replaying the protocol athlete by athlete counts verdicts: twenty-three
subjects out of sixty-one show an effect, thirty-three do not, five see it
turn against them. That does not say **by how much** they differ — a sign test
does not measure a dispersion. A hierarchical model estimates it, and shares
the time constants so that they are constrained by sixty-two histories instead
of one.

`python -m src.cli pool` — **62 athletes, 1,816 envelope points** (one every
28 days), four chains of 400 + 400 iterations in **2 h 26**, **zero
divergences**, `r_hat` ≤ 1.01.

| parameter | mean | sd | 94 % HDI | ESS |
|---|---:|---:|---|---:|
| **τ₁ (fitness, days)** | **70.3** | 10.1 | **[52.0 · 88.4]** | 295 |
| **τ₂ (fatigue, days)** | **29.7** | 3.7 | **[23.3 · 37.0]** | 733 |
| μ net effect | **+0.610** | 0.047 | **[+0.521 · +0.696]** | 708 |
| **σ net effect** | **0.205** | 0.040 | **[+0.131 · +0.278]** | 609 |
| μ fatigue gain | −0.204 | 0.138 | [−0.453 · +0.068] | 535 |
| σ fatigue gain | 0.586 | 0.120 | [+0.374 · +0.807] | 795 |
| μ trend | −0.024 | 0.049 | [−0.112 · +0.070] | 764 |
| **σ trend** | **0.334** | 0.040 | **[+0.261 · +0.409]** | 694 |

Gains are expressed in envelope standard deviations per standard deviation of
`fitness − fatigue`: each target is standardised on its own history, otherwise
pooling a gain would mean mixing watts rather than effects.

### Heterogeneity is now established — correcting a result published here

`σ` of the net effect is **0.205, HDI [+0.131 · +0.278]**. The interval
**excludes zero**.

At fourteen athletes, the same parameter came out at 0.212 **[0.000 · 0.393]**
and this page concluded that heterogeneity was not established. The mean did
not move — 0.212 then 0.205 — **only the interval tightened**, from 0.39 wide
to 0.15. The earlier conclusion was therefore not wrong, it was
**underpowered**: subjects were missing, not a different model. It is the
exact scenario the detectability floor exists to anticipate, and it deserves
to be written plainly rather than corrected silently.

Athletes therefore really differ, by about a third of the mean effect in
standard deviation. A population effect of +0.61 with a dispersion of 0.21
puts the vast majority of subjects on the right side of zero — which the fit
shows — but leaves one athlete in twenty close to indifference.

### The time constants are identified, and they contradict the literature

This was the main goal of pooling. On a single athlete, the repository's
posterior tightens τ₁ by only **41 %** relative to its prior: the data do not
identify it. Shared across sixty-two histories, it comes out at **70.3 days
[52.0 · 88.4]** and τ₂ at **29.7 [23.3 · 37.0]**.

These are informative intervals — and that is where the good news stops. The
literature puts τ₁ around 42 days and τ₂ around 10–15. τ₂'s prior is a
truncated normal with mean 12 and sd 5: the posterior lands **3.5 prior
standard deviations above**, pushing against it. At fourteen athletes the
estimate was compatible with the literature (51.7 and 19.0); at sixty-two,
both constants drifted upward while tightening. A parameter that fights its
prior and moves away from it as data are added does not read as a
physiological measurement.

The most likely explanation lay in the target, not in the athletes: **the
envelope is a 42-day rolling maximum, read every 28 days.** It is a heavily
smoothed series by construction, and a smoothed target favours slow
convolution kernels. This hypothesis was put to the test; the next section
gives it a number.

The constants are therefore **identified without being validated** — two
different things, and only the first was the stated goal.

### What the smoothing window explains about these constants

The test is direct: redo the fit with an envelope window of **21 days**
instead of 42, everything else unchanged, and see whether the time constants
follow the window. If they do, they measure the protocol rather than the
physiology, and the most quotable row of the previous table falls.

A trap lay on the way, and it alone would have made the result unreadable:
`min_periods = 3` applies to the window, so **shortening it drops days for
athletes with sparse observations**. Two athletes and 218 points leave the
21-day fit. A shared τ then has two reasons to move — the window, and the
sample composition — and no way to tell them apart by comparing two columns.
Hence a **third, control fit**: window 42, cohort minus the same two athletes.

Three fits, same sampler settings (4 chains, 400 + 400, `target_accept` 0.9,
seed 0), **0 divergences and r_hat ≤ 1.01 everywhere**.

| | published measurement | composition control | shortened window |
|---|---|---|---|
| envelope window | 42 d | 42 d | **21 d** |
| athletes · points | 62 · 1816 | 60 · 1791 | 60 · 1598 |
| lag-1 autocorrelation of the target | +0.533 | +0.539 | **+0.343** |
| median step of the target | 0.442 σ | 0.442 σ | **0.628 σ** |
| **τ₁** | 70.3 [52.0 · 88.4] | 71.7 [53.6 · 91.5] | **61.4 [43.6 · 78.4]** |
| **τ₂** | 29.7 [23.3 · 37.0] | 29.7 [22.8 · 37.0] | **24.1 [18.7 · 30.3]** |
| μ net | +0.610 | +0.611 | +0.532 |
| σ net | 0.205 [+0.131 · +0.278] | 0.202 [+0.134 · +0.282] | 0.177 [+0.106 · +0.262] |
| μ fatigue | −0.204 | −0.208 | −0.244 |
| σ fatigue | 0.586 | 0.580 | 0.524 |
| μ trend | −0.024 | −0.019 | +0.006 |
| σ trend | 0.334 | 0.332 | 0.306 |

**The control clears composition.** Removing the two athletes at constant
window moves nothing: τ₁ goes from 70.3 to 71.7, τ₂ from 29.7 to 29.7, μ net
from +0.610 to +0.611. The two athletes accounted for only 25 of the 218
missing points — the other 193 are due to the window. All the remaining gap is
therefore attributable to the window, and to it alone.

**Both starting hypotheses are false.** "These constants are physiological"
predicted no movement: τ₁ loses 14.4 % and τ₂ 18.9 %, and a physiological
constant cannot feel a smoothing setting. "These constants are the window"
predicted that they follow it: halving the window should have brought τ₂
towards 15, it sits at 24.1. In elasticity — the relative change of the
constant over that of the window, which would be 0 for pure physiology and 1
for pure window-learning — **τ₁ comes out at 0.22 and τ₂ at 0.30**. About a
quarter of these constants is protocol.

**And nothing points to a reaction specific to the time constants.** Every
parameter drops by 8 to 19 %: τ₁ −14.4 %, τ₂ −18.9 %, μ net −12.9 %, σ net
−12.4 %, σ fatigue −9.7 %, σ trend −7.8 %. This is not the signature of a
model re-learning a window, it is that of a target whose regularity changed:
lag-1 autocorrelation drops from +0.539 to +0.343 and the median step goes
from 0.442 to 0.628 standard deviations. The target being standardised per
athlete, this is a matter of regularity rather than amplitude — and a less
regular target leaves less variance explainable by anything smooth, so all
coefficients contract together.

What the protocol change **does not take away**, and which matters more for
use:

- **σ net excludes zero in all three fits** (0.205 · 0.202 · 0.177).
  Between-athlete heterogeneity is not a window artefact.
- **The ranking of athletes holds**: individual gains correlate at **+0.839**
  (Spearman +0.792) between the control and the shortened window.
- **No negative gain, in any of the three.** Shrinkage keeps forbidding what
  the out-of-sample protocol shows for five athletes, and the window is not
  responsible.

Verdict: the time constants remain **identified without being validated**,
and this is no longer a conjecture. About a quarter of their value is
protocol; the remaining three quarters are not explained by this experiment,
and nothing allows citing them as a physiological measurement. The gap to the
literature (70.3 vs. 42, 29.7 vs. 12) does not close by fixing the window.

Both experiments run, and the target characterisation reproduces in minutes
rather than hours:

```bash
python -m src.cli pool --fenetre 21                     # ~14 h 40
python -m src.cli pool --exclure 6d03803e,9e46e144      # ~2 h 30
python -m src.cli pool --decrire --fenetre 21           # the target only
```

Artefacts from an unpublished setting carry a suffix — `_f21`, `_n60` —
because an experiment that questions a measurement must not be able to
overwrite it.

### Load does not inherit a shared trend

A first fit, without a trend term, let τ₁ escape to **127 days** — 4.6
standard deviations beyond its prior. At that constant, cumulative load *is* a
trend, and the gain measured one while believing it measured the other.

With a competing per-athlete trend, the verdict holds at sixty-two subjects:
**μ trend = −0.024 [−0.112 · +0.070]**, centred on zero and now tight, while
**σ trend = 0.334 [+0.261 · +0.409]** excludes zero. Athletes have real
trends, in both directions, that cancel out on average — and the load term
does not capture a common drift. The check passes, and passes better than at
fourteen.

### And here is the signature of overfitting, quantified

| | athletes with an effect excluding zero |
|---|---|
| hierarchical fit (**in** sample) | **61 / 62** |
| horizon protocol (**out of** sample) | **23 / 61** |

This is not a contradiction, it is the repository's founding observation put
into numbers: **fitting is not predicting.** The same model, on the same
athletes, explains past performance for all but one and forecasts what comes
next for only a little over a third. A favourable Bayesian posterior is
therefore worth nothing as proof of predictive ability — and that is exactly
why this repository never publishes a fit without its out-of-sample
counterpart.

The gap widened going from fourteen to sixty-two athletes (12/14 against 4/13
before). Pooling shrinks gains towards the population mean: the larger the
cohort, the more a poorly measured athlete is pulled towards +0.61, and the
more its in-sample interval excludes zero without anything having been
predicted for it.

### The two protocols agree in rank, not in sign

Both measurements cover the same 61 athletes and can therefore be crossed. The
correlation between the hierarchical gain and the out-of-sample `skill` is
**+0.406**: the two protocols rank athletes in the same order.

| out-of-sample group | n | mean hierarchical gain |
|---|---:|---:|
| load helps (CI > 0) | 23 | **+0.669** |
| not detected | 33 | +0.601 |
| load degrades (CI < 0) | 5 | **+0.429** |

The only athlete out of sixty-two whose fitted gain does not exclude zero
(`ae4b2ab7`, +0.281 [−0.021 · +0.558]) is also one of the five for whom load
degrades the forecast. The agreement is real.

But **the hierarchical fit produces no negative gain**, not even for those
five. Shrinkage does that, and it is the declared price of pooling: it
borrows strength from the other athletes, so it refuses to believe an isolated
subject who goes against the population. If these five cases are real, this
model is structurally unable to see them — and the out-of-sample protocol
alone flags them.

### What measurement power explains, and what it does not

The project's open question — **what distinguishes a responder?** — gets its
first, partial answer here.

| group | days of history (median) | folds | CI width (median) |
|---|---:|---:|---:|
| load helps | 851 | 25 | **0.397** |
| not detected | 717 | 22 | 0.678 |
| load degrades | 617 | 17 | 0.801 |

Responders are the best-measured athletes: the rank correlation between
`skill` and history length is **+0.28**, and the non-detected interval is
**twice as wide** as the detected one. Part of the non-detection is therefore
a lack of power, not an absence of effect.

Only part. The same athlete `ae4b2ab7` has **1,245 days and 47 folds** — the
best measured of the negative group, and the tightest interval of the five
(`[−0.216 · −0.012]`). Its non-response is not explained by noise. And history
is in any case not a variable available at sign-up: it describes what was
observed, not who the athlete is.

**Nothing known on day one separates the groups yet.** The question remains
open — but it has gone from unassailable to measurable.

### Two protocol caveats

**62 athletes here, 61 in the horizon protocol.** The inclusion criterion is
not the same: this model needs envelope points, not validation folds. Totals
do not compare row by row; the 61 common athletes do cross, and that crossing
is what is published above.

**One point every 28 days.** The envelope is a 42-day rolling maximum, hence
highly redundant at a daily step, and the convolution kernel costs
`n_points × n_days` per athlete. It is the same quantity, read less often —
but it is a cost choice, it is declared, and the section on time constants
explains why it may not be neutral.

---

## The planner, made correct under measured uncertainty

Up to here the repository measured. The planner, for its part, was **imported
by nothing** — no test, no dashboard, no command — and its objective said:

```python
# optimizer.py, before
obj = sum(x[j][t] * spec_vals[t] for j in range(7) for t in range(n_types))
model.Maximize(obj)
```

Maximising specificity under a volume ceiling amounts to prescribing "train as
much as the safety valve allows". That is defensible if the athlete's
dose-response is known. The cohort says it is not known in advance: it is
detected in 23 athletes out of 61, harmful in 5, and no variable available at
sign-up separates them.

Under an unknown dose-response, maximising a proxy is the wrong side of the
uncertainty. The cost of unnecessary load is real — injury, overreaching,
time — while its benefit is, for most athletes, not measurable.

### The objective, reversed

`optimize_week` now looks for the **minimum** load that delivers the requested
stimulus. For the same constraints, the same stimulus, and a past week at
250 TRIMP:

| objective | TRIMP | sessions | specificity |
|---|---:|---:|---:|
| `max_specificity` (historical) | 271 | 6 | 0.43 |
| **`min_load`** (default) | **200** | **3** | 0.33 |

26 % less load and half the sessions, for the two key sessions required in
both cases. The old objective remains available through
`--objective max_specificity`, so that both are comparable rather than one
silently replaced.

The gap depends on the regime, and it must be said: for the repository's
athlete, whose last known week weighs 446 TRIMP, the retention floor rises to
357 and both objectives converge (360 vs. 365). The difference plays out when
there is room between floor and ceiling — that is, precisely when there is a
choice to make.

A floor prevents the opposite problem: minimising without a lower bound would
give two key sessions and five rest days, which is a minimum in the literal
sense and a regression in the physiological sense. It is
`min_weekly_retention_pct` of the past week — in `params.yaml`, not in the
code.

### Three safety rules that existed without being wired

**The acute/chronic bound.** `acute_chronic_ratio` lived in `load_model.py`,
its docstring presented it as "the only safety rule that can actually fire,
unlike rMSSD and DFA, it needs only load, which always exists" — and it was
called nowhere. `acwr_upper_bound: 1.5` was configured and never read. The
coming week's load, brought to a daily figure, is now capped at 1.5 times the
chronic load.

When this ceiling falls below the retention floor — an athlete whose chronic
base has collapsed — the floor is **lowered to the ceiling**, not dropped.
Dropping it would return a zero-TRIMP plan: the literal minimum, and useless
advice. On a base of 10 TRIMP/day, the plan goes from 400 to 105 TRIMP and
says so.

**The monotony check.** Foster's monotony is a mean/sd ratio, non-linear in
the decision variables, hence impossible to state directly in a linear
solver. Its practical content can be: a light day in every rolling window, and
repetition bounds per session type. Without the latter the optimiser stacked
the cheapest type that satisfied its constraints — it prescribed **four
`CALIBRATION` sessions**, which is a test and not training. The bounds live in
`params.yaml`.

**Declared unavailability.** `avoid_days` — "I can't train that day",
extracted from natural-language context — was posted as an `AddHint`, i.e. a
suggestion the solver may ignore. A declared unavailability thus ended up
scheduled. It is a constraint, and it takes precedence over the prescribed
stimulus: the athlete knows their schedule. Incidentally, an `AddHint` made
infeasible by the constraints makes CP-SAT write pages of warnings to stdout,
the visible symptom of the same defect.

**The HRV threshold.** `safety_valve` read `planner.rmssd_vma_threshold`, a
key that does not exist in `params.yaml`. The value configured under
`planner.hrv.rmssd_threshold_ms` was therefore never read, and a hard-coded
default always won — while the file's comment presented it as "the single
source of truth for this threshold". The two values were equal by chance,
which made the defect invisible.

### The plan says what it does not optimise

`dose_response.py` measures the load → performance relationship **for the
athlete at hand**, through the already published protocol — same envelope,
same split, same paired bootstrap. Three verdicts, not two:

| verdict | what it means | right to promise |
|---|---|---|
| `detectee` | paired interval excluding zero | speak of an effect, yes |
| `non detectee` | enough data, nothing to see | no |
| `indeterminable` | not enough data to look | no |

Conflating the last two would present an absence of measurement as an absence
of effect, which is exactly the mistake this repository works not to make. And
none of the three authorises maximising a predicted performance: the verdict
does not enter the optimisation, it enters the report.

```
$ python -m src.cli plan

=== Explication du Plan (XAI) ===

Ce plan cherche la charge la PLUS FAIBLE qui délivre le stimulus demandé,
sous les contraintes de sécurité.
Dose-reponse detectee chez cet athlete : competence +0.339 contre « moyenne »,
IC 95 % [+0.055 · +0.524], 38 plis.
Un effet de charge est donc démontré chez cet athlète — ce plan ne le maximise pas
pour autant, il délivre un stimulus au moindre coût.
```

(The CLI speaks French: "this plan looks for the LOWEST load that delivers the
requested stimulus… a load effect is demonstrated for this athlete — this plan
does not maximise it, it delivers a stimulus at least cost.")

The verdict reproduces the cohort's to the digit, because it goes through the
same code. If the two diverged, one of them would be lying.

### A degraded plan no longer reads as an optimal plan

The old code silently fell back on the heuristic as soon as a constraint made
the problem infeasible. `WeeklyPlan` now carries `notes`, and the report
publishes them: floor lowered, stimulus dropped, acute/chronic bound inactive
for lack of history. A silently inactive bound is worse than an absent one.

```bash
python -m src.cli plan                              # minimum sufficient load
python -m src.cli plan --objective max_specificity  # historical behaviour
```

---

## The planner's rules, confronted with data

The planner imposed three bounds — ACWR below 1.5, a retention floor, at most
three consecutive training days — that came from the literature and from
practice. That is a reason to write them, not a measurement. None had ever met
data, while the repository demands out-of-sample validation of everything
else.

`python -m src.cli validate-plan` — **66 athletes, 6,768 calendar weeks.** For
each week: its load, its end-of-week ACWR, its longest run of consecutive
days, and its outcome — the change in best effort between `[end−42 ; end−7]`
and `[end+7 ; end+42]`, scaled by the standard deviation of that athlete's
power. The comply / violate comparison is made **within each athlete**, then
aggregated by a sign test, like the cohort.

### The verdict

| bound | regime | statistic | athletes | gap | in favour | p | verdict |
|---|---|---|---:|---:|---:|---:|---|
| ACWR ≤ 1.5 | all weeks | best | 55 | −0.333 | 7/55 | <0.0001 | contradicted |
| ACWR ≤ 1.5 | all weeks | median | 55 | −0.104 | 18/55 | 0.0145 | contradicted |
| **ACWR ≤ 1.5** | **reference ≥ 15 obs** | **best** | 14 | −0.014 | 8/14 | 0.791 | **undeterminable** |
| **ACWR ≤ 1.5** | **reference ≥ 15 obs** | **median** | 14 | −0.042 | 4/14 | 0.180 | **undeterminable** |
| **ACWR ≥ 0.8** | **reference ≥ 15 obs** | **best** | 55 | **+0.085** | 40/55 | **0.0010** | **supported** |
| **ACWR ≥ 0.8** | **reference ≥ 15 obs** | **median** | 55 | **+0.051** | 37/55 | **0.0145** | **supported** |
| ≤ 3 consecutive days | reference ≥ 15 obs | median | 37 | −0.013 | 15/37 | 0.324 | undeterminable |

**The retention floor is supported.** Weeks whose ACWR stays above 0.8 are
followed by a better effort: +0.051 σ on the statistic insensitive to the
number of draws, 37 athletes out of 55 in favour, p = 0.0145. It survives both
controls and both statistics, and it is detected **without injection**. "Do
not prescribe detraining" is therefore a measured rule, not a convention.

**The ACWR ceiling is neither supported nor refuted**, with a detectability
floor of **0.20 σ**: its real effect, if any, is smaller than what 14
comparable athletes can show.

**The consecutive-days ceiling does not conclude** (p = 0.324). The breakdown
is suggestive though: the outcome rises from 0 to 3 consecutive days
(−0.025 → +0.091 σ) then falls from 4 onwards (−0.005). The threshold of 3 is
thus placed exactly at the peak — but the "≤ 3" constraint lumps together 0,
1, 2 and 3 days, which dilutes the gain it should capture. It is a defect in
how the constraint is stated, not in the threshold.

### How I nearly published the refutation of a safety rule

The first two rows of the table are to be read as a warning. The naive
protocol says that **respecting the ACWR upper bound costs 0.333 σ**, with 7
athletes out of 55 in favour and p < 0.0001 — a massive, clean, and false
result. Three confounders were stacked in it, and each had to be found before
being fixed.

**The envelope cannot serve as an outcome.** First attempt: change in the
envelope over 28 days. The envelope is a 42-day rolling maximum, so it moves
only in steps, and its median change was **exactly zero for all sixty-six
athletes**. The sign test was running on noise asymmetries.

**The week being judged contaminated both ends of the difference.** A rolling
maximum looking 42 days back includes the week being evaluated: that week's
hard effort *is* a performance observation. Hence a 7-day buffer on either
side, which excludes the week from both windows.

**A maximum over more draws is larger.** High-ACWR weeks are followed by more
rides: in the bracket above 2.0, the outcome window counts **5.2 more
observations** than the reference one. The maximum rises without any
performance having improved. Each bound is therefore also judged on the
**median** effort of the window, insensitive to the number of draws — and this
control alone divides the effect by three, from −0.333 to −0.104.

**And the last confounder is not controlled, it is excluded.** A high ACWR
often means chronic load was low: the athlete was coming back from a break.
The reference window is then itself depressed, and the measured "progress" is
a return to normal. The numbers say so unambiguously — the bracket above 2.0
has only **8.1 observations in the reference window against 13.3 after**.
Requiring 15 observations before the judged week, hence regular training, the
gap falls to **−0.042 and the test no longer concludes**. The per-bracket
gradient flattens over the whole ACWR range: −0.01 σ below 0.8, +0.10 above
2.0.

Publishing the first version would have meant dismantling a safety bound on a
sampling artefact compounded by selection bias. The price of the control is
visible: from 55 comparable athletes down to 14, and the floor rises to
0.20 σ. It is the right price.

### What this protocol does not do

It does not reconstruct **session types** — public summaries do not carry
them — so per-type ceilings are not testable here. It establishes **no
causality**: it compares weeks of the same athlete, not randomised arms. And
the planner's retention floor is 80 % of a reference week's load, a quantity
that does not exist in an observed history; `ACWR ≥ 0.8` is its closest
analogue, and that is a declared approximation, not the rule itself.

---

## Who responds? The open question, asked out of sample

The load effect exists in 23 athletes out of 61, is undetectable in 33 and
degrades the forecast in 5. It is the most useful result in the repository and
the least actionable: knowing that one athlete in three responds does not say
**which one**, and a planner needs the latter.

### Why the best correlation is not an answer

Searching among twelve predictors for the one most correlated with the gain,
then publishing it, means testing twelve hypotheses and declaring one. The
repository demands out-of-sample everywhere else. The question therefore
becomes: **leaving one athlete out, do the other sixty tell whether they
respond?**

`python -m src.cli responders` — nearest class centroid on standardised
predictors, one athlete left out at a time, **centring and centroids
recomputed within each fold**. Computing them once on the whole sample would
leak the target through preprocessing, which is the quietest way to
manufacture out-of-sample performance.

| predictor family | predictors | athletes | accuracy | majority class | p | floor |
|---|---:|---:|---:|---:|---:|---:|
| known on day one | 2 | 55 | 52.7 % | 60.0 % | 0.892 | **0.05 σ** |
| derived from history | 11 | 61 | 55.7 % | 62.3 % | 0.882 | not reached |
| all | 13 | 55 | 47.3 % | 60.0 % | 0.980 | 0.05 σ |

**Nothing separates the responders yet.** All three families do worse than
"always predict the majority class". The full family even does clearly worse
than the two-predictor one — the classic signature of too many variables for
too few subjects.

### The two nulls are not equal

This is the point of the floor. For the **known on day one** family, a shift
of 0.05 standard deviation in the responders' predictors is enough to be
detected: the test is sensitive, and its null is **well powered**. Nothing
known about an athlete before their first session distinguishes them — a real
answer to the project's open question, and a negative one.

For the **derived from history** family, the floor is not reached even at
0.40 standard deviation: eleven predictors on 61 subjects dilute the signal in
the Euclidean distance. "History does not predict response" is therefore
**not established** — this test could not see it.

### Correlations, with the threshold the sample can distinguish

| predictor | ρ | permuted p | detectable ρ |
|---|---:|---:|---:|
| `jours` (history span) | **+0.236** | 0.064 | 0.250 |
| `n_observations` | +0.179 | 0.166 | 0.252 |
| `age` | +0.132 | 0.333 | 0.265 |
| `jours_actifs_pct` | +0.113 | 0.385 | 0.249 |
| `cv_charge_hebdo_pct` | −0.165 | 0.205 | 0.252 |
| `pente_charge_an` | −0.129 | 0.322 | 0.252 |

None crosses its own threshold, and the Bonferroni threshold for twelve tested
predictors is at **p < 0.0042**. History span remains the best candidate —
and it is not a variable known at sign-up, so even if true it would not solve
the cold start.

### Two defects the result revealed

**A constant predictor came out "significant".** `sexe_feminin` is 0 for all
sixty-one athletes in the cache: the GoldenCheetah OpenData dataset, as this
cache sampled it, contains only men. A constant column has no **defined**
correlation, the permuted p of a NaN came out at 0.0000, and this degenerate
predictor was **the only one declared retained by the first run**. It is now
dropped — and reported as dropped, because a table that does not say what it
removed reads as if everything had been tested.

**Six athletes had a negative age.** They declare a birth year after the
dataset's publication, hence ages of **−58 to −42 years**. Their presence
pushed the age correlation to **+0.230**, the second in the table; once
removed, it falls to **+0.132**. An explicit plausibility bound now removes
them, and the athlete count goes from 61 to 55 on the families that use age —
which the table shows.

### What remains open, and what no longer is

**Closed:** nothing known about an athlete on day one predicts whether they
will respond to load. The test is sensitive and the null holds.

**Open:** history might predict without these 61 athletes being able to show
it. The way forward is not more predictors — it already has too many for the
sample — but more subjects, or a predictor stated in advance rather than
searched among twelve.
