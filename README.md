# Pearls AQI Predictor

A three-day US Air Quality Index forecast for Karachi, built end to end on a
fully serverless, zero-cost stack. GitHub Actions is the only compute,
Hopsworks holds the features and the models, and Streamlit Community Cloud
serves the dashboard. There is no machine to keep running and nothing to pay
for.

**Live dashboard:** [aqi_predictor_app](https://aqi-predictor-01.streamlit.app)
**Repository:** [aqi_predictor_github_repo](https://github.com/haris-hk/AQI-Predictor.git)

---

## Contents

1. [What this is](#1-what-this-is)
2. [Architecture](#2-architecture)
3. [Data sources](#3-data-sources)
4. [Repository layout](#4-repository-layout)
5. [How the AQI is computed](#5-how-the-aqi-is-computed)
6. [Feature engineering](#6-feature-engineering)
7. [The model ladder](#7-the-model-ladder)
8. [Evaluation protocol](#8-evaluation-protocol)
9. [The promotion rule](#9-the-promotion-rule)
10. [Serving and the model bundle](#10-serving-and-the-model-bundle)
11. [Explainability](#11-explainability)
12. [Alerts](#12-alerts)
13. [The dashboard](#13-the-dashboard)
14. [Automation and scheduling](#14-automation-and-scheduling)
15. [How state survives ephemeral runners](#15-how-state-survives-ephemeral-runners)
16. [Getting started](#16-getting-started)
17. [Local development](#17-local-development)
18. [Command reference](#18-command-reference)
19. [Configuration reference](#19-configuration-reference)
20. [Testing](#20-testing)
21. [Operations and troubleshooting](#21-operations-and-troubleshooting)
22. [Known limitations](#22-known-limitations)
23. [Project status](#23-project-status)
24. [Documentation index](#24-documentation-index)
25. [Licence](#25-licence)

---

## 1. What this is

Karachi has some of the worst air quality of any large city, and the health
guidance that matters (should a child play outside on Thursday, should the
windows stay shut) needs a forecast rather than a current reading. Every
free air-quality app shows what the air is like right now. This project
predicts what it will be like in one, two and three days, and it does so
with an evaluation protocol strict enough that the numbers can be trusted.

Concretely, the system:

- pulls hourly pollutant and weather data for Karachi from Open-Meteo, with
  no API key required
- recomputes the US EPA AQI properly from raw concentrations rather than
  trusting a convenience field
- engineers roughly 170 daily features across eight families, all of them
  causal
- trains six independent regressors, one for each combination of two targets
  (daily mean AQI, daily max AQI) and three horizons (1, 2, 3 days)
- selects the winning model family per pair from a ladder that runs from
  naive baselines through linear models, tree ensembles, SARIMAX and neural
  networks
- refuses to deploy a model that cannot beat persistence on a holdout it has
  never seen
- publishes a forecast with uncertainty bands, a model leaderboard, SHAP
  explanations and hazard alerts to a public dashboard
- does all of the above on a schedule, unattended, for free

### Design principles

Four commitments shape almost every file in the repository.

**Direct multi-horizon targets, not recursion.** Six independent regressors,
one per (target, horizon) pair. Predicting hour by hour and feeding
predictions back seventy-two times compounds error into noise by day three.

**Leakage is designed out, not checked for.** There is no random split
anywhere in the codebase. Validation is expanding-window walk-forward with a
mandatory gap of at least the forecast horizon between the end of training
and the start of validation, plus a 90-day chronological holdout that the
model search never touches. `tests/test_features.py` asserts causality
structurally by perturbing future values and checking that past features do
not move.

**Persistence is the acceptance bar.** AQI is highly autocorrelated, so an
R-squared of 0.9 proves very little on its own. Every model is scored against
persistence, seasonal-naive, climatology and drifted persistence, and the
skill score against persistence is reported next to every headline metric.

**Every dependency degrades rather than fails.** Hopsworks unreachable falls
back to a Parquet mirror. `statsmodels` missing skips the statistical tier.
`torch` missing skips the deep tier. `shap` missing falls back to permutation
importance. No alert channel configured still records the alert. A pipeline
whose every stage is a hard dependency on a third party is a pipeline that
stops.

---

## 2. Architecture

```
Open-Meteo  (air quality + ERA5 weather + historical forecast, no API key)
        │  hourly pollutants, weather, and a 7-day forecast
        ▼
GitHub Actions  ── hourly features · daily training · weekly full ladder
        │
        ▼
Hopsworks  ── feature groups · feature view · model registry
        │              (Parquet mirror on an orphan `data` branch as fallback)
        ▼
Streamlit Community Cloud  ── forecast · EDA · leaderboard · SHAP · alerts
```

| Layer | Choice | Why |
|---|---|---|
| Data | Open-Meteo Air Quality API and ERA5 archive | The only free source offering both a multi-year archive and a 7-day forecast, and it needs no API key |
| Compute | GitHub Actions | Scheduled compute with no machine to operate; unlimited minutes on public repositories |
| Feature store and registry | Hopsworks (free tier) | Managed feature groups, a shared feature view, and a versioned model registry |
| Durable state | Orphan `data` git branch | Parquet mirror and model artefacts survive ephemeral runners with no external service |
| Serving | Streamlit Community Cloud | Free public URL, no web server to run |
| Alerts | ntfy, Slack webhook, SMTP | All optional, all free, all pluggable |

**On the word "serverless".** Apache Airflow was the obvious alternative for
orchestration and was rejected: self-hosted Airflow needs a machine that stays
up, which contradicts the requirement and adds cost. That trade-off is stated
rather than glossed over. Airflow offers far better orchestration semantics
(backfill management, retries, dependency graphs), and those are given up in
exchange for genuinely zero infrastructure.

---

## 3. Data sources

### Open-Meteo (primary, no API key)

Four endpoints are used, each for a specific reason.

| Endpoint | Purpose |
|---|---|
| `air-quality` | Hourly pollutant concentrations (PM2.5, PM10, O3, NO2, SO2, CO, dust, aerosol optical depth, UV index) plus the API's own `us_aqi` field, for both archive and forecast windows |
| `archive` (ERA5) | Observed hourly weather back to 1940 |
| `forecast` | Weather forecast out to 16 days, used to build the forward-looking features at serving time |
| `historical-forecast` | Past forecast runs, that is, weather as it was predicted at the time. This is what removes the train/serve skew described below |

The distinction between the last two endpoints matters. At prediction time
the model only ever sees a weather forecast, so the training features should
contain forecast-quality weather too. If the historical forecast endpoint
covers more than 80 percent of the backfill range, the forward-looking
features are built from it. Otherwise the backfill falls back to observed
weather and records `future_weather_source: observed_weather_fallback` in
`data/backfill_summary.json` and in the generated data card, so the report can
state which regime produced a given model.

The client (`src/clients/open_meteo.py`) adds exponential backoff, explicit
handling of HTTP 429, and an on-disk response cache keyed by URL plus
parameters. Only immutable requests (those with an explicit `start_date`) are
cached, because a `past_days` or `forecast_days` request returns different
data on every call. The cache exists for the backfill: a multi-year pull is
split into dozens of requests and a failure halfway through must not restart
from zero.

### AQICN / WAQI (optional cross-check)

Open-Meteo serves CAMS, which is a **model reanalysis**, not readings from
physical instruments. AQICN serves readings from real monitoring stations.
The two will disagree. `src/clients/aqicn.py` fetches the nearest station's
current reading when `AQICN_TOKEN` is set, the hourly pipeline records the
delta against the modelled value, and the data card documents the
disagreement rather than quietly assuming the modelled values are ground
truth. With no token configured the cross-check is skipped and nothing else
changes.

### Coverage probe

The global CAMS archive and the European reanalysis begin on different dates,
and every downstream sizing decision depends on the real earliest date rather
than an assumption. `probe_coverage()` binary-searches for the true earliest
available air-quality date at Karachi's coordinates, verifies that the weather
archive and historical forecast endpoints respond, and writes the result to
`data/coverage.json`. The backfill reads that file, not the hardcoded fallback
in `src/config.py`.

---

## 4. Repository layout

```
pearls-aqi-predictor/
├── src/
│   ├── config.py                  Single source of truth: city, horizons, targets,
│   │                              CV settings, AQI bands, thresholds, store names
│   ├── clients/
│   │   ├── open_meteo.py          Four endpoints, retry, disk cache, chunking, probe
│   │   └── aqicn.py               Ground-station cross-check (optional)
│   ├── features/
│   │   ├── aqi.py                 EPA AQI: breakpoints, unit conversion, truncation,
│   │   │                          averaging windows, dominant pollutant
│   │   ├── build.py               Hourly join, daily aggregation, all feature families,
│   │   │                          future-weather features, target construction
│   │   └── schema.py              The feature contract: sanitise, validate, drift check
│   ├── store/
│   │   ├── hopsworks_store.py     Feature groups, feature view, model registry
│   │   └── local_store.py         Parquet mirror, experiment log, JSON state
│   ├── models/
│   │   ├── baselines.py           Persistence, seasonal-naive, climatology, drifted
│   │   ├── sklearn_models.py      Ridge, ElasticNet, RF, ExtraTrees, HistGB, quantile GB
│   │   ├── statistical.py         SARIMAX with a production-style rolling update
│   │   ├── deep.py                MLP and LSTM in PyTorch, with early stopping
│   │   ├── evaluate.py            Walk-forward splits, leakage guard, metrics, skill score
│   │   └── bundle.py              The deployed artefact: six models plus provenance
│   ├── explain/
│   │   └── shap_explain.py        Precomputed global and per-day SHAP, with a fallback
│   └── pipelines/
│       ├── backfill.py            Manual, chunked, resumable history pull; writes the data card
│       ├── features.py            Hourly: fetch, engineer, upsert, cross-check
│       ├── train.py               Daily: evaluate the ladder, fit, score, promote or reject
│       ├── predict.py             Hourly: forecast and grade past forecasts
│       ├── explain.py             Refresh SHAP without retraining
│       └── alerts.py              Deterioration-based alerts with de-duplication
├── app/
│   ├── streamlit_app.py           Page 1: the forecast
│   ├── common.py                  Cached loaders, band cards, freshness notes
│   ├── requirements.txt           The file Community Cloud actually reads
│   └── pages/
│       ├── 1_History_and_EDA.py   Page 2: history and exploratory analysis
│       ├── 2_Model_Performance.py Page 3: leaderboard and error diagnostics
│       ├── 3_Explainability.py    Page 4: what the model looks at
│       └── 4_Alerts.py            Page 5: alert configuration and history
├── .github/workflows/
│   ├── feature_pipeline.yml       Hourly: features, forecast, alerts
│   ├── training_pipeline.yml      Daily: quick ladder, retrain, promote
│   ├── full_ladder.yml            Weekly: full ladder including SARIMAX and torch
│   ├── backfill.yml               Manual: probe and historical backfill
│   └── ci.yml                     Lint and the offline test suite on every push
├── tests/
│   ├── synthetic.py               Karachi-like generator: seasonality, dispersion, dust
│   ├── conftest.py                Session fixture: 760 synthetic days
│   ├── test_aqi.py                EPA AQI against worked examples
│   ├── test_features.py           Causality, lags, rolls, targets
│   ├── test_evaluation.py         Splits, gaps, leakage assertions, metrics
│   ├── test_bundle_and_clients.py Bundle round-trip, alignment, bands, client parsing
│   └── test_alerts.py             Severity, deterioration logic, de-duplication
├── notebooks/01_eda.ipynb         Exploratory analysis
├── docs/
│   ├── PROJECT_PLAN.md            The full plan and every argued decision
│   ├── DATA_CARD.md               Generated by the backfill from what it actually saw
│   ├── MODEL_CARD.md              Intended use, scope, evaluation, failure modes
│   └── REPORT.md                  The project report
├── scripts/data_branch.sh         Restore and save state on the orphan `data` branch
├── requirements.txt               Core pipeline
├── requirements-app.txt           Superseded; points at app/requirements.txt
├── requirements-deep.txt          Torch, weekly job only
└── .env.example                   Local development template
```

---

## 5. How the AQI is computed

Open-Meteo returns a convenience `us_aqi` field, but the EPA index is not an
hourly quantity, so that field is used only as a cross-check in the EDA. The
real index is defined over pollutant-specific averaging windows, and
`src/features/aqi.py` implements it.

| Pollutant | Averaging window | Breakpoint units | Truncation |
|---|---|---|---|
| PM2.5 | 24 hours | µg/m³ | 1 decimal |
| PM10 | 24 hours | µg/m³ | 0 decimals |
| Ozone | 8 hours | ppm | 3 decimals |
| Carbon monoxide | 8 hours | ppm | 1 decimal |
| Nitrogen dioxide | 1 hour | ppb | 0 decimals |
| Sulphur dioxide | 1 hour | ppb | 0 decimals |

The overall index is the **maximum** sub-index across pollutants, and the
pollutant attaining it is recorded as the dominant pollutant.

Three things break naive implementations, and all three are handled:

**Units.** Open-Meteo reports the four gases in µg/m³. The EPA breakpoint
tables are in ppm for ozone and CO and ppb for NO2 and SO2. Conversion uses
the molar volume of an ideal gas at 25 °C and 1013.25 hPa together with each
molecular weight. Ozone is the classic trap: its table is in ppm like CO, not
ppb like the other two gases, and treating it as ppb inflates every ozone
sub-index by a factor of a thousand and makes ozone spuriously dominant on
every single row.

**Truncation.** The EPA specifies truncating each concentration to a set
number of decimals *before* interpolating between breakpoints. Skipping this
shifts values across breakpoint edges and changes the reported band.

**Completeness.** EPA guidance requires a minimum data completeness before an
average may be reported. A 24-hour PM2.5 mean built from three observations is
not a 24-hour mean, so `rolling_average` requires 75 percent of the window to
be present and returns NaN otherwise, rather than a value that looks
authoritative and is not.

Concentrations above the top breakpoint are clamped to 500 rather than
extrapolated, because the EPA does not define the index beyond its table and
extrapolating invents precision that does not exist.

The PM2.5 breakpoints use the 2024 revision, in which the upper bound of the
Good band moved from 12.0 to 9.0 µg/m³. The edition in use is recorded in the
data card.

Hourly indices are then collapsed to the local calendar day (Asia/Karachi) as
mean, max, min, standard deviation and an hours-observed count. Any day with
fewer than 18 hourly observations is flagged by the schema validator as a thin
day.

### AQI bands

| Range | Label |
|---|---|
| 0 to 50 | Good |
| 51 to 100 | Moderate |
| 101 to 150 | Unhealthy for Sensitive Groups |
| 151 to 200 | Unhealthy |
| 201 to 300 | Very Unhealthy |
| 301 to 500 | Hazardous |

Each band carries a colour and a guidance sentence, defined once in
`src/config.py` and used by the dashboard, the alerts and the band-accuracy
metric alike.

---

## 6. Feature engineering

`src/features/build.py` is the single transform used by the backfill, the
hourly pipeline and serving. Training and inference call the same function,
which is what makes the feature definitions structurally identical in both
paths rather than identical by convention.

Every feature is **causal**: computed from information available at the end of
day *t* and used to predict days *t+1* through *t+3*. There is exactly one
forward-looking family, and it is legitimate, as explained below.

| Family | Contents |
|---|---|
| Calendar | Day of week, day of month, month, quarter, day of year, ISO week, weekend flag, plus sine and cosine encodings of day of year, day of week and month so that 31 December sits next to 1 January rather than 364 units away |
| Lags | 1, 2, 3, 4, 5, 6, 7 and 14 day lags of daily mean AQI, daily max AQI, PM2.5 and PM10. Lags 4, 5 and 6 exist so the seasonal-naive baseline can be expressed exactly rather than approximated |
| Rolling statistics | 3, 7, 14 and 30 day trailing mean and standard deviation (plus min and max for daily mean AQI), with `closed='right'` so day *t* is inside its own window, which is correct because day *t* is observed at prediction time |
| Change rates | 1, 3 and 7 day differences and percentage changes, plus the slope of a least-squares line through the trailing week in AQI per day, plus the gap between today and the 30-day mean |
| Pollutant mix | PM2.5 to PM10 ratio, coarse PM fraction, and one-hot indicators for the dominant pollutant. In Karachi these separate dust intrusions from combustion smog, which behave and persist differently |
| Current meteorology | Daily mean of temperature, relative humidity, dew point, surface pressure, cloud cover, wind speed and boundary layer height, with min and max for temperature, wind and mixing height, precipitation sum, maximum wind gust, and a circular-safe wind direction encoded as sine and cosine components |
| Derived physics | Ventilation index (mixing height × wind speed), a stagnation flag (wind below 2 m/s and mixing height below 500 m), boundary-layer range, diurnal temperature range as an inversion proxy, dew-point depression, a rained flag and a three-day precipitation total |
| Forecast meteorology | The same weather variables for days *t+1* to *t+3*, taken from the weather forecast |

That last family is the only genuinely forward-looking signal, and it is
legitimate: a weather forecast for *t+3* really is available on day *t*. What
is not legitimate is training on *observed* weather for those days and then
serving on *forecast* weather, because that quietly gives the model a quality
of input at training time that it will never have in production. The backfill
uses the historical-forecast endpoint where coverage allows, records which
regime it used, and the dashboard's live scoring measures whatever skew
remains.

Wind direction deserves a note. It is circular, so the arithmetic mean of 350°
and 10° is 180°, which points the opposite way. The daily aggregate averages
the sine and cosine components instead.

### Targets

`add_targets` shifts the daily mean and daily max AQI backwards by 1, 2 and 3
days, producing six target columns. No recursion, no compounding.

### Feature selection at fit time

`feature_columns()` derives the model input list from the frame rather than
hardcoding it: everything numeric that is not a target and not metadata. A
feature added in one place therefore cannot be silently missing at serving
time. The present-day value of a target counts as a valid lag-zero feature and
is retained.

### The feature contract

`src/features/schema.py` is enforced on every write. `sanitise()` coerces
dtypes, replaces infinities and drops all-null columns. `validate_daily()`
returns a data-quality report covering row and column counts, missing required
columns, thin days, missing dates, duplicate dates, AQI range violations and
columns more than half null. `save_schema()` and
`check_against_saved_schema()` catch drift between the backfill and the hourly
pipeline, which is the kind of fault that produces a model quietly trained on
a column that no longer exists at serving time.

---

## 7. The model ladder

Every candidate for a given (target, horizon) pair runs through the identical
cross-validation path: same folds, same metrics, no special-casing. The
baselines are scikit-learn compatible estimators precisely so that this is
true.

### Tier 0: baselines

| Model | Definition | What it tests |
|---|---|---|
| `persistence` | ŷ(t+h) = y(t) | The bar. Tomorrow is today |
| `seasonal_naive` | ŷ(t+h) = y(t+h-7) | The same weekday one week earlier |
| `climatology` | The historical average for this time of year, smoothed across day of year with a wrap-around window | Whether a model has learned the calendar rather than the weather |
| `drifted_persistence` | Persistence plus the trailing weekly slope, damped by horizon | Whether a model's gain is just trend-following |

All four fall back to the training mean when their input column is missing or
null, so a single bad row never produces a NaN forecast.

### Tier 1 and 2: linear models and tree ensembles

Ridge, ElasticNet, RandomForest, ExtraTrees and HistGradientBoosting. Linear
models are wrapped in a pipeline of median imputation then standard scaling;
tree models get median imputation only. Imputing inside the pipeline means the
imputer is fitted on the training fold alone and never on validation data, the
same discipline as the split itself.

HistGradientBoosting is the exception and deliberately gets no imputer: it
handles NaN natively, which lets it use missingness as a signal rather than
papering over it with a median. A quantile variant is available for direct
quantile regression where an interval that widens with genuine uncertainty is
wanted rather than one global band.

Hyperparameter search is a small fixed grid (three or four configurations for
Ridge, RandomForest and HistGradientBoosting) rather than a large randomised
search. With roughly 1,500 daily rows an aggressive search overfits the
validation folds, and the gain is smaller than the gain from one more good
feature.

### Tier 3: statistical

SARIMAX with order (2,0,2) and weekly seasonal order (1,0,1,7), fitted on the
AQI series itself. It forecasts `horizon` steps from each origin and is then
extended with the newly observed day without re-estimating parameters, which
is exactly how it would run in production, rather than being handed the whole
validation window at once. The annual cycle is left to the calendar features
in the tabular models rather than asked of a model fitted on a few hundred
daily points. If `statsmodels` is absent the tier is skipped and recorded as
skipped.

### Tier 4: deep learning

An MLP on the same tabular features the tree models see, and an LSTM over a
21-day rolling window of the raw daily series. The LSTM is given a genuine
sequence rather than the same flat feature vector, so the comparison is fair
to it: it has to learn the temporal structure that the tabular models get
handed as explicit lag features. Both use Adam, smooth L1 loss, gradient
clipping, dropout and early stopping on a **chronological** tail split, never
a random one, for the same reason the outer cross-validation is never random.

The deep tier is expected to lose. With roughly 1,500 daily rows an LSTM is
being asked to learn from very little, and gradient boosting on good features
is the likely winner. Reporting that the deep model was built, evaluated on
identical folds and lost is a stronger result than quietly omitting it. If
`torch` is absent the tier is skipped.

---

## 8. Evaluation protocol

`src/models/evaluate.py` is the most important file in the modelling code,
because this is where projects of this kind usually go wrong. Hourly AQI is
strongly autocorrelated. A random split puts hour 14 in train and hour 15 in
test, then reports a magnificent R-squared for a model that has learned
nothing.

**Expanding-window walk-forward splits.** The oldest data is always in
training. Five folds by default, each validating on the 60 days after the
previous fold, with a minimum of 365 training days. When the series is too
short the configuration shrinks (validation window first, then minimum
training length, then fold count) rather than failing, and returns an empty
list only when even one honest fold is impossible.

**A mandatory gap.** `CV_GAP_DAYS` defaults to `max(HORIZONS)`, that is three
days. Without it the label of the last training row falls inside the
validation window. `assert_no_leakage()` checks this for every fold and is
invoked both by the tests and by the training pipeline itself, so a
configuration change that reintroduces leakage fails loudly.

**A chronological holdout.** The final 90 days are removed before any model
search begins and are evaluated exactly once, at the end, by
`evaluate_holdout()`.

**Metrics.** RMSE, MAE, R-squared, bias, exact AQI-band accuracy,
within-one-band accuracy, and sample count. Band accuracy belongs next to
RMSE rather than below it: a dashboard reader consumes "Unhealthy tomorrow",
not "168.4 tomorrow".

**Skill score.** `1 - RMSE_model / RMSE_persistence`. One is perfect, zero is
no better than persistence, negative is worse. This is the headline number,
not R-squared.

**Prediction intervals.** Out-of-fold residuals are collected per (target,
horizon) and their 10th, 25th, 75th and 90th percentiles are stored in the
bundle. A bare point forecast of "AQI 168" claims a precision the model does
not have, so the dashboard shows a band built from the residual distribution
for that specific horizon. Intervals widen with horizon because the residuals
do.

---

## 9. The promotion rule

An automated retraining loop with no promotion gate can only drift downward:
one bad data day produces a worse model, the worse model is deployed
unconditionally, and nobody notices until the forecasts are visibly wrong.

`should_promote()` requires both of the following:

1. the new model beats persistence at **every** target and horizon on the
   untouched holdout, and
2. its mean holdout RMSE is not more than 5 percent worse than the incumbent's,
   read from the Hopsworks model registry.

If either condition fails, the incumbent stays, the run logs exactly why, and
the rejected bundle is still written to `data/artifacts/rejected/` so the
failure can be inspected. Both the decision and the reason are stored in the
bundle metadata and surfaced on the dashboard.

---

## 10. Serving and the model bundle

`src/models/bundle.py` defines the single deployed artefact. One bundle holds:

- the six fitted regressors, keyed by (target, horizon)
- the exact feature list they were trained on
- the residual quantiles that produce the prediction intervals
- provenance metadata: run id, training timestamp, date range, row and feature
  counts, the selected model per pair, cross-validation metrics, holdout
  metrics, the promotion decision and its reason, and the random seed

Serving loads this and nothing else, so there is no way for the app to
assemble a different feature set than the one the models were fitted on.
`align()` reindexes the serving frame onto the stored feature list, adding
missing columns as NaN and dropping extras, and logs a warning when a training
feature is missing. Without that, a feature added upstream after the model was
trained would silently reorder the matrix and corrupt every prediction.

`predict_row()` takes the latest row of features and returns one row per
horizon carrying the predicted mean and max, the lower and upper interval
bounds, the AQI band label, its colour, the guidance sentence, the prediction
timestamp and the model version.

Inference runs in the pipeline, not in the app. Community Cloud apps are
memory-constrained and a model load per page view is wasteful, and a stored
prediction with a timestamp is auditable. That last point is what makes
`score_past_predictions()` possible: it joins stored forecasts to what
actually happened and reports live RMSE, MAE and bias per horizon. Backtest
metrics are an estimate of live performance; this is live performance, and the
gap between the two is the most informative number in the project, because it
is where any remaining train/serve skew shows up.

The forecast origin is the most recent day with an **observed** AQI. Later
rows exist in the feature frame, since it extends into the forecast window,
but they have no observed target and are not a valid origin.

---

## 11. Explainability

SHAP is computed in the training job and cached to disk, never inside the
Streamlit app. A TreeExplainer over a few hundred rows will exhaust a
Community Cloud memory allowance, and the app would recompute it on every page
view.

Two artefacts are produced. **Global importance** answers "what does this model
pay attention to?", which is the sanity check: if wind speed and mixing height
do not rank highly, the model has learned the calendar instead of the
meteorology. **Per-day contributions** answer "why is Thursday forecast to be
bad?", which is the question a dashboard reader actually has.

Two implementation details matter. Tree models are detected by type name
rather than by the presence of `feature_importances_`, because
HistGradientBoostingRegressor is a fully supported tree ensemble that does not
expose that attribute, and an attribute check would silently skip the model
that most often wins. And the pipeline's preprocessing steps are applied
before explaining, so SHAP sees what the tree sees.

When the winning model is linear or neural, TreeExplainer does not apply and
`permutation_fallback()` produces model-agnostic importance instead, so the
page is never blank. The explanation pipeline is separate from training so
explanations can be refreshed without retraining, and so a SHAP failure can
never fail an otherwise successful training run.

---

## 12. Alerts

Two design choices shape `src/pipelines/alerts.py`.

**Alerts fire on deterioration, not merely on a high value.** In a Karachi
winter the AQI sits above 150 for weeks. An alert that fires every hour for a
month is an alert nobody reads. A forecast day triggers an alert only when its
severity exceeds today's observed severity.

**State is persisted.** `data/alert_state.json` records what has already been
sent for each target date at what severity, so the same day never alerts twice
at the same level. An escalation (unhealthy to very unhealthy) does send again.

| Threshold | Severity |
|---|---|
| AQI ≥ 150 | `unhealthy` |
| AQI ≥ 200 | `very_unhealthy` |
| AQI ≥ 300 | `hazardous` |

Delivery is pluggable and every channel is optional: ntfy (a topic string,
free, no account), a Slack incoming webhook, and SMTP email. With none
configured the alert is still evaluated, recorded to the state file and shown
on the dashboard. `--dry-run` evaluates without sending; `--force` ignores
de-duplication.

---

## 13. The dashboard

Five Streamlit pages. Everything they render is precomputed by the pipelines:
the app fits nothing, computes no SHAP and calls no APIs. Loaders are cached
for fifteen minutes, matched to the hourly feature schedule.

| Page | Contents |
|---|---|
| **Forecast** | Today's observed mean and max with dominant pollutant, PM2.5 and wind, then the three forecast days as coloured band cards with intervals, the band legend, health guidance and a data-freshness note |
| **History and EDA** | The historical series, seasonal and weekly patterns, distributions and the relationships between meteorology and AQI |
| **Model performance** | The leaderboard built from `data/experiments.csv`, holdout metrics per target and horizon, skill against persistence, live scoring of past forecasts, and error diagnostics |
| **Explainability** | Global feature importance and the per-day contribution breakdown for the latest forecast |
| **Alerts** | Current thresholds, the active alert state and the delivery history |

---

## 14. Automation and scheduling

| Workflow | Trigger | What it does |
|---|---|---|
| `feature_pipeline.yml` | Hourly at :17, plus manual | Restore state, fetch and engineer features, compute the forecast, check alerts, persist state. Opens (or comments on) a GitHub issue labelled `pipeline-failure` if it fails |
| `training_pipeline.yml` | Daily at 02:00 UTC (07:00 Karachi, after the overnight data lands), plus manual | Quick ladder by default: baselines, linear and tree models. Evaluate, fit, score the holdout, promote or reject, refresh SHAP, persist |
| `full_ladder.yml` | Sundays at 03:00 UTC, plus manual | The complete comparison including SARIMAX, MLP, LSTM and the hyperparameter grid. Weekly rather than daily because it costs roughly an hour on a two-core runner and the ranking does not change day to day |
| `backfill.yml` | Manual only | Optionally probe coverage first, then pull the full history in chunks and write the data card |
| `ci.yml` | Push and pull request to `main` | Ruff on the error-class rules, then the full test suite. The suite is entirely offline, so it cannot be flaked by an API outage |

Both scheduled pipelines use a `concurrency` group so a slow run cannot
overlap the next one.

The hourly job deliberately re-fetches a 72-hour overlapping window rather
than just the newest hour. GitHub's scheduler is best-effort: runs drift by
minutes and are sometimes skipped entirely under platform load. A pipeline
that assumes the previous run happened accumulates silent holes. This one
re-fetches and upserts, so any gap left by a missed run is healed by the next
one without intervention.

The daily job's timeout is 90 minutes against a measured cost of roughly two
minutes per target-horizon pair on a two-core runner for the quick ladder, so
about twelve minutes for six pairs. The weekly job allows 300 minutes; the
backfill allows 360.

---

## 15. How state survives ephemeral runners

Actions runners are destroyed after every job, so the Parquet mirror and the
model artefacts need somewhere durable. Hopsworks is the primary store, but
the mirror is exactly what keeps the system working when Hopsworks is not
reachable, so it needs durable storage too.

`scripts/data_branch.sh` keeps that state on an **orphan `data` branch**.
`restore` fetches the branch shallowly and checks out `data/` before a run;
`save` commits and force-pushes it afterwards, dropping the HTTP response
cache first because that is a local speed-up rather than state. The approach
needs no external service and keeps binary churn out of `main`'s history. It
also has a useful side effect: the commits count as repository activity, which
stops GitHub from auto-disabling the scheduled workflows after 60 days of
quiet.

---

## 16. Getting started

### Step 1: push and configure

```bash
git init && git add . && git commit -m "Initial commit"
gh repo create pearls-aqi-predictor --public --source=. --push
```

Make the repository **public**. Actions minutes are unlimited on public
repositories and metered on private ones, and an hourly job is roughly 730
runs a month.

Add repository secrets under Settings, then Secrets and variables, then
Actions:

| Secret | Required | Where to get it |
|---|---|---|
| `HOPSWORKS_API_KEY` | Recommended | [app.hopsworks.ai](https://app.hopsworks.ai), Account Settings, API keys |
| `HOPSWORKS_PROJECT` | Recommended | The project name you created |
| `AQICN_TOKEN` | Optional | [aqicn.org/data-platform/token](https://aqicn.org/data-platform/token/) |
| `NTFY_TOPIC` | Optional | Any string; subscribe to it in the ntfy app |
| `SLACK_WEBHOOK_URL` | Optional | A Slack incoming webhook |
| `ALERT_EMAIL_TO`, `SMTP_USER`, `SMTP_PASSWORD` | Optional | Your mail provider's app password |

Without Hopsworks the system still runs and falls back to the Parquet mirror
on the `data` branch, but the managed feature store and model registry are
lost.

### Step 2: probe and backfill

Run the **Historical backfill** workflow with `probe_first` enabled. The probe
finds the true earliest available date for Karachi's coordinates and writes
`data/coverage.json`. The backfill then pulls everything from there to
yesterday and generates `docs/DATA_CARD.md` from what it actually saw.

This is the critical path. Nothing downstream works until it succeeds.

### Step 3: train

Run the **Training pipeline** workflow once by hand. Check the run summary:
it prints the training summary JSON, including which model won each pair, the
holdout metrics, and whether the bundle was promoted.

### Step 4: deploy the dashboard

At [share.streamlit.io](https://share.streamlit.io):

- Repository: this one, branch `main`
- Main file: `app/streamlit_app.py`
- Advanced settings, Python version: 3.11

Dependencies are picked up automatically from `app/requirements.txt`.
Community Cloud cannot be pointed at a custom dependency filename: it
recognises only `uv.lock`, `Pipfile`, `environment.yml`, `requirements.txt`
and `pyproject.toml`, searching the entrypoint's directory first and then the
repository root. That is why the app's dependency file lives in `app/` rather
than being called `requirements-app.txt`, and why it is deliberately lean:
the dashboard renders precomputed Parquet and JSON and never unpickles a
model, so scikit-learn, statsmodels, shap and torch are not installed there.

The hourly and daily schedules take over from there.

---

## 17. Local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # fill in what you have

python -m src.clients.open_meteo --probe    # verify archive coverage
python -m src.pipelines.backfill            # pull history (slow, chunked)
python -m src.pipelines.train --quick       # train
python -m src.pipelines.predict             # forecast
python -m src.pipelines.alerts --dry-run    # evaluate alerts without sending
streamlit run app/streamlit_app.py

pytest tests -q                             # 94 tests, no network needed
```

Requires Python 3.11 (the version the workflows pin). For the deep tier
locally, add `pip install -r requirements-deep.txt`.

Dependencies are pinned to minor version ranges so that a scheduled run cannot
break overnight on someone else's release.

---

## 18. Command reference

| Command | Purpose |
|---|---|
| `python -m src.clients.open_meteo --probe` | Find the true archive start date; writes `data/coverage.json` |
| `python -m src.pipelines.backfill` | Full history pull. `--start`, `--end`, `--chunk-days` (default 180), `--no-historical-forecast` |
| `python -m src.pipelines.features` | Hourly fetch and feature build. `--lookback-hours`, `--no-cross-check` |
| `python -m src.pipelines.train` | Full ladder. `--quick` skips SARIMAX, torch and tuning; `--no-register` skips Hopsworks registration |
| `python -m src.pipelines.predict` | Compute the three-day forecast. `--dry-run` to avoid writing |
| `python -m src.pipelines.explain` | Refresh SHAP without retraining |
| `python -m src.pipelines.alerts` | Evaluate and send alerts. `--force` ignores de-duplication, `--dry-run` evaluates only |
| `./scripts/data_branch.sh restore` | Pull persisted state from the `data` branch |
| `./scripts/data_branch.sh save "message"` | Commit and push state to the `data` branch |
| `pytest tests -q` | The offline test suite |

---

## 19. Configuration reference

Everything lives in `src/config.py`. No module hardcodes a coordinate, a
horizon or a threshold anywhere else.

| Setting | Default | Meaning |
|---|---|---|
| `CITY` | Karachi, 24.8607 N, 67.0011 E, Asia/Karachi | Target location. Changing it retargets the whole system |
| `HORIZONS` | `(1, 2, 3)` | Forecast horizons in days |
| `TARGETS` | `("aqi_mean", "aqi_max")` | Daily aggregates predicted |
| `HOLDOUT_DAYS` | 90 | Chronological holdout length |
| `CV_MIN_TRAIN_DAYS` | 365 | Minimum training window |
| `CV_VAL_DAYS` | 60 | Validation window per fold |
| `CV_N_SPLITS` | 5 | Walk-forward folds |
| `CV_GAP_DAYS` | `max(HORIZONS)` | Gap between train end and validation start. Must be at least the horizon |
| `ARCHIVE_START_FALLBACK` | 2022-06-01 | Used only when `data/coverage.json` is absent |
| `HOURLY_LOOKBACK_HOURS` | 72 | Overlap re-fetched on every hourly run to heal gaps |
| `ALERT_THRESHOLDS` | 150 / 200 / 300 | Unhealthy, very unhealthy, hazardous |
| `MIN_HOURS_PER_DAY` | 18 (in `schema.py`) | Below this a day is flagged as thin |
| `RANDOM_SEED` | 42 | Set for every estimator and the torch training loop |
| `MAX_RETRIES` | 5 | Client retry budget |
| `AQI_DATA_DIR` | `data/` | Environment variable overriding the data directory |

Data directory contents at runtime:

| Path | Contents |
|---|---|
| `data/hourly_raw.parquet` | Hourly observations plus the computed AQI |
| `data/daily_features.parquet` | The daily modelling frame with targets |
| `data/predictions.parquet` | Every stored forecast, for live scoring |
| `data/experiments.csv` | Append-only log of every model evaluation |
| `data/artifacts/model_bundle.joblib` | The deployed bundle |
| `data/artifacts/model_metadata.json` | Provenance and metrics |
| `data/artifacts/shap_global.json`, `shap_latest.json` | Explanations |
| `data/coverage.json` | Probe result |
| `data/backfill_summary.json`, `last_feature_run.json`, `training_summary.json`, `latest_forecast.json` | Run summaries |
| `data/alert_state.json` | Alert de-duplication state and history |
| `data/feature_schema.json` | The saved feature contract |
| `data/http_cache/` | Backfill response cache, never persisted to the `data` branch |

---

## 20. Testing

94 tests, no network access required, run on every push.

| File | Covers |
|---|---|
| `test_aqi.py` | Sub-index interpolation against worked EPA examples, breakpoint edges, unit conversion including the ozone ppm trap, truncation behaviour, averaging-window completeness, dominant pollutant selection, daily aggregation |
| `test_features.py` | Causality (perturbing future values and asserting past features do not move), lag and rolling correctness, change rates, target construction, the feature-column derivation |
| `test_evaluation.py` | Split construction, gap enforcement, the leakage assertion, shrink behaviour on short series, every metric, skill score, residual quantiles |
| `test_bundle_and_clients.py` | Bundle save and load round-trip, feature alignment with missing and extra columns, band lookup at every boundary, client response parsing |
| `test_alerts.py` | Severity thresholds, deterioration logic, de-duplication and escalation |

The fixtures come from `tests/synthetic.py`, a generator that reproduces the
structure that actually matters: a strong winter pollution season from
November to February, a diurnal emissions cycle, a weekday and weekend
difference, dispersion driven by wind speed and mixing-layer depth so that
meteorology carries genuine predictive information beyond persistence,
autocorrelated synoptic weather so the series is persistent but not trivially
so, and occasional dust events and rain washout.

It is a test fixture, not a data source. It exists so that a broken feature
transform or a leaking split fails in CI rather than in production.

---

## 21. Operations and troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| Backfill produces no data | Check `data/coverage.json`. If the probe reported an error, verify the coordinates and Open-Meteo's status. Re-run: cached chunks are not re-fetched |
| `no features available -- run the backfill first` | The `data` branch is empty or `restore` did not run. Check the restore step's output in the workflow log |
| `no trained model -- run the training pipeline first` | Expected before the first successful training run. The hourly workflow tolerates this and continues |
| Training says `NOT PROMOTED` | Working as designed. The reason is in the log, in `data/training_summary.json` and on the dashboard. The rejected bundle is in `data/artifacts/rejected/` |
| Scheduled workflows stopped firing | GitHub disables schedules after 60 days of no repository activity. The `data` branch commits normally prevent this; re-enable the workflow in the Actions tab if it happened |
| Hourly runs are skipped or late | Expected. GitHub's scheduler is best-effort, and the 72-hour overlap heals the gaps |
| Dashboard shows stale data | The freshness note reports the latest observed day. Check the most recent hourly run, then the `data` branch commit history |
| Explainability page is empty | The winner is not a tree model and the permutation fallback also failed. Check the explain step's log |
| Hopsworks writes failing | Every write logs a warning and returns False; the Parquet mirror is written first, so the run continues. Check the key and the free-tier quota |
| A GitHub issue labelled `pipeline-failure` appeared | The hourly pipeline failed and opened it automatically. The run link is in the issue body; subsequent failures comment on the same issue instead of opening new ones |

---

## 22. Known limitations

- **CAMS is a model reanalysis, not ground truth.** Open-Meteo serves modelled
  values, not readings from physical instruments. The AQICN cross-check
  quantifies the disagreement; it does not eliminate it.
- **Train/serve weather skew.** Training features use forecast-quality weather
  where the historical-forecast endpoint covers the range and observed weather
  where it does not. The backfill records which regime was used, and whatever
  skew remains shows up in the live scoring on the dashboard.
- **Limited data volume.** Roughly four years of history is only about 1,500
  daily training rows. This bounds what any model can learn and is the main
  reason the deep tier is not expected to win.
- **Dust storms.** Sharp AQI spikes with little meteorological warning. Dust
  and coarse-PM features are included, but peaks should be expected to be
  underpredicted.
- **Regime change.** Policy, industrial or fuel changes break historical
  relationships. Daily retraining adapts, and live scoring surfaces the drift,
  but neither prevents it.
- **One city.** The models are fitted to Karachi's emissions and meteorology.
  Applying them elsewhere without refitting will produce confident nonsense.
- **Not an official advisory.** This is model output, unaffiliated with any
  environmental authority, and it must not be used for medical decisions. See
  `docs/MODEL_CARD.md`.

---

## 23. Project status

| Component | State |
|---|---|
| EPA AQI implementation | Done, tested against worked examples |
| Feature pipeline | Done, verified on synthetic data |
| Backfill | Done, chunked and resumable |
| Model ladder | Done: 4 baselines, 5 sklearn families, SARIMAX, MLP, LSTM |
| Evaluation harness | Done, walk-forward with leakage guards |
| Automation | 5 workflows written |
| Dashboard | 5 pages written |
| Alerts | Done, with de-duplication |
| **First live run** | **Pending, see [Getting started](#16-getting-started)** |

The entire system has been verified end to end against a synthetic
Karachi-like series, with 94 passing tests. It has not yet run against the
live API, because the development sandbox has no egress to Open-Meteo. The
first real execution happens in GitHub Actions.

---

## 24. Documentation index

| Document | Contents |
|---|---|
| [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) | The full plan, with every design decision argued at length. Section 2 covers the data source choice, section 3 the modelling decisions and the train/serve skew analysis |
| [`docs/DATA_CARD.md`](docs/DATA_CARD.md) | Generated by the backfill from what it actually retrieved: source, coverage, distribution, gaps, failed chunks |
| [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) | Intended use, out-of-scope uses, architecture, evaluation, the promotion rule, failure modes and ethical considerations |
| [`docs/REPORT.md`](docs/REPORT.md) | The project report, with sections marked *[fill from run]* populated after the first live execution |
| [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) | Exploratory analysis, including the validation of the computed AQI against the API's convenience field |

---

## 25. Licence

MIT.
