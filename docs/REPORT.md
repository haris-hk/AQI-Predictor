# Pearls AQI Predictor — Project Report

**Author:** Fahad Qureshi · **City:** Karachi (24.8607 N, 67.0011 E)
**Live dashboard:** _add the Streamlit URL_ · **Repository:** _add the GitHub URL_

> Status: skeleton. Sections marked *[fill from run]* are populated from
> pipeline output — `data/coverage.json`, `docs/DATA_CARD.md`,
> `data/experiments.csv` and `data/training_summary.json` — after the first
> live execution. Everything else is written and final.

---

## 1. Executive summary

*[fill from run]* One paragraph: what was built, what accuracy was achieved at
each horizon, and whether it beats persistence.

The headline number is not R². It is the **skill score against persistence**:
how much better the model is than assuming tomorrow looks like today. Section 6
explains why.

## 2. Problem statement

Karachi has some of the worst air quality of any large city, and the health
guidance that matters — should a child play outside on Thursday — needs a
forecast rather than a current reading. The task: predict the US AQI three days
ahead, automate the whole pipeline, and serve it publicly at zero cost.

## 3. Architecture

Four components, none of which is a server we operate:

| Layer | Choice | Why |
|---|---|---|
| Data | Open-Meteo Air Quality + ERA5 | Only free source with both a multi-year archive and a 7-day forecast, and it needs no API key |
| Compute | GitHub Actions | Scheduled compute with no machine; unlimited minutes on public repos |
| State | Hopsworks feature store + model registry | Managed; free tier; Parquet mirror on an orphan branch as fallback |
| Serving | Streamlit Community Cloud | Free public URL, no web server |

**On "serverless".** Apache Airflow was the alternative the brief named. It was
rejected because self-hosted Airflow needs a machine that stays up, which
contradicts the serverless requirement and adds cost. That trade-off is stated
rather than glossed: Airflow offers far better orchestration semantics —
backfill management, retries, dependency graphs — and we give those up in
exchange for genuinely zero infrastructure.

## 4. Feature engineering

Roughly 170 features from eight families: calendar (with cyclical encoding so
December is adjacent to January), lags, rolling statistics, change rates,
pollutant composition, current meteorology, **forecast** meteorology, and
derived physics (ventilation index, stagnation flag, inversion proxy).

### 4.1 Computing the AQI correctly

The EPA index is not an hourly quantity. It uses a 24-hour mean for PM2.5 and
PM10, an 8-hour mean for ozone and CO, and 1-hour values for NO2 and SO2, then
takes the maximum sub-index across pollutants.

Two implementation traps, both handled in `src/features/aqi.py`:

- **Units.** Open-Meteo returns gases in µg/m³. The EPA breakpoints are in ppm
  for ozone and CO, and ppb for NO2 and SO2. Ozone is the trap: its table looks
  like the other gases' but is in ppm, and converting it to ppb inflates its
  sub-index a thousandfold, making ozone spuriously dominant on every row. This
  bug was caught during development by a unit test asserting that a high-PM2.5
  hour is PM2.5-dominant.
- **Truncation.** The EPA specifies truncating each concentration to a fixed
  number of decimals before interpolating. Skipping it shifts values across
  breakpoint edges and changes the reported band.

*[fill from run]* Agreement between our computed AQI and the API's field.

### 4.2 The train/serve skew problem

Forecast weather for days t+1..t+3 is the only genuinely forward-looking signal
available, and it is legitimate — a forecast for Thursday really does exist on
Monday. The subtlety is in training: backfilling with *observed* weather for
those days gives the model perfect information it will never have live. Train on
truth, serve on forecast, and the model degrades the moment it is deployed.

The fix is Open-Meteo's historical-forecast endpoint, which serves past forecast
runs — weather as it was predicted at the time. The backfill uses it where
coverage allows and records which regime produced the training set in
`DATA_CARD.md`.

*[fill from run]* Which regime was used, and the measured live-versus-backtest gap.

## 5. Data

*[fill from run]* Range, row counts, gaps — from `docs/DATA_CARD.md`.

## 6. Modelling

### 6.1 Why persistence is the bar

AQI is strongly autocorrelated. That has a consequence which is easy to miss: a
model can score a high R² while adding no information at all, simply by learning
to echo yesterday. Worse, a random train/test split on autocorrelated daily data
puts adjacent days on opposite sides of the split and inflates every metric.

So the evaluation is built around three reference models — persistence, seasonal
naive and climatology — and the headline metric is skill against persistence.
Zero means no gain. A model that cannot clear zero has not earned deployment,
whatever its R².

### 6.2 Validation protocol

- Expanding-window walk-forward CV, 5 folds, never shuffled
- A gap of at least the forecast horizon between train end and validation start,
  so the last training row's label cannot fall inside the validation window
- A 90-day chronological holdout, touched exactly once, after selection
- Leakage guarded structurally in `tests/test_features.py`: future values are
  perturbed and past features asserted not to move

### 6.3 The ladder

| Tier | Models |
|---|---|
| 0 | Persistence, seasonal naive, climatology, drifted persistence |
| 1 | Ridge, ElasticNet |
| 2 | RandomForest, ExtraTrees, HistGradientBoosting |
| 3 | SARIMAX |
| 4 | MLP, LSTM (PyTorch) |

### 6.4 Results

*[fill from run]* Leaderboard from `data/experiments.csv`; holdout table from
`data/training_summary.json`.

**Expected finding, to be confirmed or overturned by the run.** With roughly
1,500 daily rows, gradient boosting should beat the neural models. If it does,
that is the result and it will be reported as such: a deep model that was built,
evaluated on identical folds and lost is a stronger finding than a deep model
quietly omitted. If the LSTM wins, that is more interesting still and deserves
investigation into why.

## 7. Explainability

SHAP TreeExplainer, precomputed in the training job and cached — computing it
inside the Streamlit app would exceed the Community Cloud memory allowance.

The interesting question is not which feature ranks first (lags will, because
the series is persistent) but what ranks *next*. Forecast meteorology ranking
second is the sign the model has learned dispersion physics rather than the
calendar; calendar features dominating would mean it had learned "January is
bad" rather than "Thursday is bad".

*[fill from run]* Top features and interpretation.

## 8. Automation

| Workflow | Schedule | Purpose |
|---|---|---|
| `feature_pipeline` | hourly | Fetch, engineer, predict, check alerts |
| `training_pipeline` | daily 02:00 UTC | Quick ladder, promotion gate |
| `full_ladder` | Sunday 03:00 UTC | Complete comparison including torch |
| `backfill` | manual | Historical pull, chunked and resumable |
| `ci` | on push | Lint and 94 tests |

Three platform realities shaped this design:

1. **Cron is best-effort.** Runs drift and are sometimes skipped. So the hourly
   job re-fetches a 72-hour overlapping window and upserts, healing any gap the
   last run left, rather than assuming it happened.
2. **Runners are ephemeral.** State lives on an orphan `data` branch — no
   external service, and the commits keep the schedules alive past GitHub's
   60-day inactivity cutoff.
3. **Training is slower than expected.** Measured at roughly two minutes per
   target-horizon pair on a two-core runner, so the daily job runs the quick
   ladder (~12 minutes) and the full ladder including SARIMAX and torch runs
   weekly. The original plan budgeted 15 minutes for the full ladder daily,
   which measurement showed was not achievable.

### 8.1 The promotion rule

A new model is registered only if it beats persistence at every horizon on the
holdout and is not materially worse than the incumbent. Without this gate, an
automated retraining loop can only drift downward: one bad data day produces a
worse model, deploys it unconditionally, and nobody notices until the forecasts
are visibly wrong.

## 9. Dashboard

Five pages: forecast, history and EDA, model performance, explainability,
alerts. The app renders precomputed artefacts and fits nothing — which is what
keeps it inside the memory allowance and fast on a cold start.

Every forecast is shown with a prediction interval from the out-of-fold residual
distribution. A bare "AQI 168" claims a precision the model does not have.

## 10. Alerts

Fire on a **deterioration**, not merely on a high value. In a Karachi winter the
index sits above 150 for weeks; an alert repeating every six hours for a month
is one nobody reads. State is persisted so the same day never alerts twice at
the same severity.

## 11. What I would do next

1. **Ground-station calibration.** CAMS is a model; AQICN gives real readings.
   A bias correction fitted between them would likely be the single largest
   accuracy gain available.
2. **Hourly-resolution targets.** More samples, and more useful — "bad in the
   morning, clear by evening" is more actionable than a daily mean.
3. **Multi-city.** The pipeline is parameterised by coordinates already; the
   feature store schema supports `city_id` as a key.
4. **Probabilistic forecasting properly.** Quantile regression per horizon
   rather than residual quantiles, evaluated with pinball loss.
5. **Dust-event detection.** The largest errors will be dust intrusions, which
   arrive with little meteorological warning. Satellite AOD is the lead signal.

## 12. Reflection

*[fill from run]* What was hardest, what surprised you, what you would do
differently.
