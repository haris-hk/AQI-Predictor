# Pearls AQI Predictor — Project Build Plan

**Status:** Planning complete, build not started
**Author:** Fahad Qureshi
**Plan date:** 2026-09-05
**Canonical location:** `internship/docs/PROJECT_PLAN.md` (this file is the reference document for every future working session)

---

## 0. What we are building, in one paragraph

An end-to-end, zero-cost, 100% serverless machine learning service that forecasts the Air Quality Index for one city for the next 3 days. A feature pipeline pulls pollutant and weather observations on a schedule and writes engineered features to a managed feature store. A training pipeline reads those features on a daily schedule, trains and compares a family of models from naive baselines through gradient boosting to a small neural network, and registers the winner in a model registry. A Streamlit dashboard loads the registered model plus the newest features and renders a live 3-day forecast with explanations and hazard alerts. GitHub Actions is the only compute. Nothing runs on a server we own.

There is no infrastructure to pay for and no machine to keep alive. The whole system is four Python entry points, one YAML file of schedules, and a hosted database.

---

## 1. Architecture

```
                     ┌──────────────────────────────────────────┐
                     │   Open-Meteo Air Quality + Weather APIs   │
                     │   (no key, hourly obs + 7-day forecast)   │
                     └────────────────────┬─────────────────────┘
                                          │ raw JSON
                                          ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │  GITHUB ACTIONS  (the only compute in the system)                   │
   │                                                                     │
   │   feature_pipeline.yml   hourly   →  src/pipelines/features.py      │
   │   training_pipeline.yml  daily    →  src/pipelines/train.py         │
   │   backfill.yml           manual   →  src/pipelines/backfill.py      │
   │   alerts.yml             6-hourly →  src/pipelines/alerts.py        │
   └────────────────────┬───────────────────────────────┬───────────────┘
                        │ features                      │ model artifact
                        ▼                               ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │  HOPSWORKS SERVERLESS  (free tier)                                   │
   │                                                                      │
   │   Feature Groups:  aqi_hourly_raw  ·  aqi_daily_features             │
   │   Feature View:    aqi_3day_fv     (the training/serving contract)   │
   │   Model Registry:  aqi_forecaster  (versioned, with metrics + schema)│
   └────────────────────┬───────────────────────────────┬────────────────┘
                        │ features                      │ model
                        ▼                               ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │  STREAMLIT COMMUNITY CLOUD  (free)                                   │
   │  3-day forecast · history · EDA · SHAP · model leaderboard · alerts  │
   └─────────────────────────────────────────────────────────────────────┘
```

**Why this shape.** The brief asks for a serverless stack, and the honest reading of "serverless" here is *no long-running process we operate*. GitHub Actions gives scheduled compute with no server. Hopsworks gives durable state — features and models — with no database to run. Streamlit Community Cloud gives a public URL with no web server to run. Airflow was offered in the brief as an alternative, but self-hosted Airflow needs a machine that stays up, which breaks the serverless requirement and adds cost. I will note that trade-off explicitly in the report rather than silently picking one.

---

## 2. Data source decision

The brief names AQICN and OpenWeather as examples and explicitly invites exploring others. I evaluated three and recommend a primary plus a cross-check.

| Source | Key needed | History depth | Forecast | Verdict |
|---|---|---|---|---|
| **Open-Meteo Air Quality** | No | Multi-year CAMS reanalysis archive; `past_days` up to 92 on the live endpoint | 7 days hourly | **Primary.** History and forecast from one unauthenticated API is decisive |
| **Open-Meteo Weather / Archive** | No | 1940 onward (ERA5) | 16 days | **Primary** for meteorological drivers |
| AQICN (WAQI) | Yes, free instant token | Current + limited recent | Limited | **Cross-check only.** Real ground-station readings, useful to validate the modelled CAMS values |
| OpenWeather Air Pollution | Yes | ~Nov 2020 onward | 4 days | **Documented fallback** if Open-Meteo archive proves too shallow for our city |

Open-Meteo is the primary because it solves the hardest problem in this project — **backfill**. AQICN's free tier will not hand you two years of hourly history, and without history there is no training set. Everything else follows from that.

**Day-1 verification task:** the global CAMS archive and the European CAMS reanalysis have *different* start dates. Before writing any modelling code we run one script that finds the true earliest available date for our chosen coordinates and records it. Everything downstream is sized off that number. Do not assume it.

---

## 3. The four design decisions that will decide whether this project is good or merely complete

Most submissions of this assignment work end-to-end and are still weak, because they get these four things wrong. Getting them right is the difference between a pipeline and a forecast.

### 3.1 Target definition — daily, direct, three horizons

"Predict AQI for the next 3 days" is ambiguous. We resolve it as:

> For each horizon *h* ∈ {1, 2, 3}, predict the **daily mean US AQI** and the **daily maximum US AQI** for the calendar day *t+h*, using only information available at the moment of prediction on day *t*.

We train **six independent regressors** (2 targets × 3 horizons) rather than one recursive hourly model. Recursive hourly forecasting means feeding a prediction back in as an input 72 times; the error compounds and by hour 72 the forecast is noise. Direct multi-horizon is more code but the errors are honest and each horizon can be evaluated separately. Daily max matters more than daily mean for health messaging, which is why we predict both.

### 3.2 Leakage — the trap that produces fake R² of 0.97

Two leakage paths will silently destroy this project if unguarded.

**Random cross-validation on autocorrelated time series.** Hourly AQI is enormously autocorrelated. A random K-fold split puts hour 14 in train and hour 15 in test; the model "learns" almost nothing and still scores brilliantly. **We never use `KFold` or `train_test_split(shuffle=True)`.** We use expanding-window walk-forward validation with a gap of at least *h* days between train end and validation start, and a strictly chronological final holdout of the last 90 days that is touched exactly once, at the very end.

**Weather feature skew between training and serving.** At serving time, our features for day *t+3* include the *weather forecast* for *t+3*, which is legitimately available. But when we backfill training data, the naive approach uses *observed* weather for *t+3* — which is perfect information the model will never have in production. Train on truth, serve on forecast, and the model degrades the moment it goes live.

Three ways to handle it, in descending order of quality:

- **Best:** use Open-Meteo's historical-forecast endpoint, which serves *past forecast runs*. Training features then contain forecast-quality weather exactly as production will. This is the correct fix and it is a genuine differentiator in the report.
- **Acceptable:** train on observed weather, then measure the degradation by evaluating the live model against outcomes for a few weeks and reporting the gap.
- **Also acceptable:** exclude future-dated weather entirely and forecast from persistence and seasonality alone. Weaker model, zero skew.

We attempt the best option and fall back with the reason documented. **This decision gets its own section in the final report.**

### 3.3 Baselines — the acceptance bar

A model is only worth deploying if it beats the dumb thing. We implement three baselines *before* any ML:

1. **Persistence** — tomorrow's AQI equals today's.
2. **Seasonal naive** — the value from the same day last week.
3. **Climatology** — the historical mean for this day-of-year, smoothed.

We report a **skill score** against persistence, `1 − RMSE_model / RMSE_persistence`, alongside RMSE, MAE and R².

> **Acceptance criterion for the whole project: the deployed model must beat persistence at h=1, h=2 and h=3 on the untouched holdout.** If it does not, the honest finding is that AQI at this location is persistence-dominated, and that finding is reported rather than hidden behind a high R². A high R² against a naive split is not evidence of anything.

### 3.4 Computing AQI correctly

Open-Meteo returns an hourly `us_aqi` convenience field, but the US EPA AQI is not really an hourly quantity — the official index uses a **24-hour average for PM2.5 and PM10**, an **8-hour rolling max for O3 and CO**, and **1-hour values for NO2 and SO2**, then takes the maximum sub-index across pollutants. We implement `src/features/aqi.py` with the real EPA breakpoint table and the correct averaging windows, and we validate our computed value against the API's field.

Two implementation details that break naive attempts:

- **Units.** Open-Meteo returns O3, NO2, SO2 and CO in µg/m³; the EPA breakpoints are in ppb (O3, NO2, SO2) and ppm (CO). Convert at 25 °C and 1 atm using `ppb = µg/m³ × 24.45 / MW` with MW = 48 (O3), 46 (NO2), 64 (SO2), 28 (CO).
- **Breakpoint currency.** The EPA revised the PM2.5 AQI breakpoints in 2024 (the "Good" band upper bound moved to 9.0 µg/m³). We pull the breakpoint table from the current EPA technical assistance document at build time and cite it — we do not copy a table from a blog post.

---

## 4. Repository layout

```
pearls-aqi-predictor/
├── README.md                        # architecture, live links, how to run
├── docs/
│   ├── PROJECT_PLAN.md              # this file
│   ├── REPORT.md                    # the graded final report, written as we go
│   ├── DATA_CARD.md                 # source, coverage, gaps, licence
│   └── MODEL_CARD.md                # intended use, metrics, limitations
├── .github/workflows/
│   ├── feature_pipeline.yml         # hourly
│   ├── training_pipeline.yml        # daily
│   ├── backfill.yml                 # manual dispatch, parameterised by date range
│   ├── alerts.yml                   # every 6 hours
│   └── ci.yml                       # lint + tests on every push
├── src/
│   ├── config.py                    # city, coords, horizons, thresholds — single source of truth
│   ├── clients/
│   │   ├── open_meteo.py            # typed wrapper, retry + backoff + response cache
│   │   └── aqicn.py                 # cross-check only
│   ├── features/
│   │   ├── aqi.py                   # EPA breakpoints, unit conversion, sub-index logic
│   │   ├── build.py                 # raw hourly → engineered daily feature frame
│   │   └── schema.py                # explicit dtypes; the feature contract
│   ├── store/
│   │   ├── hopsworks_store.py       # feature groups, feature view, model registry
│   │   └── local_store.py           # Parquet mirror — the offline fallback
│   ├── models/
│   │   ├── baselines.py             # persistence, seasonal naive, climatology
│   │   ├── sklearn_models.py        # Ridge, RandomForest, HistGradientBoosting
│   │   ├── deep.py                  # small LSTM / MLP in PyTorch
│   │   ├── statistical.py           # SARIMAX or Prophet
│   │   └── evaluate.py              # walk-forward CV, metrics, skill scores
│   ├── explain/shap_explain.py      # TreeExplainer, cached artefacts
│   └── pipelines/
│       ├── features.py              # hourly entry point
│       ├── backfill.py              # historical entry point
│       ├── train.py                 # daily entry point
│       └── alerts.py                # threshold checks + notification
├── app/
│   ├── streamlit_app.py             # the dashboard
│   └── pages/                       # EDA, model leaderboard, explanations
├── notebooks/
│   ├── 01_eda.ipynb                 # graded deliverable, written after backfill
│   └── 02_model_selection.ipynb     # the experiment log
├── tests/                           # pytest: AQI math, leakage guards, schema
├── requirements.txt
└── .env.example
```

**Principle:** every pipeline is a plain Python module runnable locally with `python -m src.pipelines.features`. GitHub Actions only ever calls those modules. Nothing about the system depends on Actions, so everything is debuggable on a laptop.

---

## 5. Feature specification

### 5.1 `aqi_hourly_raw` — feature group, primary key `(city_id, ts_utc)`

Raw pollutants (pm2_5, pm10, o3, no2, so2, co, dust, aerosol_optical_depth), raw weather (temperature_2m, relative_humidity_2m, dew_point_2m, wind_speed_10m, wind_direction_10m, wind_gusts_10m, surface_pressure, precipitation, cloud_cover, boundary_layer_height), plus our computed `us_aqi_epa` and the API's `us_aqi_api` for validation. Ingestion is idempotent — re-running the same hour overwrites rather than duplicates.

### 5.2 `aqi_daily_features` — feature group, primary key `(city_id, date)`

| Family | Features | Why |
|---|---|---|
| **Targets** | `aqi_mean_h1..h3`, `aqi_max_h1..h3` | Six regression targets |
| **Calendar** | hour, day-of-week, day-of-month, month, day-of-year, is_weekend, `sin/cos` of DOY and hour | Brief requires time features; cyclical encoding so December is adjacent to January |
| **Lags** | AQI at t−1, t−2, t−3, t−7, t−14 days | Persistence is the dominant signal; give the model direct access to it |
| **Rolling** | mean/std/min/max over 3, 7, 14, 30 days | Level and volatility regime |
| **Change rate** | `Δ` and `%Δ` over 1, 3, 7 days; rolling slope | Explicitly required by the brief |
| **Pollutant mix** | pm2_5/pm10 ratio, dominant-pollutant flag, each sub-index | Distinguishes dust events from combustion events — matters a lot in South Asia |
| **Meteorology, current** | temp, RH, wind speed/direction (as sin/cos), pressure, precip, boundary-layer height | Ventilation and washout drive dispersion |
| **Meteorology, forecast** | same variables at t+1, t+2, t+3 from the forecast API | The only genuinely forward-looking signal we have; see §3.2 for the skew handling |
| **Derived physics** | ventilation index (wind × BLH), stagnation flag, temperature inversion proxy, precip-in-last-24h flag | Cheap, physically motivated, usually high in SHAP rankings |

**Skipped deliberately:** holiday calendars (adds a dependency for marginal gain at this data volume) and traffic proxies (no free source). Both noted in the report as future work.

### 5.3 Feature view

One Hopsworks feature view `aqi_3day_fv` joins the two groups and is the *only* path by which both training and serving read features. This is what makes train/serve consistency structural rather than a matter of discipline. Training datasets are versioned so any experiment is reproducible.

---

## 6. Modelling protocol

**Data volume reality check.** If the archive yields roughly four years of hourly data, that is about 35,000 hourly rows but only about **1,500 daily rows**. For a next-3-day daily target, 1,500 samples is a small-data problem.

This has a consequence worth stating plainly up front: **a deep LSTM is very unlikely to win.** Gradient boosting on well-built features almost always beats a neural network at this sample size. We still build the LSTM because the brief asks for the statistical-to-deep-learning range, and because *demonstrating that we tested it and it lost* is a stronger result than pretending it won. That is a finding, and the report will present it as one.

**The ladder, evaluated identically:**

| Tier | Models | Purpose |
|---|---|---|
| 0 | Persistence, seasonal naive, climatology | The bar every other model must clear |
| 1 | Ridge, ElasticNet | Linear reference; interpretable coefficients |
| 2 | RandomForest, HistGradientBoosting, XGBoost/LightGBM | Expected winner |
| 3 | SARIMAX or Prophet | The "statistical modelling" leg of the brief; also handles seasonality explicitly |
| 4 | MLP and a small LSTM in PyTorch | The deep-learning leg; run on hourly sequences where the sample count is larger |

**Protocol.** Expanding-window walk-forward CV, minimum 5 folds, gap of *h* days between train and validation. Metrics per horizon: RMSE, MAE, R², plus skill vs persistence and **hazard-band accuracy** — how often we put the day in the right AQI category, which is what a user actually reads off the dashboard. Hyperparameter search is a small randomised search with a fixed seed, logged. Every run appends a row to `docs/experiments.csv`; that file is the experiment log and feeds the report and the dashboard leaderboard.

**Promotion rule.** The daily training job registers a new model version *only if* it beats the incumbent on the rolling validation window. Otherwise the incumbent stays and the run logs why. This prevents a bad data day from silently degrading the live forecast — and an automated pipeline that can only ever get worse is a liability, not an asset.

---

## 7. Automation

| Workflow | Schedule | Runtime budget | Failure behaviour |
|---|---|---|---|
| `feature_pipeline.yml` | hourly | < 2 min | Retry with backoff; open a GitHub issue on 3 consecutive failures |
| `training_pipeline.yml` | daily, 02:00 UTC | < 15 min | Keep incumbent model; alert |
| `alerts.yml` | every 6 hours | < 1 min | Silent retry |
| `backfill.yml` | manual dispatch | up to 6 h, chunked | Resumable — checkpoints per date chunk |
| `ci.yml` | on push | < 3 min | Blocks merge |

**Three GitHub Actions facts that will bite if ignored:**

1. Cron schedules are **best-effort, not guaranteed**. Delays of 5–20 minutes are normal and runs are occasionally skipped under platform load. The feature pipeline must therefore be **idempotent and self-healing**: on each run, fetch the last N hours rather than only the current hour, and upsert. Never assume the previous run happened.
2. Scheduled workflows are **auto-disabled after ~60 days of repository inactivity**. Either commit periodically or accept the pause. Worth a note in the README.
3. Actions minutes are **unlimited on public repositories** and metered on private ones. An hourly job burns roughly 700–800 runs a month, which is comfortable on a public repo and tight against a free private-repo allowance. **Recommendation: make the repository public.** It is a portfolio piece, and it removes the constraint entirely.

Secrets (`HOPSWORKS_API_KEY`, `AQICN_TOKEN`, alert credentials) live in GitHub Encrypted Secrets and Streamlit's secrets manager. Never in the repo. `.env.example` documents the names only.

---

## 8. Dashboard specification

Five pages, in build order:

1. **Forecast (landing).** Today's AQI as a large coloured band card; the 3-day forecast as three cards with predicted mean and max, category label and health guidance; a chart of the last 14 days observed plus 3 days forecast with a shaded uncertainty band; last-updated timestamp and the live model version.
2. **History & EDA.** Time series with range selector, monthly and hourly heatmaps, day-of-week and seasonal decomposition, pollutant correlation matrix, and an AQI-versus-weather scatter grid.
3. **Model performance.** The leaderboard from `experiments.csv`, per-horizon error charts, predicted-versus-actual plots, residual diagnostics, and the running skill score against persistence for the deployed model.
4. **Explainability.** SHAP global importance for the deployed model plus a per-day waterfall answering "why is tomorrow forecast to be bad?" SHAP values are precomputed in the training job and cached — computing them in the app will exceed Streamlit's memory limit.
5. **Alerts.** Current threshold configuration, alert history, and a subscribe control.

**Uncertainty is required, not optional.** A bare point forecast of "AQI 168" implies a precision we do not have. We show an interval, from quantile regression or the residual distribution per horizon. This is both better science and a better dashboard.

Community Cloud apps are memory-constrained and sleep after a period of inactivity. Consequences: cache aggressively with `@st.cache_data` and `@st.cache_resource`, never load the full hourly history into the app, precompute SHAP, and expect a cold-start delay on the first visit after idle. Verify the current tier limits before submission and note them in the report.

---

## 9. Alerting

Trigger when any of the next three days is forecast above a configured threshold — default 150 (Unhealthy) with a second tier at 200 — or when the forecast crosses a band upward relative to today.

Delivery options, all free: **email via SMTP** with an app password (universal, needs a credential), **ntfy.sh** (a push topic, no account, zero setup), **Slack or Discord webhook** (one URL), or a **GitHub issue** (zero config, good for the demo, poor as a real alert). *Decision needed from you — see §12.*

Guard against alert fatigue: de-duplicate so the same forecast day does not alert twice, and only re-alert on an upward band change.

---

## 10. Phased build plan

Each phase has a definition of done. No phase starts before the previous one meets it.

| Phase | Work | Done when |
|---|---|---|
| **P0 — Foundation** *(~half a day)* | Repo, structure, config, `.env.example`, CI, Hopsworks project, verify true archive start date for our coordinates | `python -m src.clients.open_meteo --probe` prints coverage; CI green |
| **P1 — AQI core** *(~half a day)* | EPA breakpoints, unit conversion, sub-index logic, unit tests against published worked examples | Our AQI matches the API's within tolerance; tests pass |
| **P2 — Feature pipeline** *(~1 day)* | Client with retry/backoff, raw→feature transform, schema enforcement, Hopsworks write, local Parquet mirror | One hour of data lands in the feature store; re-running does not duplicate |
| **P3 — Backfill** *(~1 day)* | Chunked, resumable historical run over the full archive; data-quality report on gaps and outliers | Feature store holds the full history; `DATA_CARD.md` written |
| **P4 — EDA** *(~1 day)* | `01_eda.ipynb`: seasonality, diurnal patterns, weather relationships, event detection, missingness | Notebook committed; findings feed feature design |
| **P5 — Modelling** *(~2 days)* | Baselines first, then tiers 1–4; walk-forward CV harness; experiment log; register winner | Deployed model beats persistence at all three horizons on the untouched holdout |
| **P6 — Automation** *(~1 day)* | All four workflows, secrets, idempotency, failure alerting, promotion rule | 48 hours of unattended green runs |
| **P7 — Dashboard** *(~2 days)* | Five pages, cached, deployed to Streamlit Cloud | Public URL live; loads in under 5 s warm |
| **P8 — Explainability & alerts** *(~1 day)* | SHAP precompute and pages, alert pipeline and de-duplication | Alert fires correctly on a forced test |
| **P9 — Report & polish** *(~1 day)* | `REPORT.md`, model card, README with architecture diagram, screenshots, demo recording | All four submission artefacts complete |

**Total: roughly 11–12 working days of focused effort.** Compressible to about 6 by trimming the model ladder to baselines plus gradient boosting plus one deep model, and the dashboard to three pages. *Which of these applies depends on your deadline — see §12.*

**Critical path:** P0 → P2 → P3 → P5. Everything else can slip without blocking. If time gets tight, protect the backfill above all else; without history there is no model, and backfill cannot be parallelised away.

---

## 11. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Sandbox cannot reach the APIs** (confirmed — 403 from the egress proxy in both the cloud container and the Cowork VM) | Certain | High | Run all network-touching code in GitHub Actions, or on your Mac outside the Cowork VM, or have the domains allowlisted. **Decision needed — §12.** I can write, test and reason about every line; I cannot execute a live fetch from here |
| Archive shallower than expected for our city | Medium | High | Verified in P0 before any modelling. Fallback: OpenWeather (Nov 2020+), or drop to hourly targets to raise sample count |
| Model fails to beat persistence | Medium | Medium | Report it honestly as a finding; pivot the headline metric to hazard-band classification accuracy, which is more forgiving and more useful |
| Hopsworks free tier limits or outage | Low | High | Local Parquet mirror written on every run; the system degrades to file-based and stays alive |
| Streamlit memory limit exceeded | Medium | Medium | Precompute SHAP, cache aggressively, never load full history in-app |
| Actions cron drift or skipped runs | High | Low | Idempotent overlapping fetch window; never assume the previous run happened |
| CAMS modelled values diverge from ground stations | Medium | Low | AQICN cross-check in EDA; document the difference in the data card rather than hiding it |

---

## 12. What I need from you

Nothing here blocks me from starting P0 except the first two.

### Blocking

**1. Which city, and its coordinates.** This is the single most consequential choice and it is yours, not mine. Karachi and Novo Hamburgo are both plausible from what I know of your situation, and they lead to different projects:

- **Karachi** — high AQI, large seasonal swing, genuinely hazardous days. Strong signal, a real audience for the alerts, and a more compelling demo. Global CAMS archive.
- **Novo Hamburgo (or anywhere in Europe)** — cleaner air means less variance, which makes the model's job harder and the alerts largely theoretical. European locations do get the richer, longer CAMS reanalysis.
- If your programme specified a city, that overrides both.

**2. How network-blocked code should run.** I confirmed the APIs are unreachable from both this sandbox and the Cowork VM on your Mac. Pick one:
- **(a)** I write everything, and the first real execution happens in GitHub Actions. Slower feedback loops, no setup from you. My default if you have no preference.
- **(b)** You run a command or two on your Mac outside the Cowork VM when I need live data. Fast, needs a few minutes of your time at a handful of points.
- **(c)** You get `open-meteo.com` and `hopsworks.ai` added to the session's egress allowlist, if that is something your account can do. Best of both.

### Needed before the phase that uses them

**3. GitHub.** Your username, whether the repo should be **public** (strongly recommended — unlimited Actions minutes and it doubles as a portfolio piece) or private, and how code should reach it: I can produce commits for you to push, or you can grant access.

**4. Hopsworks account.** Sign up free at `app.hopsworks.ai`, create a project, generate an API key. I need the **project name**; the **key goes into GitHub Secrets, not into this chat.**

**5. Deadline and rubric.** The submission date, and the marking rubric if one exists. This sets whether we build the 12-day version or the 6-day version, and a rubric tells me where the marks actually are — which is often not where the brief's emphasis is.

**6. Alert channel** (§9). ntfy.sh is my recommendation for zero setup; email if you want something that looks production-grade to a marker.

**7. Report format.** The brief asks for a "detailed report." Markdown in the repo, a Word document, or a PDF? If your programme has a template, send it.

### Nice to have

**8.** An AQICN token from `aqicn.org/data-platform/token/` — free and instant — for the ground-station cross-check.
**9.** Any constraint I should know about: required libraries, a mandated Python version, whether a team member is working on part of this, or anything your programme has already told you it wants to see.

---

## 13. Open questions I will resolve myself

Recorded so they are not forgotten, and so the report can show the reasoning:

- Exact archive start date for the chosen coordinates (P0)
- Whether the historical-forecast endpoint covers our archive window, deciding the §3.2 leakage strategy (P2)
- Whether daily-max is materially harder to predict than daily-mean, and whether both stay in scope (P5)
- Whether hourly-sequence models justify their complexity over daily tabular models (P5)
- Current Streamlit Community Cloud memory ceiling, verified rather than assumed (P7)

---

*This plan is a living document. Each phase updates it with what was actually found, so that `REPORT.md` can be assembled from evidence rather than reconstructed from memory at the end.*

---

# Build log — what changed when the plan met the code

*Appended 2026-09-05, after the implementation pass. The plan above is
unchanged; this section records where reality disagreed with it. Everything
below was verified against a synthetic Karachi-like series, since the
development sandbox has no egress to the APIs.*

## Decisions confirmed

- **Open-Meteo as primary.** Confirmed as the only free source with both
  archive and forecast and no API key.
- **Direct multi-horizon targets.** Implemented as six regressors. Holdout R²
  decays 0.81 → 0.64 → 0.63 across horizons, which is the realistic shape; a
  flat R² across horizons would have signalled leakage.
- **Persistence as the bar.** Confirmed necessary. On synthetic data the
  persistence RMSE at h=3 is roughly 100 AQI points, and the learned models cut
  it to 45 — a skill score of about 0.55. Without this comparison an R² of 0.63
  would have looked mediocre when it is in fact a large gain.
- **Physics features earn their place.** SHAP ranks `fc_ventilation_index_h1`
  and `fc_wind_speed_10m_mean_h1` immediately after the pollutant levels,
  confirming the model uses forecast meteorology rather than the calendar.

## Corrections

**1. The ozone unit bug — the exact trap §3.4 predicted, caught by a test.**
Ozone's EPA breakpoint table is in **ppm**, like CO, not ppb like NO2 and SO2.
The first implementation converted it to ppb, inflating every ozone sub-index a
thousandfold and making ozone the dominant pollutant on every single row. It
was caught by a test asserting that an hour with PM2.5 at 180 µg/m³ is
PM2.5-dominant. `EPA_UNITS` now makes the unit explicit per pollutant and a
regression test guards it.

**2. SHAP silently skipped the winning model.** The tree-model check tested for
`feature_importances_`. `HistGradientBoostingRegressor` is a tree ensemble fully
supported by TreeExplainer but does not expose that attribute, so the
explanations were being skipped for the model that most often wins. Detection is
now by type, with a permutation-importance fallback so the page is never blank.

**3. The daily 15-minute training budget was not achievable.** Measured at
roughly **110 seconds per target-horizon pair** on a two-core runner — the same
spec as a GitHub Actions runner — the quick ladder over six pairs takes about
12 minutes, and the full ladder with SARIMAX, torch and the hyperparameter
search is several times that. Resolved by splitting the schedule: the daily job
runs the quick ladder, and a new weekly `full_ladder.yml` runs the complete
comparison. The daily timeout is set to 90 minutes and the weekly to 300.

**4. Serverless state had no home.** The plan did not address where the Parquet
mirror lives between runs, and Actions runners are ephemeral. Solved with an
orphan `data` branch via `scripts/data_branch.sh` — no external service, no
binary churn in `main`, and the commits keep the schedules alive past GitHub's
60-day inactivity cutoff, which the plan flagged as a risk with no mitigation.

**5. torch could not be installed in the sandbox.** Its CDN is blocked by the
egress proxy. The deep tier is written but has not executed; it is isolated so a
failure there cannot break a training run, and it is confined to the weekly job.
**This is the largest untested surface in the codebase.**

## Additions not in the plan

- **Drifted persistence** as a fourth baseline. Plain persistence is a weak bar
  in a trending series; this one adds the damped weekly slope, so a model has to
  beat trend-following rather than just level-following.
- **A promotion rule with teeth.** The plan mentioned promotion; the
  implementation refuses to register a model that fails to beat persistence at
  any horizon, and refuses one more than 5% worse than the incumbent.
- **Live scoring.** Stored forecasts are graded against what actually happened,
  so the gap between backtest and live performance — where residual train/serve
  skew shows up — is measured rather than assumed.
- **Prediction intervals** from out-of-fold residual quantiles, shown on every
  forecast card.
- **94 tests**, including structural leakage guards that perturb future values
  and assert past features do not move.

## Still open

- First live API execution (Actions).
- True archive start date for Karachi — the probe answers this on first run.
- Whether the historical-forecast endpoint covers the full archive window,
  which decides the §3.2 leakage strategy.
- Whether the LSTM is competitive. Expected not, at ~1,500 daily rows.
- Real Hopsworks round-trip; the store code degrades gracefully but has only
  been exercised in its degraded path.
