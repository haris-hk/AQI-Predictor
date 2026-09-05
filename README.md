# Pearls AQI Predictor — Karachi

A three-day Air Quality Index forecast for Karachi, built end to end on a
100% serverless, zero-cost stack. GitHub Actions is the only compute,
Hopsworks holds the features and the models, and Streamlit Community Cloud
serves the dashboard. There is no server to run and nothing to pay for.

```
Open-Meteo (no API key)
        │  pollutants + weather, history and 7-day forecast
        ▼
GitHub Actions ── hourly features · daily training · weekly full ladder
        │
        ▼
Hopsworks ── feature groups · feature view · model registry
        │                                    (Parquet mirror as fallback)
        ▼
Streamlit ── forecast · EDA · leaderboard · SHAP · alerts
```

## Status

| Component | State |
|---|---|
| EPA AQI implementation | Done, 28 unit tests against worked examples |
| Feature pipeline | Done, verified on synthetic data |
| Backfill | Done, chunked and resumable |
| Model ladder | Done: 4 baselines, 5 sklearn, SARIMAX, MLP, LSTM |
| Evaluation harness | Done, walk-forward with leakage guards |
| Automation | 5 workflows written |
| Dashboard | 5 pages written |
| Alerts | Done, with de-duplication |
| **First live run** | **Pending — see Getting started** |

The entire system has been verified end to end against a synthetic
Karachi-like series (94 passing tests). It has not yet run against the live
API, because the development sandbox has no egress to Open-Meteo. The first
real execution happens in GitHub Actions.

## Getting started

### 1. Push and configure

```bash
git init && git add . && git commit -m "Initial commit"
gh repo create pearls-aqi-predictor --public --source=. --push
```

Make the repository **public**: Actions minutes are unlimited on public repos
and metered on private ones, and an hourly job is roughly 730 runs a month.

Add repository secrets under Settings → Secrets and variables → Actions:

| Secret | Required | Where to get it |
|---|---|---|
| `HOPSWORKS_API_KEY` | Recommended | [app.hopsworks.ai](https://app.hopsworks.ai) → Account Settings → API keys |
| `HOPSWORKS_PROJECT` | Recommended | The project name you created |
| `AQICN_TOKEN` | Optional | [aqicn.org/data-platform/token](https://aqicn.org/data-platform/token/) |
| `NTFY_TOPIC` | Optional | Any string; subscribe to it in the ntfy app |
| `SLACK_WEBHOOK_URL` | Optional | A Slack incoming webhook |
| `ALERT_EMAIL_TO`, `SMTP_USER`, `SMTP_PASSWORD` | Optional | Your mail provider's app password |

Without Hopsworks the system still runs — it falls back to the Parquet
mirror on the `data` branch — but you lose the managed feature store and
model registry that the brief asks for.

### 2. Probe and backfill

Run the **Historical backfill** workflow with `probe_first` enabled. The
probe finds the true earliest available date for Karachi's coordinates and
writes `data/coverage.json`; the backfill then pulls everything from there
to yesterday and writes `docs/DATA_CARD.md` from what it actually saw.

This is the critical path. Nothing downstream works until it succeeds.

### 3. Train and deploy

Run the **Training pipeline** workflow once by hand. Then deploy the app at
[share.streamlit.io](https://share.streamlit.io):

- Repository: this one, branch `main`
- Main file: `app/streamlit_app.py`
- Requirements: `requirements-app.txt`
- Advanced settings → Secrets: paste your `HOPSWORKS_API_KEY` and `HOPSWORKS_PROJECT`

The hourly and daily schedules take over from there.

### Local development

```bash
pip install -r requirements.txt
cp .env.example .env            # fill in what you have

python -m src.clients.open_meteo --probe    # verify archive coverage
python -m src.pipelines.backfill            # pull history (slow, chunked)
python -m src.pipelines.train --quick       # train
python -m src.pipelines.predict             # forecast
streamlit run app/streamlit_app.py

pytest tests -q                             # 94 tests, no network needed
```

## How state persists on ephemeral runners

Actions runners are destroyed after every job, so the Parquet mirror and the
model artefacts need somewhere durable. `scripts/data_branch.sh` keeps them
on an orphan `data` branch: no external service, no binary churn in `main`'s
history, and a useful side effect — the commits count as repository activity,
which stops GitHub from auto-disabling the scheduled workflows after 60 days
of quiet.

## Design decisions

Four choices determine whether this is a forecast or just plumbing. Each is
argued in full in [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) §3.

**Direct multi-horizon targets, not recursion.** Six independent regressors
(daily mean and daily max × 1, 2, 3 days). Predicting hour by hour and
feeding predictions back 72 times compounds error into noise by day three.

**Leakage is designed out, not checked for.** No random splits anywhere —
expanding-window walk-forward validation with a gap of at least the horizon
between train and validation, plus a 90-day chronological holdout touched
exactly once. `tests/test_features.py` asserts causality structurally: it
perturbs future values and asserts that past features do not move.

**Persistence is the acceptance bar.** AQI is highly autocorrelated, so "R²
of 0.9" proves nothing on its own. Every model is scored against persistence,
seasonal-naive and climatology, and the promotion rule refuses to deploy a
model that fails to beat persistence at all three horizons on the holdout.

**The AQI is computed properly.** The EPA index uses a 24-hour mean for
particulates, an 8-hour mean for ozone and CO, and 1-hour values for NO2 and
SO2, then takes the maximum sub-index. Open-Meteo returns gases in µg/m³
while the EPA breakpoints are in ppm (O3, CO) and ppb (NO2, SO2) — getting
that conversion wrong inflates the ozone sub-index a thousandfold. See
`src/features/aqi.py`.

## Repository layout

```
src/clients/       Open-Meteo and AQICN wrappers, retry + on-disk cache
src/features/      EPA AQI, feature engineering, schema enforcement
src/store/         Hopsworks store and the Parquet fallback mirror
src/models/        Baselines, sklearn, SARIMAX, torch, evaluation, bundle
src/explain/       SHAP, precomputed
src/pipelines/     features · backfill · train · predict · explain · alerts
app/               Streamlit dashboard, five pages
.github/workflows/ hourly · daily · weekly · manual backfill · CI
tests/             94 tests, synthetic fixtures, no network
docs/              Plan, data card, model card, report
```

## Known limitations

- Open-Meteo serves CAMS, a **model reanalysis**, not ground-station
  readings. The AQICN cross-check quantifies the disagreement; it does not
  eliminate it.
- Training features use forecast-quality weather where the historical
  forecast endpoint covers the range, and observed weather where it does
  not. The backfill records which regime was used in `DATA_CARD.md`, and any
  remaining train/serve skew shows up in the live scoring on the dashboard.
- With roughly four years of data there are only ~1,500 daily training rows.
  The deep models are included for completeness and are not expected to win.

## Licence

MIT.
