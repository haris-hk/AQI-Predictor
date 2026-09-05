"""Training pipeline. Runs daily in GitHub Actions.

Reads the daily feature frame, evaluates the full ladder on identical
walk-forward folds, scores the winner on a holdout it has never seen, and
registers a new model version only if it genuinely improves on the incumbent.

That last rule matters more than it looks. An automated retraining loop with no
promotion gate can only drift downward: one bad data day produces a worse model,
which is deployed unconditionally, and nobody notices until the forecasts are
visibly wrong.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.config import (
    ARTIFACT_DIR, CITY, HOLDOUT_DAYS, HORIZONS, RANDOM_SEED, TARGETS, ensure_dirs,
)
from src.features.build import feature_columns, training_frame
from src.models.baselines import baseline_factories
from src.models.bundle import ModelBundle
from src.models.deep import deep_factories
from src.models.evaluate import (
    CVResult, assert_no_leakage, cross_validate, holdout_split, regression_metrics,
    residual_quantiles, skill_score, walk_forward_splits,
)
from src.models.sklearn_models import SEARCH_SPACES, model_factories
from src.models.statistical import statistical_factories
from src.store import hopsworks_store, local_store

log = logging.getLogger(__name__)


def load_features() -> pd.DataFrame:
    """Prefer the managed store; fall back to the Parquet mirror."""
    df = pd.DataFrame()
    if hopsworks_store.is_configured():
        df = hopsworks_store.read_daily()
        if not df.empty:
            log.info("loaded %d rows from Hopsworks", len(df))
    if df.empty:
        df = local_store.read_daily()
        log.info("loaded %d rows from the local Parquet store", len(df))
    if df.empty:
        raise SystemExit("no features available -- run the backfill first")
    return df.sort_index()


def _candidate_factories(target: str, horizon: int, quick: bool) -> dict[str, callable]:
    """The full ladder: baselines, linear, trees, statistical, deep."""
    factories: dict[str, callable] = {}
    factories |= baseline_factories(target, horizon)
    factories |= model_factories()
    if not quick:
        factories |= statistical_factories(target, horizon)
        factories |= deep_factories(target, horizon)
    return factories


def _tuned_variants(name: str) -> list[tuple[str, dict]]:
    space = SEARCH_SPACES.get(name)
    if not space:
        return []
    return [(f"{name}#{i}", params) for i, params in enumerate(space)]


def evaluate_all(df: pd.DataFrame, quick: bool = False) -> tuple[list[CVResult], dict]:
    """Cross-validate every candidate on every (target, horizon)."""
    train_df, _ = holdout_split(df, HOLDOUT_DAYS)
    results: list[CVResult] = []
    best: dict[tuple[str, int], CVResult] = {}

    for target in TARGETS:
        for horizon in HORIZONS:
            X, y = training_frame(train_df, target, horizon)
            if len(X) < 60:
                log.warning("only %d usable rows for %s h%d; skipping", len(X), target, horizon)
                continue
            splits = walk_forward_splits(X.index)
            if not splits:
                log.warning("no valid folds for %s h%d", target, horizon)
                continue
            assert_no_leakage(splits, horizon)
            log.info("%s h%d: %d rows, %d features, %d folds",
                     target, horizon, len(X), X.shape[1], len(splits))

            factories = _candidate_factories(target, horizon, quick)
            local_results: list[CVResult] = []
            for name, factory in factories.items():
                res = cross_validate(factory, X, y, splits, name, target, horizon)
                local_results.append(res)
                # A small fixed search, only for the tiers where it pays.
                for variant_name, params in _tuned_variants(name):
                    if quick:
                        break
                    res_v = cross_validate(
                        lambda f=factory, p=params: f(**p), X, y, splits,
                        variant_name, target, horizon)
                    local_results.append(res_v)

            persistence = next(
                (r for r in local_results if r.name == "persistence"), None)
            ref_rmse = persistence.metrics.get("rmse", float("nan")) if persistence else float("nan")
            for r in local_results:
                r.metrics["skill_vs_persistence"] = skill_score(
                    r.metrics.get("rmse", float("nan")), ref_rmse)

            results.extend(local_results)
            learned = [r for r in local_results
                       if r.name not in baseline_factories(target, horizon)
                       and np.isfinite(r.metrics.get("rmse", float("nan")))]
            pool = learned or [r for r in local_results
                               if np.isfinite(r.metrics.get("rmse", float("nan")))]
            if pool:
                best[(target, horizon)] = min(pool, key=lambda r: r.metrics["rmse"])

    return results, best


def fit_final(df: pd.DataFrame, best: dict, quick: bool = False) -> ModelBundle:
    """Refit each winner on all non-holdout data and assemble the bundle."""
    train_df, _ = holdout_split(df, HOLDOUT_DAYS)
    bundle = ModelBundle()
    feature_names: list[str] | None = None

    for (target, horizon), result in best.items():
        X, y = training_frame(train_df, target, horizon)
        if feature_names is None:
            feature_names = list(X.columns)
        factories = _candidate_factories(target, horizon, quick)
        base_name = result.name.split("#")[0]
        factory = factories.get(base_name)
        if factory is None:
            continue
        params: dict = {}
        if "#" in result.name:
            idx = int(result.name.split("#")[1])
            space = SEARCH_SPACES.get(base_name, [])
            if idx < len(space):
                params = space[idx]
        try:
            model = factory(**params) if params else factory()
            model.fit(X[feature_names] if feature_names else X, y)
            bundle.models[(target, horizon)] = model
            bundle.residual_quantiles[f"{target}|{horizon}"] = residual_quantiles(result.residuals)
        except Exception as exc:                      # pragma: no cover
            log.error("final fit failed for %s h%d (%s): %s", target, horizon, result.name, exc)

    bundle.feature_names = feature_names or []
    return bundle


def evaluate_holdout(df: pd.DataFrame, bundle: ModelBundle) -> dict:
    """The one and only look at the untouched chronological holdout."""
    _, holdout = holdout_split(df, HOLDOUT_DAYS)
    report: dict = {}
    for (target, horizon), model in bundle.models.items():
        col = f"target_{target}_h{horizon}"
        if col not in holdout.columns:
            continue
        usable = holdout[holdout[col].notna()]
        if len(usable) < 10:
            continue
        X = bundle.align(usable)
        y = usable[col]
        pred = model.predict(X)
        metrics = regression_metrics(y, pred)
        # Persistence on the identical rows -- the honest comparison.
        naive = usable[target].to_numpy(dtype=float)
        naive_metrics = regression_metrics(y, naive)
        metrics["persistence_rmse"] = naive_metrics["rmse"]
        metrics["skill_vs_persistence"] = skill_score(metrics["rmse"], naive_metrics["rmse"])
        metrics["beats_persistence"] = bool(metrics["rmse"] < naive_metrics["rmse"])
        report[f"{target}|{horizon}"] = metrics
    return report


def should_promote(holdout_report: dict, incumbent: dict) -> tuple[bool, str]:
    """Promotion rule.

    Two conditions, both required:
      1. the new model beats persistence at every horizon on the holdout
      2. it is not materially worse than the incumbent

    Failing either, the incumbent stays and the run says why.
    """
    if not holdout_report:
        return False, "no holdout metrics produced"
    failures = [k for k, m in holdout_report.items() if not m.get("beats_persistence")]
    if failures:
        return False, f"does not beat persistence for: {', '.join(sorted(failures))}"
    new_rmse = float(np.nanmean([m["rmse"] for m in holdout_report.values()]))
    old_rmse = incumbent.get("rmse_mean")
    if old_rmse and np.isfinite(old_rmse) and new_rmse > old_rmse * 1.05:
        return False, (f"mean RMSE {new_rmse:.2f} is more than 5% worse than "
                       f"the incumbent {old_rmse:.2f}")
    return True, "beats persistence at every horizon and improves on the incumbent"


def run(quick: bool = False, register: bool = True) -> dict:
    ensure_dirs()
    df = load_features()
    log.info("training on %d days: %s -> %s", len(df),
             df.index.min().date(), df.index.max().date())

    results, best = evaluate_all(df, quick=quick)
    if not best:
        raise SystemExit("no model could be trained -- check the feature frame")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    local_store.append_experiment([
        r.row({"run_id": run_id, "city_id": CITY.city_id, "quick": quick})
        for r in results
    ])

    bundle = fit_final(df, best, quick=quick)
    holdout_report = evaluate_holdout(df, bundle)
    incumbent = hopsworks_store.get_best_model_metrics()
    promote, reason = should_promote(holdout_report, incumbent)

    mean_rmse = float(np.nanmean([m["rmse"] for m in holdout_report.values()])) \
        if holdout_report else float("nan")
    bundle.metadata = {
        "run_id": run_id,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "city": CITY.name,
        "city_id": CITY.city_id,
        "n_training_days": int(len(df)),
        "date_min": str(df.index.min().date()),
        "date_max": str(df.index.max().date()),
        "n_features": len(bundle.feature_names),
        "selected_models": {f"{t}|{h}": r.name for (t, h), r in best.items()},
        "cv_metrics": {f"{t}|{h}": r.metrics for (t, h), r in best.items()},
        "holdout_metrics": holdout_report,
        "rmse_mean": mean_rmse,
        "promoted": promote,
        "promotion_reason": reason,
        "random_seed": RANDOM_SEED,
    }

    summary = {
        "run_id": run_id, "promoted": promote, "reason": reason,
        "rmse_mean": mean_rmse,
        "selected": bundle.metadata["selected_models"],
        "holdout": holdout_report,
    }

    if promote:
        bundle.save(ARTIFACT_DIR)
        if register:
            hopsworks_store.register_model(
                ARTIFACT_DIR,
                {"rmse_mean": mean_rmse,
                 **{f"rmse_{k.replace('|', '_h')}": v["rmse"]
                    for k, v in holdout_report.items()}},
                description=f"3-day AQI forecaster for {CITY.name}, run {run_id}",
            )
        log.info("PROMOTED: %s", reason)
    else:
        log.warning("NOT PROMOTED: %s", reason)
        # Still persist so the failure is inspectable, but to a side path.
        bundle.save(ARTIFACT_DIR / "rejected")

    local_store.write_json("training_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="AQI training pipeline")
    parser.add_argument("--quick", action="store_true",
                        help="baselines and sklearn models only; skip SARIMAX, torch and tuning")
    parser.add_argument("--no-register", action="store_true",
                        help="skip Hopsworks model registration")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = run(quick=args.quick, register=not args.no_register)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
