"""Hazard alerts.

Two design choices worth stating. First, an alert fires on a *deterioration*,
not merely on a high value -- in a Karachi winter the AQI sits above 150 for
weeks, and an alert that fires every six hours for a month is an alert nobody
reads. Second, state is persisted, so the same forecast day never alerts twice
at the same severity.

Delivery is pluggable and every channel is optional. With none configured the
alert is still written to the state file and surfaced on the dashboard.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage

import pandas as pd
import requests

from src.config import ALERT_THRESHOLDS, CITY, band_for, ensure_dirs
from src.store import local_store

log = logging.getLogger(__name__)

STATE_FILE = "alert_state.json"


def severity_of(aqi: float) -> str | None:
    for name in ("hazardous", "very_unhealthy", "unhealthy"):
        if aqi >= ALERT_THRESHOLDS[name]:
            return name
    return None


def build_alerts(forecast: pd.DataFrame, today_aqi: float | None) -> list[dict]:
    """One alert per forecast day that crosses a threshold, with a reason."""
    alerts: list[dict] = []
    today_sev = severity_of(today_aqi) if today_aqi is not None else None
    order = ["unhealthy", "very_unhealthy", "hazardous"]

    for target_date, row in forecast.iterrows():
        value = row.get("aqi_max", row.get("aqi_mean"))
        if value is None or pd.isna(value):
            continue
        sev = severity_of(float(value))
        if sev is None:
            continue
        # Only alert when the forecast is worse than today. A sustained bad
        # spell is not news after the first message.
        deteriorating = (
            today_sev is None or order.index(sev) > order.index(today_sev)
        )
        alerts.append({
            "target_date": str(pd.Timestamp(target_date).date()),
            "horizon": int(row.get("horizon", 0)),
            "aqi": float(value),
            "severity": sev,
            "band": band_for(float(value)).label,
            "guidance": band_for(float(value)).guidance,
            "deteriorating": bool(deteriorating),
        })
    return alerts


def deduplicate(alerts: list[dict], state: dict) -> list[dict]:
    """Suppress anything already sent at the same or higher severity."""
    order = {"unhealthy": 0, "very_unhealthy": 1, "hazardous": 2}
    sent = state.get("sent", {})
    fresh = []
    for alert in alerts:
        key = alert["target_date"]
        previous = sent.get(key)
        if previous is not None and order[alert["severity"]] <= order[previous]:
            continue
        if not alert["deteriorating"] and previous is not None:
            continue
        fresh.append(alert)
    return fresh


# ------------------------------------------------------------------- delivery
def send_ntfy(alerts: list[dict], topic: str) -> bool:
    try:
        body = "\n".join(
            f"{a['target_date']}: AQI {a['aqi']:.0f} ({a['band']})" for a in alerts)
        resp = requests.post(
            f"https://ntfy.sh/{topic}",
            data=body.encode("utf-8"),
            headers={
                "Title": f"Air quality alert for {CITY.name}",
                "Priority": "high" if any(a["severity"] != "unhealthy" for a in alerts) else "default",
                "Tags": "warning,fog",
            },
            timeout=20,
        )
        resp.raise_for_status()
        return True
    except requests.RequestException as exc:
        log.warning("ntfy delivery failed: %s", exc)
        return False


def send_slack(alerts: list[dict], webhook: str) -> bool:
    try:
        lines = [f"*Air quality alert for {CITY.name}*"]
        lines += [f"- {a['target_date']}: AQI *{a['aqi']:.0f}* ({a['band']}) - {a['guidance']}"
                  for a in alerts]
        resp = requests.post(webhook, json={"text": "\n".join(lines)}, timeout=20)
        resp.raise_for_status()
        return True
    except requests.RequestException as exc:
        log.warning("Slack delivery failed: %s", exc)
        return False


def send_email(alerts: list[dict], to_addr: str) -> bool:
    host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER", "")
    password = os.getenv("SMTP_PASSWORD", "")
    if not user or not password:
        log.info("SMTP credentials not set; skipping email")
        return False
    msg = EmailMessage()
    msg["Subject"] = f"Air quality alert for {CITY.name}"
    msg["From"] = user
    msg["To"] = to_addr
    msg.set_content("\n".join(
        f"{a['target_date']}: AQI {a['aqi']:.0f} ({a['band']})\n  {a['guidance']}\n"
        for a in alerts))
    try:
        with smtplib.SMTP(host, port, timeout=30) as server:
            server.starttls()
            server.login(user, password)
            server.send_message(msg)
        return True
    except Exception as exc:                          # pragma: no cover
        log.warning("email delivery failed: %s", exc)
        return False


def dispatch(alerts: list[dict]) -> dict[str, bool]:
    channels: dict[str, bool] = {}
    if topic := os.getenv("NTFY_TOPIC"):
        channels["ntfy"] = send_ntfy(alerts, topic)
    if webhook := os.getenv("SLACK_WEBHOOK_URL"):
        channels["slack"] = send_slack(alerts, webhook)
    if to_addr := os.getenv("ALERT_EMAIL_TO"):
        channels["email"] = send_email(alerts, to_addr)
    if not channels:
        log.info("no alert channel configured; alert recorded to state only")
    return channels


def run(force: bool = False, dry_run: bool = False) -> dict:
    ensure_dirs()
    forecast = local_store.read_predictions()
    features = local_store.read_daily()
    if forecast.empty:
        return {"status": "no forecast available"}

    latest_run = forecast["predicted_at"].max() if "predicted_at" in forecast else None
    if latest_run is not None:
        forecast = forecast[forecast["predicted_at"] == latest_run]

    today_aqi = None
    if not features.empty:
        observed = features["aqi_max"].dropna()
        today_aqi = float(observed.iloc[-1]) if len(observed) else None

    alerts = build_alerts(forecast, today_aqi)
    state = local_store.read_json(STATE_FILE) or {"sent": {}, "history": []}
    to_send = alerts if force else deduplicate(alerts, state)

    delivered: dict[str, bool] = {}
    if to_send and not dry_run:
        delivered = dispatch(to_send)
        for alert in to_send:
            state.setdefault("sent", {})[alert["target_date"]] = alert["severity"]
        state.setdefault("history", []).append({
            "at": datetime.now(timezone.utc).isoformat(),
            "alerts": to_send, "channels": delivered,
        })
        state["history"] = state["history"][-200:]
        local_store.write_json(STATE_FILE, state)

    summary = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "today_aqi": today_aqi,
        "candidates": alerts,
        "sent": to_send,
        "channels": delivered,
        "suppressed": len(alerts) - len(to_send),
    }
    log.info("%d candidate alerts, %d sent, %d suppressed as duplicates",
             len(alerts), len(to_send), summary["suppressed"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Hazardous AQI alerts")
    parser.add_argument("--force", action="store_true", help="ignore de-duplication")
    parser.add_argument("--dry-run", action="store_true", help="evaluate but do not send")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(json.dumps(run(args.force, args.dry_run), indent=2, default=str))


if __name__ == "__main__":
    main()
