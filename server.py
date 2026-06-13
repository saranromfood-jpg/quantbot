"""Render entrypoint: runs the trading bot in a background thread + serves the dashboard.
Secrets come from env vars: API_KEY, API_SECRET (set in Render dashboard)."""
import logging
import os
import threading

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("quantbot.server")


def load_cfg():
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")))
    # env overrides (Render dashboard -> Environment)
    cfg["exchange"]["api_key"] = os.environ.get("API_KEY", cfg["exchange"]["api_key"])
    cfg["exchange"]["api_secret"] = os.environ.get("API_SECRET", cfg["exchange"]["api_secret"])
    if "TESTNET" in os.environ:
        cfg["exchange"]["testnet"] = os.environ["TESTNET"].lower() == "true"
    if "TRADING_MODE" in os.environ:
        cfg["trading"]["mode"] = os.environ["TRADING_MODE"]  # paper | live
    return cfg


def run_bot(cfg):
    try:
        if cfg.get("portfolio", {}).get("enabled"):
            from portfolio_engine import PortfolioEngine
            PortfolioEngine(cfg).run()
        else:
            from engine import Engine
            Engine(cfg).run()
    except Exception:
        log.exception("bot crashed")


if __name__ == "__main__":
    cfg = load_cfg()
    if cfg["trading"]["mode"] == "live" and not cfg["exchange"]["testnet"]:
        log.warning("*** LIVE MODE + REAL MONEY ***")
    t = threading.Thread(target=run_bot, args=(cfg,), daemon=True, name="bot")
    t.start()
    from dashboard_app import app
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port)
