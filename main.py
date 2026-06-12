"""QuantBot entrypoint - live/paper trading loop.
Usage: python main.py"""
import logging
import yaml

from engine import Engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

if __name__ == "__main__":
    cfg = yaml.safe_load(open("config.yaml"))
    if cfg["trading"]["mode"] == "live" and cfg["exchange"].get("testnet") is False:
        print("=" * 60)
        print("คำเตือน: LIVE MODE + เงินจริง")
        print("แนะนำให้ผ่าน backtest และ paper trading ก่อนเสมอ")
        print("=" * 60)
        if input("พิมพ์ 'CONFIRM' เพื่อยืนยัน: ").strip() != "CONFIRM":
            raise SystemExit("ยกเลิก")
    Engine(cfg).run()
