# QuantBot — Crypto Trading Bot สไตล์ Hedge Fund

บอทเทรด crypto อัตโนมัติ: ดึงข้อมูลเรียลไทม์ → วิเคราะห์ 3 กลยุทธ์ → รวมสัญญาณแบบ ensemble → คุมความเสี่ยง → ส่งคำสั่งเทรด พร้อม dashboard เรียลไทม์

## สถาปัตยกรรม

```
DataFeed (ccxt) → Indicators → Strategies ─┐
                                           ├→ Ensemble → RiskManager → Executor → Exchange
   TrendMomentum / MeanReversion / ML ─────┘                              │
                                              Portfolio → state.json → Dashboard
```

**กลยุทธ์ (แต่ละตัวให้คะแนน -1 ถึง +1 แล้วถ่วงน้ำหนักรวมกัน):**
- **TrendMomentum (45%)** — EMA 21/55 + MACD + RSI, เทรดเฉพาะตอน ADX > 20 (มีเทรนด์จริง)
- **MeanReversion (30%)** — z-score/Bollinger, เล่นเฉพาะตลาด sideways (ADX < 25)
- **MLFactor (25%)** — Gradient Boosting ทำนายทิศทางแท่งถัดไปจาก 9 ฟีเจอร์, เทรนใหม่ทุก 24 ชม. (walk-forward)

**Risk management (หัวใจของ hedge fund จริงๆ):**
- Position sizing ตามความผันผวน: เสี่ยง 1% ของพอร์ตต่อไม้
- Stop loss / take profit อิง ATR + trailing stop
- ขาดทุนวันละเกิน 3% → หยุดเทรดทั้งวัน
- Drawdown เกิน 10% → kill switch ปิดทุกโพซิชัน

## ติดตั้ง

```bash
pip install -r requirements.txt
```

## ใช้งาน (ตามลำดับนี้ — สำคัญมาก)

**1. Backtest** — ทดสอบกับข้อมูลย้อนหลังจริงก่อน
```bash
python backtest.py              # ดึงข้อมูลจริงจาก exchange
python backtest.py --synthetic  # ไม่มีเน็ต/ทดสอบระบบ
```

**2. Paper trading บน testnet** — สมัคร API key ที่ testnet.binance.vision ใส่ใน `config.yaml` แล้ว:
```bash
python main.py                  # config default คือ paper + testnet
```
รันคู่กับ dashboard:
```bash
python dashboard/app.py         # เปิด http://127.0.0.1:8050
```

**3. Live** — เมื่อผ่าน paper trading 2-4 สัปดาห์และผลเป็นบวกสม่ำเสมอ แก้ `config.yaml`:
```yaml
exchange.testnet: false
trading.mode: live
```
ใช้ API key จริง (ตั้งสิทธิ์ trade เท่านั้น **ปิด withdrawal เด็ดขาด**)

## ปรับแต่ง

ทุกพารามิเตอร์อยู่ใน `config.yaml`: เหรียญ, timeframe, น้ำหนักกลยุทธ์, เกณฑ์เข้า, ขนาดความเสี่ยง

## คำเตือน

การเทรด crypto มีความเสี่ยงสูงมาก ผล backtest ไม่การันตีผลในอนาคต บอทนี้เป็นเครื่องมือ ไม่ใช่คำแนะนำการลงทุน — เริ่มด้วยเงินที่เสียได้เท่านั้น และห้ามข้ามขั้น backtest/paper trading
