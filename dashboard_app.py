"""Real-time dashboard. Run: python dashboard_app.py  ->  http://127.0.0.1:8050"""
import json
import os

import yaml
from flask import Flask, jsonify, render_template_string

ROOT = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(ROOT, "state.json")
cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))

app = Flask(__name__)

PAGE = """<!doctype html><html lang="th"><head><meta charset="utf-8">
<title>QuantBot Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
 body{font-family:'Segoe UI',sans-serif;background:#0d1117;color:#e6edf3;margin:0;padding:20px}
 h1{font-size:20px;margin:0 0 16px} .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:20px}
 .card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px}
 .label{font-size:12px;color:#8b949e} .val{font-size:24px;font-weight:600;margin-top:4px}
 .green{color:#3fb950}.red{color:#f85149}.amber{color:#d29922}
 table{width:100%;border-collapse:collapse;font-size:13px} th,td{padding:6px 8px;text-align:left;border-bottom:1px solid #21262d}
 th{color:#8b949e;font-weight:500} .section{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px;margin-bottom:16px}
 #chartwrap{height:280px} .badge{padding:2px 8px;border-radius:10px;font-size:11px}
 .b-run{background:#1f6f43}.b-kill{background:#8e2c2c}
</style></head><body>
<h1>🤖 QuantBot <span id="status" class="badge b-run">-</span> <span class="label" id="updated"></span></h1>
<div class="grid">
 <div class="card"><div class="label">Equity (THB)</div><div class="val" id="equity">-</div></div>
 <div class="card"><div class="label">PnL รวม</div><div class="val" id="pnl">-</div></div>
 <div class="card"><div class="label">Cash</div><div class="val" id="cash">-</div></div>
 <div class="card"><div class="label">จำนวนเทรด</div><div class="val" id="ntrades">-</div></div>
 <div class="card"><div class="label">Win rate</div><div class="val" id="winrate">-</div></div>
</div>
<div class="section"><div class="label" style="margin-bottom:8px">Equity Curve</div><div id="chartwrap"><canvas id="chart"></canvas></div></div>
<div class="section"><div class="label" style="margin-bottom:8px">สถานะตลาด (Market Filter)</div><div id="market" style="font-size:15px">-</div></div>
<div class="section"><div class="label" style="margin-bottom:8px">สัญญาณรายเหรียญ</div><table id="signals"><thead><tr><th>เหรียญ</th><th>ราคา</th><th>เหนือ EMA200?</th><th>ถืออยู่?</th></tr></thead><tbody></tbody></table></div>
<div class="section"><div class="label" style="margin-bottom:8px">โพซิชันที่เปิดอยู่</div><table id="positions"><thead><tr><th>เหรียญ</th><th>ฝั่ง</th><th>จำนวน</th><th>เข้า</th><th>ราคาปัจจุบัน</th><th>Stop</th><th>Target</th></tr></thead><tbody></tbody></table></div>
<div class="section"><div class="label" style="margin-bottom:8px">ประวัติเทรดล่าสุด</div><table id="trades"><thead><tr><th>เหรียญ</th><th>ฝั่ง</th><th>เข้า</th><th>ออก</th><th>PnL</th><th>เหตุผล</th><th>เวลาปิด</th></tr></thead><tbody></tbody></table></div>
<script>
let chart;
function fmt(x){return x==null?'-':Number(x).toLocaleString(undefined,{maximumFractionDigits:2})}
async function refresh(){
 const r = await fetch('/api/state'); if(!r.ok) return;
 const s = await r.json(); if(!s) return;
 document.getElementById('equity').textContent = fmt(s.equity);
 const pnl = document.getElementById('pnl');
 pnl.textContent = (s.pnl_total>=0?'+':'')+fmt(s.pnl_total)+' ('+(s.pnl_pct>=0?'+':'')+s.pnl_pct+'%)';
 pnl.className = 'val '+(s.pnl_total>=0?'green':'red');
 document.getElementById('cash').textContent = fmt(s.cash);
 document.getElementById('ntrades').textContent = s.n_trades;
 document.getElementById('winrate').textContent = s.win_rate==null?'-':s.win_rate+'%';
 document.getElementById('updated').textContent = 'อัปเดต: '+(s.updated_at||'').slice(0,19);
 const st = document.getElementById('status');
 st.textContent = s.status; st.className = 'badge '+(s.status==='running'?'b-run':'b-kill');
 const mk = (s.signals||{})._market;
 const mkdiv = document.getElementById('market');
 if(mk){ mkdiv.innerHTML = mk.on
     ? `<span class="green">🟢 ตลาดกระทิง (BTC ${fmt(mk.btc)} > EMA200 ${fmt(mk.ema200)}) — เปิดเทรด</span>`
     : `<span class="amber">🟡 ตลาดหมี (BTC ${fmt(mk.btc)} < EMA200 ${fmt(mk.ema200)}) — หนีเข้าเงินสด</span>`; }
 let tb = document.querySelector('#signals tbody'); tb.innerHTML='';
 for(const [sym,v] of Object.entries(s.signals||{})){
   if(sym==='_market') continue;
   tb.innerHTML += `<tr><td>${sym}</td><td>${fmt(v.price)}</td>
     <td class="${v.above_ema200?'green':'red'}">${v.above_ema200?'ใช่':'ไม่'}</td>
     <td>${v.held?'✅':'-'}</td></tr>`;
 }
 tb = document.querySelector('#positions tbody'); tb.innerHTML='';
 for(const p of (s.positions||[])){
   tb.innerHTML += `<tr><td>${p.symbol}</td><td class="${p.side==='long'?'green':'red'}">${p.side}</td>
     <td>${p.qty.toFixed(6)}</td><td>${fmt(p.entry)}</td><td>${fmt(p.mark)}</td><td>${fmt(p.stop)}</td><td>${fmt(p.target)}</td></tr>`;
 }
 tb = document.querySelector('#trades tbody'); tb.innerHTML='';
 for(const t of (s.trades||[]).slice().reverse()){
   tb.innerHTML += `<tr><td>${t.symbol}</td><td>${t.side}</td><td>${fmt(t.entry)}</td><td>${fmt(t.exit)}</td>
     <td class="${t.pnl>=0?'green':'red'}">${(t.pnl>=0?'+':'')+fmt(t.pnl)}</td><td>${t.reason}</td><td>${t.closed_at.slice(0,19)}</td></tr>`;
 }
 const labels = (s.equity_curve||[]).map(x=>x[0].slice(5,16));
 const vals = (s.equity_curve||[]).map(x=>x[1]);
 if(!chart){
   chart = new Chart(document.getElementById('chart'),{type:'line',
     data:{labels,datasets:[{data:vals,borderColor:'#58a6ff',borderWidth:1.5,pointRadius:0,fill:true,backgroundColor:'rgba(88,166,255,.08)'}]},
     options:{maintainAspectRatio:false,plugins:{legend:{display:false}},
       scales:{x:{ticks:{color:'#8b949e',maxTicksLimit:10},grid:{color:'#21262d'}},
               y:{ticks:{color:'#8b949e'},grid:{color:'#21262d'}}}}});
 } else { chart.data.labels=labels; chart.data.datasets[0].data=vals; chart.update('none'); }
}
refresh(); setInterval(refresh, 5000);
</script></body></html>"""


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/api/state")
def state():
    if not os.path.exists(STATE):
        return jsonify(None)
    with open(STATE) as f:
        return jsonify(json.load(f))


if __name__ == "__main__":
    d = cfg.get("dashboard", {})
    app.run(host=d.get("host", "127.0.0.1"), port=d.get("port", 8050))
