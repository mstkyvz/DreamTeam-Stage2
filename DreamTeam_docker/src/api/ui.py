# -*- coding: utf-8 -*-
u"""The explainability page. One self-contained document: no CDN, no external font, no bundler.

Offline is not optional here. Solution mode runs with networking disabled, and explainability
mode is allowed network access *solely* for the ngrok tunnel -- a page that pulls a stylesheet
from a CDN would be both a rule violation and a blank screen behind a firewall.
"""
from __future__ import annotations

import json

PAGE = u"""<!doctype html>
<html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Relevance Predictor — Explainability</title>
<style>
:root{--bg:#0f1116;--panel:#171a21;--line:#262b36;--fg:#e6e9ef;--dim:#98a2b3;
 --irrelevant:#e5484d;--relevant:#30a46c;--semi:#f5a524;--accent:#6e8bff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 ui-sans-serif,system-ui,"Segoe UI",Roboto,sans-serif}
header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;gap:16px;align-items:baseline;flex-wrap:wrap}
header h1{font-size:15px;margin:0;font-weight:600;letter-spacing:.2px}
header .meta{color:var(--dim);font-size:12px}
main{display:grid;grid-template-columns:340px 1fr;gap:0;min-height:calc(100vh - 52px)}
@media(max-width:900px){main{grid-template-columns:1fr}}
aside{border-right:1px solid var(--line);padding:16px;overflow:auto;max-height:calc(100vh - 52px)}
section{padding:20px;overflow:auto}
label{display:block;font-size:12px;color:var(--dim);margin:12px 0 4px}
input,textarea,select{width:100%;background:#0c0e13;color:var(--fg);border:1px solid var(--line);
 border-radius:8px;padding:8px 10px;font:inherit}
textarea{resize:vertical;min-height:56px}
button{background:var(--accent);color:#fff;border:0;border-radius:8px;padding:9px 16px;
 font:inherit;font-weight:600;cursor:pointer;margin-top:14px;width:100%}
button:disabled{opacity:.5;cursor:progress}
ul.rows{list-style:none;margin:8px 0 0;padding:0;max-height:34vh;overflow:auto;border:1px solid var(--line);border-radius:8px}
ul.rows li{padding:7px 10px;cursor:pointer;border-bottom:1px solid #1d2129;font-family:ui-monospace,monospace;font-size:12px}
ul.rows li:hover{background:#1b1f28}
ul.rows li[aria-selected=true]{background:#243046;color:#fff}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:16px}
.card h2{margin:0 0 4px;font-size:13px;letter-spacing:.4px;text-transform:uppercase;color:var(--dim);font-weight:600}
.card p.hint{margin:0 0 14px;color:var(--dim);font-size:12px}
.verdict{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap}
.pill{font-weight:700;font-size:19px;padding:3px 12px;border-radius:999px}
.pill.irrelevant{background:rgba(229,72,77,.16);color:var(--irrelevant)}
.pill.relevant{background:rgba(48,164,108,.16);color:var(--relevant)}
.pill.semi-relevant{background:rgba(245,165,36,.16);color:var(--semi)}
.bar{display:grid;grid-template-columns:110px 1fr 62px;gap:10px;align-items:center;margin:7px 0;font-size:12px}
.track{height:9px;background:#0c0e13;border-radius:999px;overflow:hidden}
.fill{height:100%;border-radius:999px}
table{width:100%;border-collapse:collapse;font-size:12px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid #1d2129}
th{color:var(--dim);font-weight:600}
td.num{text-align:right;font-family:ui-monospace,monospace}
.delta{display:inline-block;min-width:120px;height:8px;position:relative;background:#0c0e13;border-radius:2px}
.delta i{position:absolute;top:0;height:100%;border-radius:2px}
.delta .zero{position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:#39404f}
.flip{color:var(--semi);font-weight:600}
.fields dt{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.3px;margin-top:9px}
.fields dd{margin:1px 0 0;word-break:break-word}
code{font-family:ui-monospace,monospace;background:#0c0e13;padding:1px 5px;border-radius:4px}
.err{color:var(--irrelevant)}
</style></head><body>
<header>
  <h1>Relevance Predictor — Explainability</h1>
  <span class="meta" id="meta">yükleniyor…</span>
</header>
<main>
<aside>
  <label for="filter">row_id ara</label>
  <input id="filter" placeholder="örn. 1042" autocomplete="off">
  <ul class="rows" id="rows"></ul>
  <label for="query">Sorgu (düzenlenebilir)</label>
  <textarea id="query" placeholder="önce bir row_id seçin"></textarea>
  <button id="explain" disabled>Açıkla</button>
  <div id="error" class="err" style="margin-top:10px;font-size:12px"></div>
</aside>
<section>
  <div class="card">
    <h2>Ürün alanları</h2>
    <p class="hint">Tahminciye giden görünür kanıt.</p>
    <dl class="fields" id="fields"><dd style="color:var(--dim)">Bir row_id seçin.</dd></dl>
  </div>
  <div class="card">
    <h2>Karar</h2>
    <p class="hint">Tahmincinin karar değişkeni: <code>{"etiket": "</code> ön-doldurulur, TEK token
      çözülür, ve 3-vektör her etiketi tekil olarak ön-ekleyen kelime dağarcığı token'larının
      marjinalidir. Aşağıdaki olasılıklar argmax'ın üzerinde alındığı vektörün ta kendisidir —
      sonradan eklenmiş bir güven skoru değil.</p>
    <div class="verdict" id="verdict"><span style="color:var(--dim)">—</span></div>
    <div id="posterior"></div>
    <p class="hint" id="marginNote" style="margin-top:12px"></p>
  </div>
  <div class="card">
    <h2>Alan katkıları — örtme (occlusion) ile</h2>
    <p class="hint">Her alan sırayla boşaltılıp satır <em>aynı</em> yoldan yeniden puanlanır;
      log-olasılıktaki değişim o alanın katkısıdır. Tek-özellik koalisyonlu Shapley tarzı bir
      atıf: vekil bir modelden kestirilmiş değil, konuşlandırıldığı haliyle model üzerinde
      <strong>kesin</strong>. Maliyeti alan başına tek bir <code>max_tokens=1</code> ileri geçiş.</p>
    <table id="attr"><tbody><tr><td style="color:var(--dim)">—</td></tr></tbody></table>
  </div>
  <div class="card">
    <h2>Nasıl karar verildi</h2>
    <p class="hint" id="pipelineNote"></p>
  </div>
</section>
</main>
<script>
const LABELS=["irrelevant","relevant","semi-relevant"];
let ROWS=[],current=null,META={};
const $=id=>document.getElementById(id);
const esc=s=>String(s==null?"":s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

fetch("/api/meta").then(r=>r.json()).then(m=>{META=m;
  $("meta").textContent=`${m.rows.toLocaleString("tr")} satır · ${m.stage} · ${m.model} · ${m.attention_impl||"?"}`;
  $("pipelineNote").innerHTML=m.pipeline_note;});
fetch("/api/rows").then(r=>r.json()).then(d=>{ROWS=d.row_ids;renderRows("");});

function renderRows(f){const q=f.trim().toLowerCase();
  const hit=(q?ROWS.filter(r=>r.toLowerCase().includes(q)):ROWS).slice(0,300);
  $("rows").innerHTML=hit.map(r=>`<li data-id="${esc(r)}" aria-selected="${r===current}">${esc(r)}</li>`).join("")
    ||'<li style="color:var(--dim);cursor:default">eşleşme yok</li>';}
$("filter").addEventListener("input",e=>renderRows(e.target.value));
$("rows").addEventListener("click",e=>{const li=e.target.closest("li[data-id]");if(li)select(li.dataset.id);});

function select(id){current=id;renderRows($("filter").value);$("error").textContent="";
  fetch("/api/item/"+encodeURIComponent(id)).then(r=>r.json()).then(row=>{
    $("query").value=row.query||"";$("explain").disabled=false;
    $("fields").innerHTML=Object.entries(row).filter(([k])=>k!=="row_id")
      .map(([k,v])=>`<dt>${esc(k)}</dt><dd>${esc(v)||'<span style="color:var(--dim)">—</span>'}</dd>`).join("");});}

$("explain").addEventListener("click",()=>{
  if(!current)return;const b=$("explain");b.disabled=true;b.textContent="Çalışıyor…";$("error").textContent="";
  fetch("/api/explain",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({row_id:current,query:$("query").value})})
   .then(r=>r.ok?r.json():r.json().then(e=>Promise.reject(e.detail||r.statusText)))
   .then(render).catch(e=>{$("error").textContent=String(e);})
   .finally(()=>{b.disabled=false;b.textContent="Açıkla";});});

function render(x){
  $("verdict").innerHTML=`<span class="pill ${x.label}">${x.label}</span>`+
    `<span style="color:var(--dim);font-size:12px">marj ${x.margin.toFixed(3)} nat`+
    (x.margin<0.35?" — kararsız, kademe bu satırı şampiyona yükseltirdi":"")+`</span>`;
  $("posterior").innerHTML=LABELS.map((l,i)=>{const p=x.posterior[i];
    return `<div class="bar"><span>${l}</span><span class="track">`+
      `<i class="fill" style="width:${(p*100).toFixed(1)}%;background:var(--${l})"></i></span>`+
      `<span class="num" style="text-align:right;font-family:ui-monospace,monospace">${p.toFixed(4)}</span></div>`;}).join("");
  $("marginNote").innerHTML=`Marj = en yüksek iki log-olasılığın farkı. Yönlendirici tam olarak `+
    `bu sayıya göre sıralar; en belirsiz %${(META.routed_fraction*100).toFixed(0)} şampiyona gider.`;

  const max=Math.max(0.05,...x.attributions.map(a=>Math.abs(a.delta_for_predicted)));
  $("attr").innerHTML=`<thead><tr><th>alan</th><th>Δ log-olasılık (tahmin edilen sınıf)</th>`+
    `<th class="num">Δ</th><th>alan olmadan</th></tr></thead><tbody>`+
    x.attributions.map(a=>{const d=a.delta_for_predicted,w=Math.abs(d)/max*50;
      const left=d>=0?50:50-w,col=d>=0?"var(--relevant)":"var(--irrelevant)";
      return `<tr><td>${esc(a.field_name)}</td>`+
        `<td><span class="delta"><span class="zero"></span>`+
        `<i style="left:${left}%;width:${w}%;background:${col}"></i></span></td>`+
        `<td class="num">${d>=0?"+":""}${d.toFixed(3)}</td>`+
        `<td class="${a.flipped?"flip":""}">${esc(a.occluded_label)}${a.flipped?" ⟵ karar değişti":""}</td></tr>`;
      }).join("")+`</tbody>`;}
</script></body></html>
"""


def render_page() -> str:
    return PAGE
