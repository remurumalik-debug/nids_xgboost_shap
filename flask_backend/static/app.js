let META = null;
let SAMPLES = null;
let PIPELINE = null;
let SHAP_SUMMARY = null;
let currentValues = {};
let selectedSampleIdx = null;
let liveTimer = null;
let livePaused = false;

const GROUPS = [
  { name: "Basic Connection", cols: ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land"] },
  { name: "Content Features", cols: ["wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
      "root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files",
      "num_outbound_cmds","is_host_login","is_guest_login"] },
  { name: "Traffic (time-based)", cols: ["count","srv_count","serror_rate","srv_serror_rate","rerror_rate",
      "srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate"] },
  { name: "Host-based Traffic", cols: ["dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
      "dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
      "dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate"] },
];

const PIPE_ICONS = { db:"🗄", filter:"⏚", layers:"▤", cpu:"⚙", eye:"◉", check:"✓" };

async function init(){
  [META, SAMPLES, PIPELINE, SHAP_SUMMARY] = await Promise.all([
    fetch("/api/meta").then(r => r.json()),
    fetch("/api/samples").then(r => r.json()),
    fetch("/api/pipeline").then(r => r.json()),
    fetch("/api/shap_summary").then(r => r.json()),
  ]);

  setupTabs();

  // Overview
  renderMetricCards();
  renderPipeline();
  startLiveFeed();

  // Traffic analysis
  renderSampleGrid();
  renderFormGroups();
  if (SAMPLES.length) selectSample(0);
  document.querySelectorAll(".mode-btn").forEach(btn => {
    btn.addEventListener("click", () => switchMode(btn.dataset.mode));
  });
  document.getElementById("predict-btn").addEventListener("click", runPredict);

  // Training
  renderHyperparams();
  renderSplitChart();
  renderConfusionMatrix();
  renderAttackBreakdown();

  // SHAP page
  renderGlobalChart();
  renderBeeswarm();

  document.getElementById("live-toggle").addEventListener("click", toggleLive);
}

/* ============ TAB NAVIGATION ============ */
function setupTabs(){
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.toggle("active", b === btn));
      document.querySelectorAll(".tab-view").forEach(v => v.classList.add("hidden"));
      document.getElementById("tab-" + btn.dataset.tab).classList.remove("hidden");
    });
  });
}

/* ============ OVERVIEW: metric cards ============ */
function renderMetricCards(){
  const t = META.metrics.test;
  const cards = [
    { key:"accuracy", label:"Accuracy", icon:"◈", val:(t.accuracy*100).toFixed(1)+"%", sub:"XGBoost classifier" },
    { key:"precision", label:"Precision", icon:"◎", val:(t.precision*100).toFixed(1)+"%", sub:"attack detection" },
    { key:"recall", label:"Recall", icon:"▲", val:(t.recall*100).toFixed(1)+"%", sub:"threat coverage" },
    { key:"f1", label:"F1-Score", icon:"≈", val:(t.f1*100).toFixed(1)+"%", sub:"balanced measure" },
  ];
  document.getElementById("metric-cards").innerHTML = cards.map(c => `
    <div class="metric-card ${c.key}">
      <div class="mc-label">${c.icon} ${c.label}</div>
      <div class="mc-value">${c.val}</div>
      <div class="mc-sub">${c.sub}</div>
    </div>
  `).join("");
}

/* ============ OVERVIEW: pipeline ============ */
function renderPipeline(){
  const row = document.getElementById("pipeline-row");
  row.innerHTML = PIPELINE.map((s, i) => `
    ${i > 0 ? '<span class="pipe-arrow">→</span>' : ''}
    <div class="pipe-stage" data-idx="${i}">
      <div class="pipe-icon">${PIPE_ICONS[s.icon] || "•"}</div>
      <div class="pipe-title">${s.title}</div>
    </div>
  `).join("");
  row.querySelectorAll(".pipe-stage").forEach(el => {
    el.addEventListener("click", () => showPipelineDetail(parseInt(el.dataset.idx)));
  });
  showPipelineDetail(0);
}

function showPipelineDetail(idx){
  document.querySelectorAll(".pipe-stage").forEach((el, i) => el.classList.toggle("active", i === idx));
  const s = PIPELINE[idx];
  document.getElementById("pipeline-detail").innerHTML = `<b>${s.title}</b> — ${s.summary}<br><br>${s.detail}`;
}

/* ============ OVERVIEW: live feed ============ */
function startLiveFeed(){
  pushLiveRow(); // seed one immediately
  liveTimer = setInterval(() => { if (!livePaused) pushLiveRow(); }, 2500);
}

function toggleLive(){
  livePaused = !livePaused;
  document.getElementById("live-toggle").textContent = livePaused ? "▶ Resume" : "⏸ Pause";
}

async function pushLiveRow(){
  const s = SAMPLES[Math.floor(Math.random() * SAMPLES.length)];
  const payload = buildPayloadFromRecord(s);
  let result;
  try {
    result = await fetch("/api/predict", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload)
    }).then(r => r.json());
  } catch (e) { return; }

  const feed = document.getElementById("live-feed");
  const row = document.createElement("div");
  row.className = "live-row";
  const time = new Date().toLocaleTimeString();
  row.innerHTML = `
    <span class="lr-time">${time}</span>
    <span class="lr-flow">${s.protocol_type_label}/${s.service_label} · src ${s.src_bytes}B · ${s.flag_label}</span>
    <span class="lr-verdict ${result.prediction}">${result.prediction === "attack" ? "⚠ ATTACK" : "✓ NORMAL"} ${(result.probability*100).toFixed(0)}%</span>
  `;
  feed.appendChild(row);
  while (feed.children.length > 12) feed.removeChild(feed.firstChild);
}

/* ============ TRAFFIC ANALYSIS ============ */
function switchMode(mode){
  document.querySelectorAll(".mode-btn").forEach(b => b.classList.toggle("active", b.dataset.mode === mode));
  document.getElementById("sample-mode").classList.toggle("hidden", mode !== "sample");
  document.getElementById("manual-mode").classList.toggle("hidden", mode !== "manual");
}

function renderSampleGrid(){
  const grid = document.getElementById("sample-grid");
  grid.innerHTML = "";
  SAMPLES.forEach((s, idx) => {
    const isAttack = s.true_binary === 1;
    const card = document.createElement("div");
    card.className = "sample-card";
    card.dataset.idx = idx;
    card.innerHTML = `
      <div class="sc-left">
        <div class="sc-title">${s.protocol_type_label}/${s.service_label} · ${s.flag_label}</div>
        <div class="sc-meta">src ${s.src_bytes}B → dst ${s.dst_bytes}B · duration ${s.duration}s</div>
      </div>
      <span class="sample-tag ${isAttack ? 'attack' : 'normal'}">${s.true_label}</span>
    `;
    card.addEventListener("click", () => selectSample(idx));
    grid.appendChild(card);
  });
}

function selectSample(idx){
  selectedSampleIdx = idx;
  currentValues = { ...SAMPLES[idx] };
  document.querySelectorAll(".sample-card").forEach(c => {
    c.classList.toggle("selected", parseInt(c.dataset.idx) === idx);
  });
  syncManualFormFromValues();
}

function renderFormGroups(){
  const container = document.getElementById("form-groups");
  container.innerHTML = "";
  GROUPS.forEach((g, gi) => {
    const div = document.createElement("div");
    div.className = "form-group" + (gi === 0 ? " open" : "");
    const fieldsHtml = g.cols.map(col => fieldHtml(col)).join("");
    div.innerHTML = `
      <div class="form-group-head"><span>${g.name}</span><span class="chev">▸</span></div>
      <div class="form-group-body">${fieldsHtml}</div>
    `;
    div.querySelector(".form-group-head").addEventListener("click", () => {
      div.classList.toggle("open");
    });
    container.appendChild(div);
  });

  container.querySelectorAll("input,select").forEach(el => {
    el.addEventListener("input", () => {
      currentValues[el.dataset.col] = el.value;
      selectedSampleIdx = null;
      document.querySelectorAll(".sample-card").forEach(c => c.classList.remove("selected"));
    });
  });
}

function fieldHtml(col){
  const desc = META.descriptions[col] || "";
  if (META.categorical.includes(col)){
    const opts = META.category_options[col].map(o => `<option value="${o}">${o}</option>`).join("");
    return `<div class="field"><label title="${desc}">${col}</label>
      <select data-col="${col}">${opts}</select></div>`;
  }
  return `<div class="field"><label title="${desc}">${col}</label>
    <input data-col="${col}" type="number" step="any" value="0"></div>`;
}

function syncManualFormFromValues(){
  document.querySelectorAll("#form-groups [data-col]").forEach(el => {
    const col = el.dataset.col;
    let val = currentValues[col];
    if (META.categorical.includes(col)){
      val = currentValues[col + "_label"] ?? val;
    }
    if (val !== undefined) el.value = val;
  });
}

function buildPayloadFromRecord(rec){
  const payload = {};
  META.feature_cols.forEach(col => {
    payload[col] = META.categorical.includes(col) ? (rec[col + "_label"] ?? rec[col]) : (rec[col] ?? 0);
  });
  return payload;
}

function buildPayload(){
  const payload = buildPayloadFromRecord(currentValues);
  if (selectedSampleIdx === null){
    document.querySelectorAll("#form-groups [data-col]").forEach(el => {
      payload[el.dataset.col] = el.value;
    });
  }
  return payload;
}

async function runPredict(){
  const payload = buildPayload();
  const btn = document.getElementById("predict-btn");
  btn.textContent = "Running...";
  btn.disabled = true;

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    renderResult(data);
  } catch (e){
    console.error(e);
  } finally {
    btn.textContent = "Run Detection";
    btn.disabled = false;
  }
}

function renderResult(data){
  document.getElementById("empty-state").classList.add("hidden");
  document.getElementById("result-content").classList.remove("hidden");

  const badge = document.getElementById("verdict-badge");
  badge.textContent = data.prediction === "attack" ? "⚠ ATTACK" : "✓ NORMAL";
  badge.className = "verdict-badge " + data.prediction;

  document.getElementById("verdict-sub").textContent =
    `confidence ${(Math.max(data.probability, 1-data.probability)*100).toFixed(1)}%`;

  drawGauge(data.probability);
  document.getElementById("gauge-label").textContent = (data.probability*100).toFixed(1) + "%";

  document.getElementById("baseline-value").textContent = (data.base_value*100).toFixed(1) + "%";
  document.getElementById("baseline-fill").style.width = Math.min(data.base_value*100, 100) + "%";

  const truthBanner = document.getElementById("truth-banner");
  if (selectedSampleIdx !== null){
    const s = SAMPLES[selectedSampleIdx];
    const predBinary = data.prediction === "attack" ? 1 : 0;
    const correct = predBinary === s.true_binary;
    truthBanner.classList.remove("hidden");
    truthBanner.className = "truth-banner " + (correct ? "correct" : "incorrect");
    truthBanner.textContent = `Ground truth: ${s.true_label} — model was ${correct ? "correct" : "incorrect"}`;
  } else {
    truthBanner.classList.add("hidden");
  }

  renderShapChart(data.contributions);
}

function drawGauge(prob){
  const svg = document.getElementById("gauge-svg");
  const cx = 60, cy = 65, r = 50;
  const startAngle = Math.PI, endAngle = 0;
  const angle = startAngle + (endAngle - startAngle) * prob;

  function polar(cx, cy, r, angle){
    return [cx + r*Math.cos(angle), cy - r*Math.sin(angle)];
  }

  const bgStart = polar(cx, cy, r, Math.PI);
  const bgEnd = polar(cx, cy, r, 0);
  const needleEnd = polar(cx, cy, r-8, angle);
  const color = prob >= 0.5 ? "#ff6b6b" : "#4fd1c5";
  const fillEnd = polar(cx, cy, r, angle);

  svg.innerHTML = `
    <path d="M ${bgStart[0]} ${bgStart[1]} A ${r} ${r} 0 0 1 ${bgEnd[0]} ${bgEnd[1]}"
      fill="none" stroke="#232b38" stroke-width="10" stroke-linecap="round"/>
    <path d="M ${bgStart[0]} ${bgStart[1]} A ${r} ${r} 0 ${prob > 0.5 ? 1 : 0} 1 ${fillEnd[0]} ${fillEnd[1]}"
      fill="none" stroke="${color}" stroke-width="10" stroke-linecap="round"/>
    <circle cx="${cx}" cy="${cy}" r="3" fill="#8792a3"/>
    <line x1="${cx}" y1="${cy}" x2="${needleEnd[0]}" y2="${needleEnd[1]}" stroke="#e7ecf3" stroke-width="2"/>
  `;
}

function renderShapChart(contributions){
  const container = document.getElementById("shap-chart");
  container.innerHTML = "";
  const maxAbs = Math.max(...contributions.map(c => Math.abs(c.shap)), 0.0001);

  contributions.forEach(c => {
    const pct = (Math.abs(c.shap) / maxAbs) * 50;
    const isAttack = c.shap > 0;
    const row = document.createElement("div");
    row.className = "shap-row";
    row.innerHTML = `
      <div class="shap-feat" title="${META.descriptions[c.feature] || ''}"><b>${c.feature}</b> = ${formatVal(c.value)}</div>
      <div class="shap-track">
        <div class="shap-center"></div>
        <div class="shap-bar ${isAttack ? 'attack' : 'normal'}" style="width:${pct}%"></div>
      </div>
      <div class="shap-val">${c.shap > 0 ? '+' : ''}${c.shap.toFixed(3)}</div>
    `;
    container.appendChild(row);
  });
}

function formatVal(v){
  if (typeof v === "number") return Number.isInteger(v) ? v : v.toFixed(2);
  return v;
}

/* ============ MODEL TRAINING PAGE ============ */
function renderHyperparams(){
  const hp = META.metrics.hyperparameters;
  const rows = Object.entries(hp).map(([k, v]) => `
    <div class="kv-row"><span class="k">${k}</span><span class="v">${v}</span></div>
  `).join("");
  const extra = `
    <div class="kv-row"><span class="k">train rows</span><span class="v">${META.metrics.n_train.toLocaleString()}</span></div>
    <div class="kv-row"><span class="k">validation rows</span><span class="v">${(META.metrics.n_val||0).toLocaleString()}</span></div>
    <div class="kv-row"><span class="k">test rows</span><span class="v">${META.metrics.n_test.toLocaleString()}</span></div>
  `;
  document.getElementById("hyperparams").innerHTML = rows + extra;
}

function renderSplitChart(){
  const m = META.metrics;
  const metrics = ["accuracy","precision","recall","f1","auc"];
  const container = document.getElementById("split-chart");
  container.innerHTML = metrics.map(k => `
    <div class="split-metric">
      <div class="sm-label"><span>${k}</span></div>
      <div class="split-bars">
        <div class="split-bar train" style="width:${(m.train[k]*100).toFixed(1)}%" title="train ${m.train[k].toFixed(3)}"></div>
      </div>
      <div class="split-bars">
        <div class="split-bar val" style="width:${(m.val[k]*100).toFixed(1)}%" title="val ${m.val[k].toFixed(3)}"></div>
      </div>
      <div class="split-bars">
        <div class="split-bar test" style="width:${(m.test[k]*100).toFixed(1)}%" title="test ${m.test[k].toFixed(3)}"></div>
      </div>
    </div>
  `).join("") + `
    <div class="split-legend">
      <span><i style="background:#5a6474"></i>train</span>
      <span><i style="background:var(--amber)"></i>validation</span>
      <span><i style="background:var(--teal)"></i>test (held-out)</span>
    </div>
  `;
}

function renderConfusionMatrix(){
  const cm = META.metrics.confusion_matrix.matrix; // [[TN,FP],[FN,TP]]
  const [[tn, fp],[fn, tp]] = cm;
  const total = tn+fp+fn+tp;
  const cellStyle = (n, colorVar) => `background:${colorVar}; color:var(--text);`;
  document.getElementById("cm-wrap").innerHTML = `
    <div class="cm-grid">
      <div class="cm-cell" style="${cellStyle(tn,'var(--teal-dim)')}"><span class="cm-n">${tn}</span><span class="cm-l">True Normal</span></div>
      <div class="cm-cell" style="${cellStyle(fp,'var(--red-dim)')}"><span class="cm-n">${fp}</span><span class="cm-l">False Attack</span></div>
      <div class="cm-cell" style="${cellStyle(fn,'var(--red-dim)')}"><span class="cm-n">${fn}</span><span class="cm-l">Missed Attack</span></div>
      <div class="cm-cell" style="${cellStyle(tp,'var(--teal-dim)')}"><span class="cm-n">${tp}</span><span class="cm-l">True Attack</span></div>
    </div>
    <div class="cm-labels">
      Rows = predicted <b>normal</b> (top) / <b>attack</b> (bottom)<br>
      Cols = actual <b>normal</b> (left) / <b>attack</b> (right)<br>
      ${total.toLocaleString()} total test flows
    </div>
  `;
}

function renderAttackBreakdown(){
  const rows = META.metrics.attack_type_breakdown;
  const container = document.getElementById("attack-breakdown");
  container.innerHTML = rows.map(r => {
    const color = r.rate >= 0.9 ? "var(--teal)" : r.rate >= 0.5 ? "var(--amber)" : "var(--red)";
    return `
      <div class="ab-row">
        <div class="ab-name">${r.label}</div>
        <div class="ab-track"><div class="ab-fill" style="width:${(r.rate*100).toFixed(1)}%; background:${color};"></div></div>
        <div class="ab-count">${r.detected}/${r.count}</div>
        <div class="ab-rate" style="color:${color}">${(r.rate*100).toFixed(0)}%</div>
      </div>
    `;
  }).join("");
}

/* ============ SHAP EXPLAINABILITY PAGE ============ */
function renderGlobalChart(){
  const container = document.getElementById("global-chart");
  container.innerHTML = "";
  const top = META.global_importance.slice(0, 15);
  const maxImp = Math.max(...top.map(d => d.importance));
  top.forEach(d => {
    const row = document.createElement("div");
    row.className = "g-row";
    row.innerHTML = `
      <div class="g-feat" title="${META.descriptions[d.feature] || ''}">${d.feature}</div>
      <div class="g-track"><div class="g-fill" style="width:${(d.importance/maxImp*100).toFixed(1)}%"></div></div>
      <div class="g-val">${d.importance.toFixed(3)}</div>
    `;
    container.appendChild(row);
  });
}

function renderBeeswarm(){
  const container = document.getElementById("beeswarm-chart");
  container.innerHTML = "";
  const allVals = SHAP_SUMMARY.flatMap(f => f.shap_values.map(Math.abs));
  const maxAbs = Math.max(...allVals, 0.0001);

  SHAP_SUMMARY.forEach(f => {
    const row = document.createElement("div");
    row.className = "bee-row";
    const dots = f.shap_values.map((v, i) => {
      const pct = 50 + (v / maxAbs) * 48;
      const t = f.normalized_feature_values[i]; // 0..1
      const r = Math.round(79 + t * (255-79));
      const g = Math.round(209 + t * (180-209));
      const b = Math.round(197 + t * (84-197));
      return `<div class="bee-dot" style="left:${pct}%; background:rgb(${r},${g},${b})"></div>`;
    }).join("");
    row.innerHTML = `
      <div class="bee-label" title="${META.descriptions[f.feature] || ''}">${f.feature}</div>
      <div class="bee-plot"><div class="bee-center"></div>${dots}</div>
    `;
    container.appendChild(row);
  });
}

init();
