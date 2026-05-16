"use strict";

const PALETTE = [
  "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
  "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f",
];

const $ = (id) => document.getElementById(id);
const fmt = (n, d = 4) => Number(n).toFixed(d);

let priceChart = null;
let equityChart = null;

async function loadRuns() {
  const res = await fetch("/api/runs");
  if (!res.ok) {
    $("run-meta").textContent = `error: ${res.status}`;
    return;
  }
  const runs = await res.json();
  const select = $("run-select");
  select.innerHTML = "";
  for (const run of runs) {
    const opt = document.createElement("option");
    opt.value = run.run_id;
    const date = run.created_at?.replace("T", " ").slice(0, 19) ?? "?";
    opt.textContent = `${run.run_id.slice(0, 8)} · ${date} · ${run.matching_mode} · ${run.total_rounds} rounds`;
    select.appendChild(opt);
  }
  if (runs.length) {
    await loadRun(runs[0].run_id);
  } else {
    $("run-meta").textContent = "no runs in database yet";
  }
}

let currentDecisions = [];
let currentAgentMap = {};

async function loadRun(runId) {
  const [detail, prices, equity, decisions] = await Promise.all([
    fetch(`/api/runs/${runId}`).then((r) => r.json()),
    fetch(`/api/runs/${runId}/prices`).then((r) => r.json()),
    fetch(`/api/runs/${runId}/equity`).then((r) => r.json()),
    fetch(`/api/runs/${runId}/decisions`).then((r) => r.json()),
  ]);

  const run = detail.run;
  const agentMap = Object.fromEntries(detail.agents.map((a) => [a.agent_id, a]));
  currentAgentMap = agentMap;
  currentDecisions = decisions;
  $("run-meta").textContent =
    `matching=${run.matching_mode}  rounds=${run.total_rounds}  prompt=${(run.prompt_sha256 || "").slice(0, 8)}`;

  drawPrices(prices);
  drawEquity(equity, agentMap);
  drawLeaderboard(equity, agentMap, prices);
  populateAgentFilter(agentMap);
  drawDecisions();
}

function drawPrices(prices) {
  const labels = prices.map((p) => p.round_index);
  const clearing = prices.map((p) => p.clearing_price);
  const volume = prices.map((p) => p.cleared_volume);

  const data = {
    labels,
    datasets: [
      { label: "Clearing price", data: clearing, yAxisID: "y", borderColor: "#1f77b4", tension: 0.15, pointRadius: 0 },
      { label: "Cleared volume", data: volume, yAxisID: "y1", type: "bar", backgroundColor: "rgba(150,150,150,0.4)" },
    ],
  };
  const options = {
    responsive: true,
    interaction: { mode: "index", intersect: false },
    scales: {
      y: { position: "left", title: { display: true, text: "price (GC)" } },
      y1: { position: "right", grid: { drawOnChartArea: false }, title: { display: true, text: "volume" } },
    },
  };
  if (priceChart) priceChart.destroy();
  priceChart = new Chart($("price-chart"), { type: "line", data, options });
}

function drawEquity(equity, agentMap) {
  const agentIds = Object.keys(equity);
  if (!agentIds.length) return;
  const labels = equity[agentIds[0]].map((r) => r.round_index);
  const datasets = agentIds.map((aid, idx) => ({
    label: agentMap[aid]?.display_name ?? aid,
    data: equity[aid].map((r) => r.equity),
    borderColor: PALETTE[idx % PALETTE.length],
    backgroundColor: "transparent",
    tension: 0.15,
    pointRadius: 0,
  }));

  if (equityChart) equityChart.destroy();
  equityChart = new Chart($("equity-chart"), {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true,
      interaction: { mode: "index", intersect: false },
      scales: { y: { title: { display: true, text: "equity (GC)" } } },
    },
  });
}

function drawLeaderboard(equity, agentMap, prices) {
  const tbody = document.querySelector("#leaderboard tbody");
  tbody.innerHTML = "";
  const finalPrice = prices.length ? prices[prices.length - 1].clearing_price : 0;

  const rows = Object.entries(equity).map(([aid, series]) => {
    const last = series[series.length - 1] ?? { cash: 0, shares: 0, equity: 0 };
    return {
      agent_id: aid,
      display_name: agentMap[aid]?.display_name ?? aid,
      provider: agentMap[aid]?.provider ?? "?",
      cash: last.cash,
      shares: last.shares,
      equity: last.equity,
    };
  });
  rows.sort((a, b) => b.equity - a.equity);

  rows.forEach((r, idx) => {
    const tr = document.createElement("tr");
    if (idx === 0) tr.classList.add("winner");
    tr.innerHTML = `
      <td>${idx + 1}</td>
      <td>${r.display_name}</td>
      <td>${r.provider}</td>
      <td>${fmt(r.cash)}</td>
      <td>${r.shares}</td>
      <td>${fmt(r.equity)}</td>`;
    tbody.appendChild(tr);
  });
}

function populateAgentFilter(agentMap) {
  const sel = $("decision-filter");
  const current = sel.value;
  sel.innerHTML = '<option value="">all</option>';
  for (const [aid, agent] of Object.entries(agentMap)) {
    const opt = document.createElement("option");
    opt.value = aid;
    opt.textContent = agent.display_name ?? aid;
    sel.appendChild(opt);
  }
  if ([...sel.options].some((o) => o.value === current)) sel.value = current;
}

function drawDecisions() {
  const tbody = document.querySelector("#decisions tbody");
  tbody.innerHTML = "";
  const filter = $("decision-filter").value;
  const limit = parseInt($("decision-limit").value, 10);
  // Latest rounds first — usually more interesting than the early stretch.
  const rows = [...currentDecisions]
    .sort((a, b) => b.round_index - a.round_index || a.agent_id.localeCompare(b.agent_id))
    .filter((r) => !filter || r.agent_id === filter);
  const slice = limit > 0 ? rows.slice(0, limit) : rows;

  for (const d of slice) {
    const tr = document.createElement("tr");
    const name = currentAgentMap[d.agent_id]?.display_name ?? d.agent_id;
    const limitText = d.action === "HOLD" ? "—" : fmt(d.limit_price);
    const qtyText = d.action === "HOLD" ? "—" : d.quantity;
    const beforeText = d.price_before == null ? "—" : fmt(d.price_before);
    const afterText = d.price_after == null ? "—" : fmt(d.price_after);
    tr.innerHTML = `
      <td>${d.round_index + 1}</td>
      <td>${name}</td>
      <td><span class="act-${d.action}">${d.action}</span></td>
      <td>${qtyText}</td>
      <td>${limitText}</td>
      <td>${beforeText}</td>
      <td>${afterText}</td>
      <td>${(d.rationale ?? "").replace(/</g, "&lt;")}</td>`;
    tbody.appendChild(tr);
  }
}

$("run-select").addEventListener("change", (e) => loadRun(e.target.value));
$("decision-filter").addEventListener("change", drawDecisions);
$("decision-limit").addEventListener("change", drawDecisions);
loadRuns();
