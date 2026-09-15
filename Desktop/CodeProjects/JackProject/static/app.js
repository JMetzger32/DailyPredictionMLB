// DailyPredictionMLB — shared formatting helpers used across betting.html,
// accuracy.html, and other pages that render bet/accuracy stats.

function fmtRecord(stats) {
  if (!stats || stats.games === 0) return "No data";
  return `${stats.correct}-${stats.losses}`;
}

function fmtAccuracy(stats) {
  if (!stats || stats.games === 0) return "";
  return `${(stats.accuracy * 100).toFixed(1)}%  (${stats.games} games)`;
}

function fmtPL(pl) {
  if (pl === null || pl === undefined) return "—";
  const sign = pl >= 0 ? "+" : "−";
  return `${sign}$${Math.abs(pl).toFixed(2)}`;
}

function fmtOdds(ml) {
  if (ml === null || ml === undefined) return "—";
  return ml >= 0 ? `+${ml}` : `${ml}`;
}

function fmtEdge(edge) {
  if (edge === null || edge === undefined) return "—";
  const sign = edge >= 0 ? "+" : "";
  return `${sign}${(edge * 100).toFixed(1)}pp`;
}

// Market run line from the home team's perspective, e.g. "-1.5 (+120)".
// MLB run lines are always ±1.5; the sign says which side is laying it.
function fmtRunLine(b) {
  const pt = b.home_spread_point, ml = b.home_spread_ml;
  if (pt === null || pt === undefined) return "—";
  const ptStr = pt > 0 ? `+${pt}` : `${pt}`;
  return ml === null || ml === undefined ? ptStr : `${ptStr} (${fmtOdds(ml)})`;
}

function plColor(pl) {
  if (pl === null) return "";
  return pl >= 0 ? "text-success" : "text-danger";
}

function accuracyColor(pct) {
  if (pct >= 0.60) return "text-success";
  if (pct >= 0.50) return "text-warning";
  return "text-danger";
}

function barColor(pct) {
  if (pct >= 0.60) return "bg-success";
  if (pct >= 0.50) return "bg-warning";
  return "bg-danger";
}

function fmtStake(s) {
  if (s === null || s === undefined) return "—";
  return `$${s.toFixed(2)}`;
}
