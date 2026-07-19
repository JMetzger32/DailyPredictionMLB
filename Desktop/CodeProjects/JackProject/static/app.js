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
