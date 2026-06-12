import { useEffect, useState } from "react";
import { ChevronDown } from "lucide-react";
import { fetchStats } from "../api.js";
import CustomSelect from "../CustomSelect.jsx";
import { localizeTechnicalText } from "../helpers.js";

function teamLabel(team) {
  if (team === 0) return "Equipa A";
  if (team === 1) return "Equipa B";
  return "—";
}

function buildHeatmapOptions(stats) {
  if (!stats?.heatmaps) return [];
  const heatmaps = stats.heatmaps;
  const options = [
    { key: "global", label: "Global", file: heatmaps.global },
    { key: "team0", label: "Equipa A", file: heatmaps.team?.["0"] },
    { key: "team1", label: "Equipa B", file: heatmaps.team?.["1"] },
    { key: "ball", label: "Bola", file: heatmaps.ball }
  ].filter((option) => option.file);

  (heatmaps.players || []).forEach((player) => {
    options.push({
      key: `player_${player.tracker_id}`,
      label: `Jogador ${player.tracker_id}`,
      file: player.file
    });
  });
  return options;
}

export default function StatsSection({ job }) {
  const jobId = job?.job_id;
  const ready = job?.status === "succeeded" && Boolean(job?.stats_ready);
  const [stats, setStats] = useState(null);
  const [error, setError] = useState("");
  const [heatmap, setHeatmap] = useState("global");

  useEffect(() => {
    if (!ready || !jobId) return undefined;
    let active = true;
    fetchStats(jobId)
      .then((data) => {
        if (active) setStats(data);
      })
      .catch((err) => {
        if (active) setError(err.message);
      });
    return () => {
      active = false;
    };
  }, [ready, jobId]);

  if (!ready) return null;

  const possession = stats?.possession;
  const team0 = possession?.team?.["0"]?.pct ?? 0;
  const team1 = possession?.team?.["1"]?.pct ?? 0;
  const heatmapOptions = buildHeatmapOptions(stats);
  const selected = heatmapOptions.find((option) => option.key === heatmap) || heatmapOptions[0];
  const heatmapUrl = selected ? `/api/jobs/${jobId}/heatmap/${selected.file}` : null;

  return (
    <details className="panel stats-panel readiness-details" data-testid="stats">
      <summary className="readiness-summary">
        <div>
          <h2>Estatísticas</h2>
        </div>
        <div className="readiness-summary-state">
          <ChevronDown className="summary-chevron" size={18} />
        </div>
      </summary>

      <div className="readiness-body">
        {error ? <p className="notice error">{localizeTechnicalText(error)}</p> : null}

        {stats ? (
          <>
            <section className="stats-group">
              <header className="stats-group-head">
                <span className="eyebrow">Posse de bola</span>
                <small>Bola solta · {possession?.loose_pct ?? 0}%</small>
              </header>
              <div className="possession-bar">
                <span className="poss-team team-a" style={{ width: `${team0}%` }} />
                <span className="poss-team team-b" style={{ width: `${team1}%` }} />
              </div>
              <div className="possession-legend">
                <span className="team-tag team-a">Equipa A · {team0}%</span>
                <span className="team-tag team-b">Equipa B · {team1}%</span>
              </div>
            </section>

            <section className="stats-group">
              <span className="eyebrow">Jogadores</span>
              <table className="stats-table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Equipa</th>
                    <th>Dist. (km)</th>
                    <th>Vel. máx (km/h)</th>
                    <th>Posse (s)</th>
                  </tr>
                </thead>
                <tbody>
                  {(stats.players || []).map((player) => (
                    <tr key={player.tracker_id}>
                      <td>{player.tracker_id}</td>
                      <td>{teamLabel(player.team)}</td>
                      <td>{player.distance_km}</td>
                      <td>{player.max_speed_kmh}</td>
                      <td>{player.possession_seconds}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>

            <section className="stats-group">
              <span className="eyebrow">Mapa de calor</span>
              <CustomSelect
                ariaLabel="Selecionar heatmap"
                value={selected ? selected.key : "global"}
                onChange={(value) => setHeatmap(value)}
                options={heatmapOptions.map((option) => ({ value: option.key, label: option.label }))}
              />
              {heatmapUrl ? (
                <img className="heatmap-image" src={heatmapUrl} alt={`Heatmap ${selected.label}`} />
              ) : null}
            </section>
          </>
        ) : (
          <p className="muted">A carregar estatísticas…</p>
        )}
      </div>
    </details>
  );
}
