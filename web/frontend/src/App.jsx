import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ChevronDown,
  CheckCircle2,
  Cpu,
  Download,
  FileVideo,
  Gauge,
  Layers3,
  Loader2,
  Moon,
  Play,
  Radio,
  RefreshCw,
  Settings2,
  SlidersHorizontal,
  StopCircle,
  Sun,
  Upload
} from "lucide-react";
import { cancelJob, createJob, fetchJob, fetchStats, fetchSystem } from "./api.js";
import CustomSelect from "./CustomSelect.jsx";

// Carregado sob demanda: evita puxar three.js até o utilizador abrir a vista 3D
const Match3DPanel = lazy(() => import("./Match3DViewer.jsx"));

const FALLBACK_DEFAULTS = {
  mode: "RADAR",
  device: "cuda",
  debug: false,
  player_track_imgsz: 1024,
  pitch_every_n_frames: 5,
  ball_track_imgsz: 960,
  ball_track_every_n_frames: 2,
  ball_track_conf: 0.25,
  ball_max_hold_frames: 3
};

const TERMINAL_STATUSES = new Set(["succeeded", "failed", "cancelled"]);

const MODE_LABELS = {
  PITCH_DETECTION: "Campo",
  PLAYER_DETECTION: "Jogadores",
  BALL_DETECTION: "Bola",
  PLAYER_TRACKING: "Seguimento",
  TEAM_CLASSIFICATION: "Equipas",
  RADAR: "Radar"
};

const PIPELINE_STAGES = ["Carregar", "Detetar", "Seguir", "Radar", "Exportar"];

const STATUS_LABELS = {
  idle: "A aguardar",
  queued: "Em fila",
  running: "Em processamento",
  succeeded: "Concluído",
  failed: "Falhou",
  cancelled: "Cancelado"
};

function getInitialTheme() {
  if (typeof window === "undefined") return "dark";

  try {
    const storedTheme = window.localStorage.getItem("footar-theme");
    if (storedTheme === "light" || storedTheme === "dark") return storedTheme;
  } catch {
    // Ignora limitações de storage em navegação privada ou testes.
  }

  return window.matchMedia?.("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

function formatBytes(bytes) {
  if (!bytes) return "0 MB";
  const mb = bytes / (1024 * 1024);
  return `${mb.toFixed(mb >= 10 ? 0 : 1)} MB`;
}

function jobPercent(job) {
  const progress = Number(job?.progress || 0);
  return Math.round(Math.max(0, Math.min(1, progress)) * 100);
}

function statusLabel(status) {
  return STATUS_LABELS[status || "idle"] || status || STATUS_LABELS.idle;
}

function localizeTechnicalText(text) {
  if (!text) return "";

  return String(text)
    .replace(/Job queued/gi, "Processamento em fila")
    .replace(/Processor exited with code/gi, "Processador terminou com código")
    .replace(/ModuleNotFoundError: No module named/gi, "Modulo Python em falta")
    .replace(/Processing/gi, "A processar")
    .replace(/Running/gi, "A executar")
    .replace(/Started/gi, "Iniciado")
    .replace(/Completed/gi, "Concluído")
    .replace(/Finished/gi, "Terminado")
    .replace(/Saving/gi, "A guardar")
    .replace(/Saved/gi, "Guardado")
    .replace(/\bOutput\b/gi, "Resultado")
    .replace(/Error/gi, "Erro")
    .replace(/Failed/gi, "Falhou")
    .replace(/frames/gi, "fotogramas")
    .replace(/frame/gi, "fotograma")
    .replace(/missing/gi, "em falta");
}

function parseFootarEvent(text) {
  const match = String(text || "").match(/FOOTAR_EVENT\s+({.*})/);
  if (!match) return null;

  try {
    return JSON.parse(match[1]);
  } catch {
    return null;
  }
}

function numericValue(source, keys) {
  for (const key of keys) {
    const value = Number(source?.[key]);
    if (Number.isFinite(value)) return value;
  }

  return null;
}

function progressEvent(processed, total, progress) {
  const safeProcessed = Number.isFinite(processed) ? Math.round(processed) : null;
  const safeTotal = Number.isFinite(total) ? Math.round(total) : null;
  const safeProgress = Number.isFinite(progress) ? Math.round(Math.max(0, Math.min(1, progress)) * 100) : null;
  const percentText = safeProgress === null ? "" : ` (${safeProgress}%)`;

  return {
    tone: "active",
    title: "Análise em curso",
    body:
      safeProcessed !== null && safeTotal !== null
        ? `${safeProcessed} de ${safeTotal} fotogramas analisados${percentText}.`
        : "A ler o vídeo e a atualizar o progresso da análise.",
    meta: "Visão"
  };
}

function footarEventToCard(payload) {
  if (!payload?.event) return null;

  const eventName = String(payload.event).toLowerCase();
  if (eventName === "progress") {
    const processed = numericValue(payload, ["processed_frames", "processed", "frame"]);
    const total = numericValue(payload, ["total_frames", "total"]);
    const progress = numericValue(payload, ["progress"]);
    return progressEvent(processed, total, progress);
  }

  if (eventName === "completed" || eventName === "complete" || eventName === "output") {
    return {
      tone: "done",
      title: "Resultado pronto",
      body: "O vídeo processado já está disponível para pré-visualização.",
      meta: "Exportação"
    };
  }

  if (eventName === "error" || eventName === "failed") {
    return {
      tone: "issue",
      title: "Atenção necessária",
      body: "O processamento encontrou um problema.",
      meta: "Erro"
    };
  }

  return null;
}

function logToEvent(line, jobStatus) {
  const text = String(line || "");
  const lower = text.toLowerCase();
  const footarEvent = footarEventToCard(parseFootarEvent(text));
  if (footarEvent) return footarEvent;

  const processedMatch = text.match(/processed\s+(\d+)\s*\/\s*(\d+)\s+frames/i);

  if (processedMatch) {
    return progressEvent(Number(processedMatch[1]), Number(processedMatch[2]), null);
  }

  if (lower.includes("job queued")) {
    return {
      tone: "queued",
      title: "Vídeo recebido",
      body: "O processamento entrou na fila e está pronto para arrancar.",
      meta: "Fila"
    };
  }

  if (lower.includes("rendering radar")) {
    return {
      tone: "active",
      title: "Radar em construção",
      body: "A compor a vista tática do lance.",
      meta: "Radar"
    };
  }

  if (lower.includes("pitch") || lower.includes("homography")) {
    return {
      tone: "active",
      title: "Campo calibrado",
      body: "A alinhar as marcações do relvado com o vídeo.",
      meta: "Campo"
    };
  }

  if (lower.includes("ball")) {
    return {
      tone: "active",
      title: "Bola em análise",
      body: "A validar a posição da bola ao longo da jogada.",
      meta: "Bola"
    };
  }

  if (lower.includes("track")) {
    return {
      tone: "active",
      title: "Jogadores em seguimento",
      body: "A ligar movimentos dos jogadores entre fotogramas.",
      meta: "Seguimento"
    };
  }

  if (lower.includes("team")) {
    return {
      tone: "active",
      title: "Equipas em leitura",
      body: "A separar jogadores por equipa quando há informação suficiente.",
      meta: "Equipas"
    };
  }

  if (lower.includes("error") || lower.includes("failed") || jobStatus === "failed") {
    return {
      tone: "issue",
      title: "Atenção necessária",
      body: localizeTechnicalText(text),
      meta: "Erro"
    };
  }

  if (lower.includes("saved") || lower.includes("output") || lower.includes("completed") || lower.includes("finished")) {
    return {
      tone: "done",
      title: "Resultado a ser preparado",
      body: "A guardar o vídeo processado para pré-visualização e descarga.",
      meta: "Exportação"
    };
  }

  return null;
}

function uniqueEvents(events) {
  const seen = new Set();
  return events.filter((event) => {
    if (!event) return false;
    const key = `${event.title}-${event.meta}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function processingEvents(job) {
  const logs = job?.logs || [];

  if (logs.length > 0) {
    const parsed = logs.map((line) => logToEvent(line, job?.status)).filter(Boolean);
    const latestProgress = [...parsed].reverse().find((event) => event.title === "Análise em curso");
    const contextEvents = uniqueEvents(parsed.filter((event) => event.title !== "Análise em curso")).slice(-2);
    const readableEvents = [latestProgress, ...contextEvents].filter(Boolean);

    if (readableEvents.length > 0) return readableEvents;
  }

  if (job?.status === "succeeded") {
    return [
      {
        tone: "done",
        title: "Processamento concluído",
        body: "O vídeo está pronto para pré-visualização e para descarregar.",
        meta: "Pronto"
      }
    ];
  }

  if (job?.status === "failed") {
    return [
      {
        tone: "issue",
        title: "Processamento interrompido",
        body: localizeTechnicalText(job.error) || "Ocorreu um erro durante a análise.",
        meta: "Erro"
      }
    ];
  }

  if (job?.status === "running") {
    if (job.processed_frames || job.total_frames) {
      return [progressEvent(Number(job.processed_frames), Number(job.total_frames), Number(job.progress))];
    }

    return [
      {
        tone: "active",
        title: "Processamento em curso",
        body: "A analisar o vídeo e a preparar a vista tática.",
        meta: "Em direto"
      }
    ];
  }

  if (job?.status === "queued") {
    return [
      {
        tone: "queued",
        title: "Vídeo recebido",
        body: "O processamento entrou na fila e está pronto para arrancar.",
        meta: "Fila"
      }
    ];
  }

  return [
    {
      tone: "idle",
      title: "À espera do apito inicial",
      body: "Carrega um vídeo e inicia o processamento para veres o relato em tempo real.",
      meta: "Pronto"
    }
  ];
}

function normalizeDefaults(system) {
  const defaults = { ...FALLBACK_DEFAULTS, ...(system?.defaults || {}) };
  if (system && !system.cuda?.available && defaults.device === "cuda") {
    defaults.device = "cpu";
  }
  return defaults;
}

function Field({ label, children }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
    </label>
  );
}

function SelectField({ label, value, options, onChange, disabled }) {
  return (
    <div className="field">
      <span>{label}</span>
      <CustomSelect
        ariaLabel={label}
        value={value}
        options={options}
        onChange={onChange}
        disabled={disabled}
      />
    </div>
  );
}

function MetricPill({ icon, label, value, tone = "neutral" }) {
  return (
    <div className={`metric-pill ${tone}`}>
      {icon}
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function PitchSchematic() {
  return (
    <div className="pitch-schematic" aria-hidden="true">
      <span className="midline" />
      <span className="center-circle" />
      <span className="box box-left" />
      <span className="box box-right" />
      <span className="player-dot dot-a" />
      <span className="player-dot dot-b" />
      <span className="player-dot dot-c" />
      <span className="ball-dot" />
    </div>
  );
}

function PipelineStrip({ running, complete }) {
  return (
    <div className="pipeline-strip" aria-label="Fluxo de processamento">
      {PIPELINE_STAGES.map((stage, index) => {
        const isActive = running && index <= 2;
        const isDone = complete || (running && index < 2);
        return (
          <div className={`pipeline-stage ${isActive ? "active" : ""} ${isDone ? "done" : ""}`} key={stage}>
            <span />
            <strong>{stage}</strong>
          </div>
        );
      })}
    </div>
  );
}

function NumberField({ label, value, min, max, step = 1, onChange }) {
  return (
    <Field label={label}>
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(event.target.value)}
      />
    </Field>
  );
}

function AdvancedControls({ params, devices, system, updateParam }) {
  return (
    <div className="advanced-controls">
      <SelectField
        label="Dispositivo"
        value={params.device}
        onChange={(value) => updateParam("device", value)}
        options={devices.map((device) => ({
          value: device,
          label: device,
          disabled: device === "cuda" && system && !system.cuda?.available
        }))}
      />

      <NumberField
        label="Resolução dos jogadores"
        min="320"
        max="1920"
        value={params.player_track_imgsz}
        onChange={(value) => updateParam("player_track_imgsz", value)}
      />

      <NumberField
        label="Campo a cada N"
        min="1"
        max="120"
        value={params.pitch_every_n_frames}
        onChange={(value) => updateParam("pitch_every_n_frames", value)}
      />

      <NumberField
        label="Resolução da bola"
        min="320"
        max="1920"
        value={params.ball_track_imgsz}
        onChange={(value) => updateParam("ball_track_imgsz", value)}
      />

      <NumberField
        label="Bola a cada N"
        min="1"
        max="120"
        value={params.ball_track_every_n_frames}
        onChange={(value) => updateParam("ball_track_every_n_frames", value)}
      />

      <Field label="Confiança da bola">
        <div className="range-pair">
          <input
            type="range"
            min="0.01"
            max="1"
            step="0.01"
            value={params.ball_track_conf}
            onChange={(event) => updateParam("ball_track_conf", event.target.value)}
          />
          <input
            type="number"
            min="0.01"
            max="1"
            step="0.01"
            value={params.ball_track_conf}
            onChange={(event) => updateParam("ball_track_conf", event.target.value)}
          />
        </div>
      </Field>

      <NumberField
        label="Retenção da bola"
        min="0"
        max="300"
        value={params.ball_max_hold_frames}
        onChange={(value) => updateParam("ball_max_hold_frames", value)}
      />

      <label className="toggle-row">
        <input
          type="checkbox"
          checked={Boolean(params.debug)}
          onChange={(event) => updateParam("debug", event.target.checked)}
        />
        <span>Debug</span>
      </label>

      <div className="tuning-summary">
        <Gauge size={18} />
        <span>
          {params.player_track_imgsz}px jogadores · campo/{params.pitch_every_n_frames} · bola/{params.ball_track_every_n_frames}
        </span>
      </div>
    </div>
  );
}

function SystemPanel({ system, error, onRefresh }) {
  const missingModels = system?.models?.filter((model) => !model.exists) || [];
  const modelsReady = Boolean(system && missingModels.length === 0);
  const processor = system?.processor;
  const processorMissing = processor?.missing_modules || [];
  const processorReady = processor ? Boolean(processor.ready) : true;
  const cudaReady = Boolean(system?.cuda?.available);

  return (
    <details className="panel system-panel readiness-details">
      <summary className="readiness-summary">
        <div>
          <h2>Sistema</h2>
        </div>
        <div className="readiness-summary-state">
          <MetricPill
            icon={modelsReady ? <CheckCircle2 size={17} /> : <AlertTriangle size={17} />}
            label="Modelos"
            value={modelsReady ? "OK" : "Falta"}
            tone={modelsReady ? "ok" : "warn"}
          />
          <ChevronDown className="summary-chevron" size={18} />
        </div>
      </summary>

      <div className="readiness-body">
        <button className="icon-button" type="button" title="Atualizar" onClick={onRefresh}>
          <RefreshCw size={18} />
        </button>

        {error ? <p className="notice error">{localizeTechnicalText(error)}</p> : null}

        <div className="metric-grid">
          <MetricPill
            icon={modelsReady ? <CheckCircle2 size={17} /> : <AlertTriangle size={17} />}
            label="Modelos"
            value={modelsReady ? "Modelos carregados" : "Modelos em falta"}
            tone={modelsReady ? "ok" : "warn"}
          />
          <MetricPill
            icon={processorReady ? <CheckCircle2 size={17} /> : <AlertTriangle size={17} />}
            label="Python"
            value={processorReady ? "Módulos OK" : "Módulos em falta"}
            tone={processorReady ? "ok" : "warn"}
          />
          <MetricPill
            icon={<Cpu size={17} />}
            label="GPU"
            value={cudaReady ? "CUDA" : "CPU"}
            tone={cudaReady ? "ok" : "warn"}
          />
        </div>

        <p className={`device-line ${cudaReady ? "ok" : "warn"}`}>
          {cudaReady ? system.cuda.device_name : "CUDA indisponível"}
        </p>

        {processor ? (
          <p className={`device-line ${processorReady ? "ok" : "warn"}`}>
            Python: {processor.executable}
          </p>
        ) : null}

        <div className="model-list">
          {(system?.models || []).map((model) => (
            <div className="model-row" key={model.key}>
              <span>{model.key}</span>
              <strong>{model.exists ? formatBytes(model.size_bytes) : "em falta"}</strong>
            </div>
          ))}
        </div>

        {missingModels.length > 0 ? (
          <p className="notice">
            {missingModels.map((model) => model.path).join(", ")}
          </p>
        ) : null}

        {processorMissing.length > 0 ? (
          <p className="notice error">
            Módulos Python em falta no processador: {processorMissing.join(", ")}
          </p>
        ) : null}
      </div>
    </details>
  );
}

function ProcessingFeed({ job, percent, running, complete }) {
  const events = processingEvents(job);

  return (
    <section className="processing-feed" data-testid="logs" aria-label="Relato do processamento">
      <div className="feed-visual">
        <div className={`analysis-orb ${running ? "active" : ""} ${complete ? "complete" : ""}`}>
          <span className="orb-ring ring-a" />
          <span className="orb-ring ring-b" />
          <span className="orb-scan" />
          <strong>{percent}%</strong>
          <small>{complete ? "pronto" : running ? "a analisar" : "em espera"}</small>
        </div>
        <div className="feed-caption">
          <span>Relato tático</span>
          <strong>{statusLabel(job?.status)}</strong>
        </div>
      </div>

      <ol className="event-list">
        {events.map((event, index) => (
          <li className={`event-card ${event.tone}`} key={`${event.title}-${index}`}>
            <span className="event-dot" />
            <div>
              <strong>{event.title}</strong>
              <p>{event.body}</p>
            </div>
            <small>{event.meta}</small>
          </li>
        ))}
      </ol>
    </section>
  );
}

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

function StatsSection({ job }) {
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

export function StatusPanel({ job, onCancel }) {
  const percent = jobPercent(job);
  const running = job && !TERMINAL_STATUSES.has(job.status);
  const outputUrl = job?.output_url;
  const previewUrl = job?.preview_url || outputUrl;
  const liveStreamUrl = job?.live_stream_url;
  const complete = job?.status === "succeeded";
  const showLiveStream = Boolean(job?.live_enabled && liveStreamUrl && running);
  const positionsReady = Boolean(complete && job?.positions_ready);
  const [show3d, setShow3d] = useState(false);

  useEffect(() => {
    setShow3d(false);
  }, [job?.job_id]);

  return (
    <section className="panel status-panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Painel ao vivo</p>
          <h1>Processamento</h1>
        </div>
        {running ? (
          <button className="danger-button" type="button" onClick={onCancel}>
            <StopCircle size={18} />
            Cancelar
          </button>
        ) : null}
      </div>

      <PipelineStrip running={Boolean(running)} complete={complete} />

      <div className={`job-state ${job?.status || "idle"}`}>
        {running ? <Loader2 className="spin" size={20} /> : <Activity size={20} />}
        <span>{statusLabel(job?.status)}</span>
      </div>

      <div className="progress-shell" aria-label="Progresso">
        <div className="progress-bar" style={{ width: `${percent}%` }} />
      </div>
      <div className="progress-meta">
        <strong>{percent}%</strong>
        <span>
          {job?.processed_frames ?? 0}
          {job?.total_frames ? ` / ${job.total_frames}` : ""} fotogramas
        </span>
      </div>

      {job?.error ? <p className="notice error">{localizeTechnicalText(job.error)}</p> : null}

      {showLiveStream ? (
        <div className="live-block">
          <div className="live-block-head">
            <span>
              <Radio size={16} />
              Live
            </span>
            <small>{job?.processed_frames ? `${job.processed_frames} fotogramas` : "A aguardar primeiro fotograma"}</small>
          </div>
          <img className="live-preview" src={liveStreamUrl} alt="Processamento em direto" />
        </div>
      ) : null}

      {outputUrl ? (
        <div className="output-block">
          {show3d && positionsReady ? (
            <Suspense
              fallback={
                <div className="match3d-loading">
                  <Loader2 className="spin" size={20} />
                  <span>A carregar o motor 3D…</span>
                </div>
              }
            >
              <Match3DPanel jobId={job.job_id} />
            </Suspense>
          ) : (
            <video controls preload="metadata" key={previewUrl}>
              <source src={previewUrl} type="video/mp4" />
            </video>
          )}
          <div className="output-actions">
            <a className="primary-button" href={outputUrl}>
              <Download size={18} />
              Descarregar
            </a>
            {positionsReady ? (
              <button
                className="ghost-button"
                type="button"
                onClick={() => setShow3d((value) => !value)}
              >
                {show3d ? <FileVideo size={18} /> : <Layers3 size={18} />}
                {show3d ? "Vídeo Processado" : "Visualização 3D"}
              </button>
            ) : null}
            {previewUrl !== outputUrl ? (
              <a className="ghost-button" href={previewUrl} target="_blank" rel="noreferrer">
                Abrir pré-visualização
              </a>
            ) : null}
          </div>
        </div>
      ) : null}

      <StatsSection job={job} />

      <ProcessingFeed job={job} percent={percent} running={Boolean(running)} complete={complete} />
    </section>
  );
}

export default function App() {
  const [system, setSystem] = useState(null);
  const [systemError, setSystemError] = useState("");
  const [params, setParams] = useState(FALLBACK_DEFAULTS);
  const [file, setFile] = useState(null);
  const [fileUrl, setFileUrl] = useState("");
  const [job, setJob] = useState(null);
  const [submitError, setSubmitError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [theme, setTheme] = useState(getInitialTheme);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [liveMode, setLiveMode] = useState(false);

  const running = Boolean(job && !TERMINAL_STATUSES.has(job.status));
  const canSubmit = Boolean(file && system?.ready && !running && !isSubmitting);
  const modes = system?.modes || ["RADAR"];
  const devices = system?.devices || ["cuda", "cpu"];

  const fileLabel = useMemo(() => {
    if (!file) return "Selecionar vídeo";
    return `${file.name} · ${formatBytes(file.size)}`;
  }, [file]);

  async function loadSystem() {
    try {
      setSystemError("");
      const nextSystem = await fetchSystem();
      setSystem(nextSystem);
      setParams(normalizeDefaults(nextSystem));
    } catch (error) {
      setSystemError(error.message);
    }
  }

  useEffect(() => {
    loadSystem();
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;

    try {
      window.localStorage.setItem("footar-theme", theme);
    } catch {
      // O tema continua a funcionar mesmo sem persistência.
    }
  }, [theme]);

  useEffect(() => {
    return () => {
      if (fileUrl) URL.revokeObjectURL(fileUrl);
    };
  }, [fileUrl]);

  useEffect(() => {
    if (!job?.job_id || TERMINAL_STATUSES.has(job.status)) return undefined;

    let active = true;
    async function poll() {
      try {
        const nextJob = await fetchJob(job.job_id);
        if (active) setJob(nextJob);
      } catch (error) {
        if (active) setSubmitError(error.message);
      }
    }

    poll();
    const timer = window.setInterval(poll, 1500);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [job?.job_id, job?.status]);

  function updateParam(key, value) {
    setParams((current) => ({ ...current, [key]: value }));
  }

  function selectFile(nextFile) {
    if (!nextFile) return;
    setFile(nextFile);
    setSubmitError("");
    const nextUrl = URL.createObjectURL(nextFile);
    setFileUrl(nextUrl);
  }

  async function handleSubmit(event) {
    event.preventDefault();
    if (!canSubmit) return;

    setIsSubmitting(true);
    setSubmitError("");
    try {
      const created = await createJob(file, params, { live: liveMode });
      setJob({
        job_id: created.job_id,
        status: "queued",
        progress: 0,
        processed_frames: 0,
        total_frames: null,
        logs: ["Job queued"],
        live_enabled: liveMode,
        live_frame_url: liveMode ? `/api/jobs/${created.job_id}/live-frame` : null,
        live_stream_url: liveMode ? `/api/jobs/${created.job_id}/live-stream` : null
      });
    } catch (error) {
      setSubmitError(error.message);
    } finally {
      setIsSubmitting(false);
    }
  }

  async function handleCancel() {
    if (!job?.job_id) return;
    await cancelJob(job.job_id);
    const nextJob = await fetchJob(job.job_id);
    setJob(nextJob);
  }

  return (
    <main className="app-shell" data-theme={theme}>
      <div className="stadium-scene" aria-hidden="true">
        <div className="stadium-stand stand-top" />
        <div className="stadium-stand stand-bottom" />
        <div className="stadium-floodlights" />
        <div className="stadium-pitch">
          <span className="stadium-line half-line" />
          <span className="stadium-line center-mark" />
          <span className="stadium-line penalty-left" />
          <span className="stadium-line penalty-right" />
          <span className="stadium-line goal-left" />
          <span className="stadium-line goal-right" />
        </div>
      </div>

      <header className="topbar">
        <div className="brand-lockup">
          <img className="brand-logo" src="/footar_logo_white.svg" alt="FootAR" />
        </div>
        <div className="topbar-actions">
          <button
            className={`advanced-toggle ${advancedOpen ? "open" : ""}`}
            type="button"
            aria-expanded={advancedOpen}
            aria-controls="advanced-menu"
            onClick={() => setAdvancedOpen((current) => !current)}
          >
            <SlidersHorizontal size={18} />
            <span>Avançado</span>
            <ChevronDown className="summary-chevron" size={18} />
          </button>

          {advancedOpen ? (
            <section className="advanced-popover" id="advanced-menu" aria-label="Definições avançadas">
              <div className="advanced-popover-head">
                <div>
                  <p className="eyebrow">Avançado</p>
                  <h2>Controlo</h2>
                </div>
                <button
                  className="theme-toggle"
                  type="button"
                  onClick={() => setTheme((currentTheme) => (currentTheme === "dark" ? "light" : "dark"))}
                  aria-label={theme === "dark" ? "Ativar tema claro" : "Ativar tema escuro"}
                >
                  {theme === "dark" ? <Sun size={17} /> : <Moon size={17} />}
                </button>
              </div>

              <section className="advanced-card" aria-label="Ajustes avançados">
                <div className="advanced-card-title">
                  <SlidersHorizontal size={18} />
                  <strong>Ajustes avançados</strong>
                </div>
                <AdvancedControls params={params} devices={devices} system={system} updateParam={updateParam} />
              </section>

              <SystemPanel system={system} error={systemError} onRefresh={loadSystem} />
            </section>
          ) : null}
        </div>
      </header>

      <div className="workspace">
        <section className="left-stack">
          <section className="panel upload-panel">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Entrada</p>
                <h2>Vídeo</h2>
              </div>
              <FileVideo size={22} />
            </div>

            <form onSubmit={handleSubmit}>
              <label
                className={`drop-zone ${isDragging ? "dragging" : ""}`}
                onDragEnter={(event) => {
                  event.preventDefault();
                  setIsDragging(true);
                }}
                onDragOver={(event) => event.preventDefault()}
                onDragLeave={() => setIsDragging(false)}
                onDrop={(event) => {
                  event.preventDefault();
                  setIsDragging(false);
                  selectFile(event.dataTransfer.files?.[0]);
                }}
              >
                <input
                  aria-label="Selecionar vídeo"
                  type="file"
                  accept="video/*"
                  onChange={(event) => selectFile(event.target.files?.[0])}
                />
                <span className="drop-icon">
                  <Upload size={28} />
                </span>
                <span>{fileLabel}</span>
              </label>

              {fileUrl ? (
                <video className="input-preview" src={fileUrl} controls />
              ) : (
                <div className="preview-placeholder">
                  <PitchSchematic />
                </div>
              )}

              {submitError ? <p className="notice error">{localizeTechnicalText(submitError)}</p> : null}

              <div className="action-row">
                <label className={`live-switch ${liveMode ? "active" : ""}`}>
                  <input
                    type="checkbox"
                    checked={liveMode}
                    disabled={running || isSubmitting}
                    onChange={(event) => setLiveMode(event.target.checked)}
                    aria-label="Ativar processamento Live"
                  />
                  <span className="live-switch-track" aria-hidden="true">
                    <span className="live-switch-thumb" />
                  </span>
                  <span className="live-switch-label">{liveMode ? "Live" : "Off"}</span>
                </label>
                <button className="primary-button" type="submit" disabled={!canSubmit}>
                  <Play size={18} />
                  {isSubmitting ? "A preparar" : "Processar"}
                </button>
              </div>
            </form>
          </section>

          <section className="panel controls-panel">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Preparação tática</p>
                <h2>Parâmetros</h2>
              </div>
              <Settings2 size={22} />
            </div>

            <div className="mode-ribbon">
              <Layers3 size={18} />
              <span>{MODE_LABELS[params.mode] || params.mode}</span>
            </div>

            <div className="control-grid">
              <SelectField
                label="Modo"
                value={params.mode}
                onChange={(value) => updateParam("mode", value)}
                options={modes.map((mode) => ({ value: mode, label: MODE_LABELS[mode] || mode }))}
              />
            </div>
          </section>
        </section>

        <StatusPanel job={job} onCancel={handleCancel} />
      </div>
    </main>
  );
}
