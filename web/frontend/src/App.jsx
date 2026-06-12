import { useEffect, useMemo, useState } from "react";
import {
  ChevronDown,
  FileVideo,
  Layers3,
  Moon,
  Play,
  Settings2,
  SlidersHorizontal,
  Sun,
  Upload
} from "lucide-react";
import { cancelJob, createJob, fetchJob, fetchSystem } from "./api.js";
import { FALLBACK_DEFAULTS, MODE_LABELS, TERMINAL_STATUSES } from "./constants.js";
import { formatBytes, getInitialTheme, localizeTechnicalText, normalizeDefaults } from "./helpers.js";
import AdvancedControls from "./components/AdvancedControls.jsx";
import { PitchSchematic } from "./components/Decor.jsx";
import { SelectField } from "./components/Fields.jsx";
import StatusPanel from "./components/StatusPanel.jsx";
import SystemPanel from "./components/SystemPanel.jsx";

// Re-exportado para compatibilidade (testes e consumidores antigos importam de App.jsx)
export { default as StatusPanel } from "./components/StatusPanel.jsx";

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

  async function loadSystem(options = {}) {
    try {
      setSystemError("");
      const nextSystem = await fetchSystem(options);
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

              <SystemPanel system={system} error={systemError} onRefresh={() => loadSystem({ refresh: true })} />
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
