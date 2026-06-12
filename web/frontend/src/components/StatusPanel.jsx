import { lazy, Suspense, useEffect, useState } from "react";
import { Activity, Download, FileVideo, Layers3, Loader2, Radio, StopCircle } from "lucide-react";
import { TERMINAL_STATUSES } from "../constants.js";
import { jobPercent, localizeTechnicalText, statusLabel } from "../helpers.js";
import { PipelineStrip } from "./Decor.jsx";
import ProcessingFeed from "./ProcessingFeed.jsx";
import StatsSection from "./StatsSection.jsx";

// Carregado sob demanda: evita puxar three.js até o utilizador abrir a vista 3D
const Match3DPanel = lazy(() => import("../Match3DViewer.jsx"));

export default function StatusPanel({ job, onCancel }) {
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
