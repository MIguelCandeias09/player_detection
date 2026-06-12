import { AlertTriangle, CheckCircle2, ChevronDown, Cpu, RefreshCw } from "lucide-react";
import { formatBytes, localizeTechnicalText } from "../helpers.js";
import { MetricPill } from "./Fields.jsx";

export default function SystemPanel({ system, error, onRefresh }) {
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
