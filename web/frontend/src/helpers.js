import { FALLBACK_DEFAULTS, STATUS_LABELS } from "./constants.js";

export function getInitialTheme() {
  if (typeof window === "undefined") return "dark";

  try {
    const storedTheme = window.localStorage.getItem("footar-theme");
    if (storedTheme === "light" || storedTheme === "dark") return storedTheme;
  } catch {
    // Ignora limitações de storage em navegação privada ou testes.
  }

  return window.matchMedia?.("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

export function formatBytes(bytes) {
  if (!bytes) return "0 MB";
  const mb = bytes / (1024 * 1024);
  return `${mb.toFixed(mb >= 10 ? 0 : 1)} MB`;
}

export function jobPercent(job) {
  const progress = Number(job?.progress || 0);
  return Math.round(Math.max(0, Math.min(1, progress)) * 100);
}

export function statusLabel(status) {
  return STATUS_LABELS[status || "idle"] || status || STATUS_LABELS.idle;
}

export function localizeTechnicalText(text) {
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

export function normalizeDefaults(system) {
  const defaults = { ...FALLBACK_DEFAULTS, ...(system?.defaults || {}) };
  if (system && !system.cuda?.available && defaults.device === "cuda") {
    defaults.device = "cpu";
  }
  return defaults;
}
