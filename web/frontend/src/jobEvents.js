import { localizeTechnicalText } from "./helpers.js";

export function parseFootarEvent(text) {
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

export function processingEvents(job) {
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
