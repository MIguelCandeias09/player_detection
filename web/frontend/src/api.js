export async function fetchSystem() {
  const response = await fetch("/api/system");
  if (!response.ok) {
    throw new Error("Nao foi possivel obter o estado do sistema");
  }
  return response.json();
}

export async function createJob(file, params, options = {}) {
  const formData = new FormData();
  formData.append("video", file);
  Object.entries(params).forEach(([key, value]) => {
    formData.append(key, String(value));
  });
  formData.append("live", String(Boolean(options.live)));

  const response = await fetch("/api/jobs", {
    method: "POST",
    body: formData
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload.detail;
    const message = typeof detail === "string" ? detail : detail?.message || "Falha ao criar job";
    throw new Error(message);
  }
  return payload;
}

export async function fetchJob(jobId) {
  const response = await fetch(`/api/jobs/${jobId}`);
  if (!response.ok) {
    throw new Error("Nao foi possivel atualizar o job");
  }
  return response.json();
}

export async function cancelJob(jobId) {
  const response = await fetch(`/api/jobs/${jobId}/cancel`, { method: "POST" });
  if (!response.ok) {
    throw new Error("Nao foi possivel cancelar o job");
  }
  return response.json();
}

export async function fetchPositions(jobId) {
  const response = await fetch(`/api/jobs/${jobId}/positions`);
  if (!response.ok) {
    throw new Error("Nao foi possivel obter as posicoes do jogo");
  }
  return response.json();
}

export async function fetchStats(jobId) {
  const response = await fetch(`/api/jobs/${jobId}/stats`);
  if (!response.ok) {
    throw new Error("Nao foi possivel obter as estatisticas");
  }
  return response.json();
}
