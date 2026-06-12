export const FALLBACK_DEFAULTS = {
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

export const TERMINAL_STATUSES = new Set(["succeeded", "failed", "cancelled"]);

export const MODE_LABELS = {
  PITCH_DETECTION: "Campo",
  PLAYER_DETECTION: "Jogadores",
  BALL_DETECTION: "Bola",
  PLAYER_TRACKING: "Seguimento",
  TEAM_CLASSIFICATION: "Equipas",
  RADAR: "Radar"
};

export const PIPELINE_STAGES = ["Carregar", "Detetar", "Seguir", "Radar", "Exportar"];

export const STATUS_LABELS = {
  idle: "A aguardar",
  queued: "Em fila",
  running: "Em processamento",
  succeeded: "Concluído",
  failed: "Falhou",
  cancelled: "Cancelado"
};
