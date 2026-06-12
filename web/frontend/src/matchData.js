// Lógica pura de dados/animação do visualizador 3D (sem three.js), para ser
// testável e partilhada entre o Canvas e os controlos.
import { PITCH_LENGTH, PITCH_WIDTH } from "./pitchTexture.js";

export const HOLD_SECONDS = 0.8; // tempo que um jogador "aguenta" sem deteções antes de desaparecer

// Fração da distância ao alvo que sobra após 1 s de damping (quanto menor, mais ágil)
const DAMPING_REMAINDER = 0.0001;
// Acima disto o movimento é teleporte (reaparecimento/seek), não correção a suavizar
const SNAP_DISTANCE_M = 5;

export function toScene(x, y) {
  // dados: x 0..105 (comprimento), y 0..68 (largura), origem no canto superior esquerdo
  return [x - PITCH_LENGTH / 2, y - PITCH_WIDTH / 2];
}

export function buildIndex(data) {
  const frames = data.frames || [];
  if (frames.length === 0) return null;
  const firstI = frames[0].i;
  const lastI = frames[frames.length - 1].i;
  const fps = data.fps || 25;

  const roster = new Map();
  if (Array.isArray(data.roster) && data.roster.length > 0) {
    for (const r of data.roster) {
      roster.set(r.id, { t: r.t, c: r.c });
    }
  } else {
    // Payloads antigos: equipa/classe vinham em cada entrada por frame
    for (const frame of frames) {
      for (const p of frame.players) {
        roster.set(p.id, { t: p.t, c: p.c });
      }
    }
  }

  return {
    frames,
    firstI,
    lastI,
    fps,
    duration: Math.max((lastI - firstI) / fps, 0.04),
    roster
  };
}

// Frame mais próximo (binary search): maior idx com frames[idx].i <= f
export function findFrameIndex(frames, f) {
  let lo = 0;
  let hi = frames.length - 1;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if (frames[mid].i <= f) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

export function samplePositions(index, timeSec, out) {
  const { frames, firstI, fps } = index;
  const f = firstI + timeSec * fps;
  const idxA = findFrameIndex(frames, f);
  const frameA = frames[idxA];
  const frameB = frames[Math.min(idxA + 1, frames.length - 1)];
  const gap = frameB.i - frameA.i;
  // Não interpolar através de buracos longos (homografia perdida > 1 s)
  let alpha = gap > 0 && gap <= fps ? (f - frameA.i) / gap : 0;
  alpha = Math.max(0, Math.min(1, alpha));

  out.players.clear();
  const posB = new Map();
  if (alpha > 0) {
    for (const p of frameB.players) posB.set(p.id, p);
  }
  for (const p of frameA.players) {
    const b = posB.get(p.id);
    out.players.set(p.id, {
      x: b ? p.x + (b.x - p.x) * alpha : p.x,
      y: b ? p.y + (b.y - p.y) * alpha : p.y
    });
  }

  const ballA = frameA.ball;
  const ballB = alpha > 0 ? frameB.ball : null;
  if (ballA && ballB) {
    out.ball = { x: ballA.x + (ballB.x - ballA.x) * alpha, y: ballA.y + (ballB.y - ballA.y) * alpha };
  } else {
    out.ball = ballA || null;
  }
  return out;
}

// Fração do caminho até ao alvo a percorrer neste tick; independente do frame-rate
// (compor dois meios-passos equivale a um passo inteiro)
export function dampingAlpha(delta) {
  return 1 - Math.pow(DAMPING_REMAINDER, delta);
}

export function shouldSnap(dx, dz) {
  return dx * dx + dz * dz > SNAP_DISTANCE_M * SNAP_DISTANCE_M;
}
