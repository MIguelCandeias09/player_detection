// Gera a textura do relvado (riscas + linhas FIFA) num canvas 2D.
// Dimensões reais: 105m x 68m, escala 10 px/m.

const SCALE = 10; // px por metro
const LENGTH_M = 105;
const WIDTH_M = 68;

export const PITCH_LENGTH = LENGTH_M;
export const PITCH_WIDTH = WIDTH_M;

export function createPitchCanvas() {
  const canvas = document.createElement("canvas");
  canvas.width = LENGTH_M * SCALE;
  canvas.height = WIDTH_M * SCALE;
  const ctx = canvas.getContext("2d");

  // Riscas de relvado alternadas (efeito corta-relva)
  const stripeCount = 14;
  const stripeWidth = canvas.width / stripeCount;
  for (let s = 0; s < stripeCount; s += 1) {
    ctx.fillStyle = s % 2 === 0 ? "#2e8b3b" : "#27772f";
    ctx.fillRect(s * stripeWidth, 0, stripeWidth + 1, canvas.height);
  }

  // Vinheta suave nas margens para profundidade
  const vignette = ctx.createRadialGradient(
    canvas.width / 2, canvas.height / 2, canvas.height * 0.45,
    canvas.width / 2, canvas.height / 2, canvas.width * 0.75
  );
  vignette.addColorStop(0, "rgba(0,0,0,0)");
  vignette.addColorStop(1, "rgba(0,0,0,0.22)");
  ctx.fillStyle = vignette;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = "rgba(255,255,255,0.92)";
  ctx.lineWidth = 0.15 * SCALE;
  ctx.lineJoin = "round";

  const m = (v) => v * SCALE;
  const midX = canvas.width / 2;
  const midY = canvas.height / 2;

  // Contorno
  ctx.strokeRect(ctx.lineWidth / 2, ctx.lineWidth / 2, canvas.width - ctx.lineWidth, canvas.height - ctx.lineWidth);

  // Linha de meio-campo + círculo central + ponto
  ctx.beginPath();
  ctx.moveTo(midX, 0);
  ctx.lineTo(midX, canvas.height);
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(midX, midY, m(9.15), 0, Math.PI * 2);
  ctx.stroke();

  ctx.fillStyle = "rgba(255,255,255,0.92)";
  ctx.beginPath();
  ctx.arc(midX, midY, m(0.25), 0, Math.PI * 2);
  ctx.fill();

  // Áreas (esquerda e direita)
  const penaltyDepth = 16.5;
  const penaltyWidth = 40.32;
  const goalDepth = 5.5;
  const goalWidth = 18.32;
  const penaltySpot = 11;

  for (const side of [0, 1]) {
    const dir = side === 0 ? 1 : -1;
    const edgeX = side === 0 ? 0 : canvas.width;

    // Grande área
    ctx.strokeRect(
      side === 0 ? 0 : edgeX - m(penaltyDepth),
      midY - m(penaltyWidth / 2),
      m(penaltyDepth),
      m(penaltyWidth)
    );

    // Pequena área
    ctx.strokeRect(
      side === 0 ? 0 : edgeX - m(goalDepth),
      midY - m(goalWidth / 2),
      m(goalDepth),
      m(goalWidth)
    );

    // Marca de penálti
    const spotX = edgeX + dir * m(penaltySpot);
    ctx.beginPath();
    ctx.arc(spotX, midY, m(0.25), 0, Math.PI * 2);
    ctx.fill();

    // Meia-lua fora da grande área (raio 9.15 m a partir da marca de penálti)
    const boxX = edgeX + dir * m(penaltyDepth);
    const cosLimit = (boxX - spotX) / m(9.15);
    const angle = Math.acos(Math.max(-1, Math.min(1, cosLimit)));
    ctx.beginPath();
    if (side === 0) {
      ctx.arc(spotX, midY, m(9.15), -angle, angle);
    } else {
      ctx.arc(spotX, midY, m(9.15), Math.PI - angle, Math.PI + angle);
    }
    ctx.stroke();

    // Arcos de canto
    for (const cornerY of [0, canvas.height]) {
      ctx.beginPath();
      const startAngle = side === 0
        ? (cornerY === 0 ? 0 : -Math.PI / 2)
        : (cornerY === 0 ? Math.PI / 2 : Math.PI);
      ctx.arc(edgeX, cornerY, m(1), startAngle, startAngle + Math.PI / 2);
      ctx.stroke();
    }
  }

  return canvas;
}
