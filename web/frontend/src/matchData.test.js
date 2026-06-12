import { describe, expect, it } from "vitest";
import {
  buildIndex,
  dampingAlpha,
  samplePositions,
  shouldSnap,
  toScene
} from "./matchData.js";

const DATA = {
  fps: 25,
  frames: [
    {
      i: 100,
      players: [
        { id: 1, t: 0, c: 2, x: 10, y: 20 },
        { id: 2, t: 1, c: 2, x: 50, y: 30 }
      ],
      ball: { x: 10, y: 10 }
    },
    { i: 105, players: [{ id: 1, t: 0, c: 2, x: 20, y: 30 }], ball: { x: 20, y: 20 } },
    { i: 205, players: [{ id: 1, t: 0, c: 2, x: 60, y: 40 }], ball: null }
  ]
};

function emptySample() {
  return { players: new Map(), ball: null };
}

describe("buildIndex", () => {
  it("computes duration from frame span and collects the roster", () => {
    const index = buildIndex(DATA);

    expect(index.firstI).toBe(100);
    expect(index.lastI).toBe(205);
    expect(index.duration).toBeCloseTo(4.2);
    expect([...index.roster.keys()].sort()).toEqual([1, 2]);
  });

  it("returns null without frames", () => {
    expect(buildIndex({ frames: [] })).toBeNull();
  });

  it("prefers the roster block from the payload when present", () => {
    // Payloads novos: equipa/classe so no roster (voto maioritario do backend)
    const data = {
      fps: 25,
      roster: [{ id: 1, t: 0, c: 2 }],
      frames: [
        { i: 100, players: [{ id: 1, x: 10, y: 20 }], ball: null },
        { i: 105, players: [{ id: 1, x: 20, y: 30 }], ball: null }
      ]
    };

    const index = buildIndex(data);

    expect(index.roster.get(1)).toEqual({ t: 0, c: 2 });
  });
});

describe("samplePositions", () => {
  it("interpolates players and ball between adjacent frames", () => {
    const index = buildIndex(DATA);
    // t=0.1s -> frame fracionario 102.5, a meio entre 100 e 105
    const out = samplePositions(index, 0.1, emptySample());

    expect(out.players.get(1).x).toBeCloseTo(15);
    expect(out.players.get(1).y).toBeCloseTo(25);
    expect(out.ball.x).toBeCloseTo(15);
    expect(out.ball.y).toBeCloseTo(15);
  });

  it("keeps the last position for a player absent from the next frame", () => {
    const index = buildIndex(DATA);
    const out = samplePositions(index, 0.1, emptySample());

    expect(out.players.get(2).x).toBeCloseTo(50);
    expect(out.players.get(2).y).toBeCloseTo(30);
  });

  it("does not interpolate across gaps longer than one second", () => {
    const index = buildIndex(DATA);
    // t=1.2s -> frame 130, entre 105 e 205 (gap de 100 frames > fps)
    const out = samplePositions(index, 1.2, emptySample());

    expect(out.players.get(1).x).toBeCloseTo(20);
    expect(out.players.get(1).y).toBeCloseTo(30);
    expect(out.ball.x).toBeCloseTo(20);
  });
});

describe("toScene", () => {
  it("centers pitch coordinates on the origin", () => {
    expect(toScene(52.5, 34)).toEqual([0, 0]);
    expect(toScene(0, 0)).toEqual([-52.5, -34]);
  });
});

describe("dampingAlpha", () => {
  it("is a fraction between 0 and 1", () => {
    const alpha = dampingAlpha(1 / 60);
    expect(alpha).toBeGreaterThan(0);
    expect(alpha).toBeLessThan(1);
  });

  it("is frame-rate independent: two half steps equal one full step", () => {
    const half = dampingAlpha(0.05);
    const composed = 1 - (1 - half) * (1 - half);
    expect(composed).toBeCloseTo(dampingAlpha(0.1), 6);
  });
});

describe("shouldSnap", () => {
  it("keeps damping for short corrections", () => {
    expect(shouldSnap(3, 0)).toBe(false);
    expect(shouldSnap(3, 3)).toBe(false);
  });

  it("snaps on teleport-sized jumps", () => {
    expect(shouldSnap(4, 4)).toBe(true);
    expect(shouldSnap(-6, 0)).toBe(true);
  });
});
