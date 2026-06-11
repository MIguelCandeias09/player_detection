// Visualização 3D do jogo (estilo Football Manager): campo, jogadores e bola
// animados a partir do positions.json produzido pelo pipeline.
import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Text, Billboard } from "@react-three/drei";
import * as THREE from "three";
import { Loader2, Pause, Play } from "lucide-react";
import { fetchPositions } from "./api.js";
import { createPitchCanvas, PITCH_LENGTH, PITCH_WIDTH } from "./pitchTexture.js";

// Paleta alinhada com o radar do pipeline (COLORS em main_seg.py)
const TEAM_COLORS = {
  0: "#FF1744", // Equipa 0
  1: "#2196F3", // Equipa 1
  gk: "#FF6347", // Guarda-redes
  ref: "#FFD700" // Árbitro
};

const HOLD_SECONDS = 0.8; // tempo que um jogador "aguenta" sem deteções antes de desaparecer

function entityColor(t, c) {
  if (t === 0 || t === 1) return TEAM_COLORS[t];
  if (c === 1) return TEAM_COLORS.gk;
  if (c === 3) return TEAM_COLORS.ref;
  return "#9e9e9e";
}

function toScene(x, y) {
  // dados: x 0..105 (comprimento), y 0..68 (largura), origem no canto superior esquerdo
  return [x - PITCH_LENGTH / 2, y - PITCH_WIDTH / 2];
}

function buildIndex(data) {
  const frames = data.frames || [];
  if (frames.length === 0) return null;
  const firstI = frames[0].i;
  const lastI = frames[frames.length - 1].i;
  const fps = data.fps || 25;

  const roster = new Map();
  for (const frame of frames) {
    for (const p of frame.players) {
      roster.set(p.id, { t: p.t, c: p.c });
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
function findFrameIndex(frames, f) {
  let lo = 0;
  let hi = frames.length - 1;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if (frames[mid].i <= f) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

function samplePositions(index, timeSec, out) {
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

function Pitch() {
  const texture = useMemo(() => {
    const tex = new THREE.CanvasTexture(createPitchCanvas());
    tex.anisotropy = 8;
    tex.colorSpace = THREE.SRGBColorSpace;
    return tex;
  }, []);

  return (
    <group>
      {/* Plano envolvente escuro (fora do campo) */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.05, 0]}>
        <planeGeometry args={[PITCH_LENGTH + 26, PITCH_WIDTH + 26]} />
        <meshStandardMaterial color="#142a1c" roughness={1} />
      </mesh>
      {/* Relvado */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <planeGeometry args={[PITCH_LENGTH, PITCH_WIDTH]} />
        <meshStandardMaterial map={texture} roughness={0.95} />
      </mesh>
      <Goal x={-PITCH_LENGTH / 2} />
      <Goal x={PITCH_LENGTH / 2} />
    </group>
  );
}

function Goal({ x }) {
  const postMaterial = useMemo(
    () => new THREE.MeshStandardMaterial({ color: "#ffffff", roughness: 0.4 }),
    []
  );
  const r = 0.08;
  const height = 2.44;
  const halfWidth = 7.32 / 2;
  return (
    <group position={[x, 0, 0]}>
      <mesh material={postMaterial} position={[0, height / 2, -halfWidth]} castShadow>
        <cylinderGeometry args={[r, r, height, 12]} />
      </mesh>
      <mesh material={postMaterial} position={[0, height / 2, halfWidth]} castShadow>
        <cylinderGeometry args={[r, r, height, 12]} />
      </mesh>
      <mesh material={postMaterial} position={[0, height, 0]} rotation={[Math.PI / 2, 0, 0]} castShadow>
        <cylinderGeometry args={[r, r, 7.32 + r * 2, 12]} />
      </mesh>
    </group>
  );
}

function Player({ color, label, refCallback }) {
  return (
    <group ref={refCallback} visible={false}>
      {/* Anel de seleção no chão (estilo FM) */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.03, 0]}>
        <ringGeometry args={[0.62, 0.85, 32]} />
        <meshBasicMaterial color={color} transparent opacity={0.85} />
      </mesh>
      {/* Corpo */}
      <mesh position={[0, 1.0, 0]} castShadow>
        <capsuleGeometry args={[0.42, 1.05, 6, 14]} />
        <meshStandardMaterial color={color} roughness={0.55} emissive={color} emissiveIntensity={0.12} />
      </mesh>
      {/* Número do jogador */}
      <Billboard position={[0, 2.45, 0]}>
        <Text
          fontSize={0.72}
          color="#ffffff"
          outlineWidth={0.07}
          outlineColor="#0a0a0a"
          anchorX="center"
          anchorY="middle"
        >
          {label}
        </Text>
      </Billboard>
    </group>
  );
}

function Scene({ index, storeRef }) {
  const playerRefs = useRef(new Map());
  const lastSeen = useRef(new Map());
  const ballRef = useRef(null);
  const sample = useRef({ players: new Map(), ball: null });

  const rosterEntries = useMemo(() => [...index.roster.entries()], [index]);

  useFrame((_, delta) => {
    const store = storeRef.current;
    if (store.playing) {
      store.time += delta * store.speed;
      if (store.time >= index.duration) store.time = 0; // loop
    }
    const time = store.time;
    samplePositions(index, time, sample.current);

    for (const [id] of rosterEntries) {
      const group = playerRefs.current.get(id);
      if (!group) continue;
      const pos = sample.current.players.get(id);
      if (pos) {
        const [sx, sz] = toScene(pos.x, pos.y);
        group.position.set(sx, 0, sz);
        group.visible = true;
        lastSeen.current.set(id, time);
      } else {
        const seen = lastSeen.current.get(id);
        group.visible = seen !== undefined && Math.abs(time - seen) < HOLD_SECONDS;
      }
    }

    if (ballRef.current) {
      const ball = sample.current.ball;
      if (ball) {
        const [sx, sz] = toScene(ball.x, ball.y);
        ballRef.current.position.set(sx, 0.32, sz);
        ballRef.current.visible = true;
      } else {
        ballRef.current.visible = false;
      }
    }
  });

  return (
    <group>
      <Pitch />
      {rosterEntries.map(([id, info]) => (
        <Player
          key={id}
          color={entityColor(info.t, info.c)}
          label={String(id)}
          refCallback={(node) => {
            if (node) playerRefs.current.set(id, node);
            else playerRefs.current.delete(id);
          }}
        />
      ))}
      <mesh ref={ballRef} visible={false} castShadow>
        <sphereGeometry args={[0.32, 24, 24]} />
        <meshStandardMaterial color="#ffffff" roughness={0.3} emissive="#ffffff" emissiveIntensity={0.18} />
      </mesh>
    </group>
  );
}

const CAMERA_PRESETS = {
  tv: { position: [0, 44, 54], label: "TV" },
  goal: { position: [-72, 17, 0], label: "Baliza" },
  top: { position: [0, 88, 0.1], label: "Topo" }
};

function CameraRig({ presetRef }) {
  const { camera, controls } = useThree();
  const target = useRef(null);

  useFrame((_, delta) => {
    if (presetRef.current) {
      if (!target.current) target.current = new THREE.Vector3();
      target.current.set(...presetRef.current);
      const t = 1 - Math.pow(0.0001, delta);
      camera.position.lerp(target.current, t);
      if (controls) {
        controls.target.lerp(new THREE.Vector3(0, 0, 0), t);
        controls.update();
      }
      if (camera.position.distanceTo(target.current) < 0.3) {
        presetRef.current = null;
      }
    }
  });
  return null;
}

function formatTime(seconds) {
  const total = Math.max(0, Math.floor(seconds));
  const mins = Math.floor(total / 60);
  const secs = total % 60;
  return `${mins}:${String(secs).padStart(2, "0")}`;
}

const SPEEDS = [0.5, 1, 2];

export function Match3DViewer({ data }) {
  const index = useMemo(() => buildIndex(data), [data]);
  const storeRef = useRef({ playing: true, time: 0, speed: 1 });
  const presetRef = useRef(CAMERA_PRESETS.tv.position);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [uiTime, setUiTime] = useState(0);
  const [activePreset, setActivePreset] = useState("tv");

  useEffect(() => {
    storeRef.current.playing = playing;
  }, [playing]);

  useEffect(() => {
    storeRef.current.speed = speed;
  }, [speed]);

  useEffect(() => {
    const timer = setInterval(() => setUiTime(storeRef.current.time), 200);
    return () => clearInterval(timer);
  }, []);

  if (!index) {
    return <p className="muted">Sem dados de posições para este vídeo.</p>;
  }

  return (
    <div className="match3d-viewer">
      <div className="match3d-canvas">
        <Canvas
          shadows
          dpr={[1, 2]}
          camera={{ position: CAMERA_PRESETS.tv.position, fov: 42, near: 0.5, far: 600 }}
        >
          <color attach="background" args={["#0c1626"]} />
          <fog attach="fog" args={["#0c1626", 140, 320]} />
          <hemisphereLight args={["#bcd6ff", "#1d3a24", 0.55]} />
          <directionalLight
            position={[45, 75, 25]}
            intensity={1.35}
            castShadow
            shadow-mapSize-width={2048}
            shadow-mapSize-height={2048}
            shadow-camera-left={-75}
            shadow-camera-right={75}
            shadow-camera-top={55}
            shadow-camera-bottom={-55}
          />
          <Scene index={index} storeRef={storeRef} />
          <OrbitControls
            makeDefault
            enableDamping
            maxPolarAngle={Math.PI / 2.12}
            minDistance={12}
            maxDistance={170}
            onStart={() => {
              presetRef.current = null;
            }}
          />
          <CameraRig presetRef={presetRef} />
        </Canvas>
        <div className="match3d-presets">
          {Object.entries(CAMERA_PRESETS).map(([key, preset]) => (
            <button
              key={key}
              type="button"
              className={`match3d-preset ${activePreset === key ? "active" : ""}`}
              onClick={() => {
                presetRef.current = preset.position;
                setActivePreset(key);
              }}
            >
              {preset.label}
            </button>
          ))}
        </div>
      </div>
      <div className="match3d-controls">
        <button
          type="button"
          className="match3d-play"
          onClick={() => setPlaying((value) => !value)}
          aria-label={playing ? "Pausar" : "Reproduzir"}
        >
          {playing ? <Pause size={18} /> : <Play size={18} />}
        </button>
        <span className="match3d-time">{formatTime(uiTime)}</span>
        <input
          type="range"
          min={0}
          max={index.duration}
          step={0.04}
          value={Math.min(uiTime, index.duration)}
          onChange={(event) => {
            const value = Number(event.target.value);
            storeRef.current.time = value;
            setUiTime(value);
          }}
        />
        <span className="match3d-time">{formatTime(index.duration)}</span>
        <button
          type="button"
          className="match3d-speed"
          onClick={() => {
            const next = SPEEDS[(SPEEDS.indexOf(speed) + 1) % SPEEDS.length];
            setSpeed(next);
          }}
        >
          {speed}x
        </button>
      </div>
    </div>
  );
}

export default function Match3DPanel({ jobId }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    setData(null);
    setError("");
    fetchPositions(jobId)
      .then((payload) => {
        if (!cancelled) setData(payload);
      })
      .catch((exc) => {
        if (!cancelled) setError(exc.message || "Falha ao carregar posições");
      });
    return () => {
      cancelled = true;
    };
  }, [jobId]);

  if (error) {
    return <p className="notice error">{error}</p>;
  }
  if (!data) {
    return (
      <div className="match3d-loading">
        <Loader2 className="spin" size={20} />
        <span>A preparar a visualização 3D…</span>
      </div>
    );
  }
  return <Match3DViewer data={data} />;
}
