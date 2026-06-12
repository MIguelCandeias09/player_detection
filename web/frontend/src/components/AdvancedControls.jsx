import { Gauge } from "lucide-react";
import { Field, NumberField, SelectField } from "./Fields.jsx";

export default function AdvancedControls({ params, devices, system, updateParam }) {
  return (
    <div className="advanced-controls">
      <SelectField
        label="Dispositivo"
        value={params.device}
        onChange={(value) => updateParam("device", value)}
        options={devices.map((device) => ({
          value: device,
          label: device,
          disabled: device === "cuda" && system && !system.cuda?.available
        }))}
      />

      <NumberField
        label="Resolução dos jogadores"
        min="320"
        max="1920"
        value={params.player_track_imgsz}
        onChange={(value) => updateParam("player_track_imgsz", value)}
      />

      <NumberField
        label="Campo a cada N"
        min="1"
        max="120"
        value={params.pitch_every_n_frames}
        onChange={(value) => updateParam("pitch_every_n_frames", value)}
      />

      <NumberField
        label="Resolução da bola"
        min="320"
        max="1920"
        value={params.ball_track_imgsz}
        onChange={(value) => updateParam("ball_track_imgsz", value)}
      />

      <NumberField
        label="Bola a cada N"
        min="1"
        max="120"
        value={params.ball_track_every_n_frames}
        onChange={(value) => updateParam("ball_track_every_n_frames", value)}
      />

      <Field label="Confiança da bola">
        <div className="range-pair">
          <input
            type="range"
            min="0.01"
            max="1"
            step="0.01"
            value={params.ball_track_conf}
            onChange={(event) => updateParam("ball_track_conf", event.target.value)}
          />
          <input
            type="number"
            min="0.01"
            max="1"
            step="0.01"
            value={params.ball_track_conf}
            onChange={(event) => updateParam("ball_track_conf", event.target.value)}
          />
        </div>
      </Field>

      <NumberField
        label="Retenção da bola"
        min="0"
        max="300"
        value={params.ball_max_hold_frames}
        onChange={(value) => updateParam("ball_max_hold_frames", value)}
      />

      <label className="toggle-row">
        <input
          type="checkbox"
          checked={Boolean(params.debug)}
          onChange={(event) => updateParam("debug", event.target.checked)}
        />
        <span>Debug</span>
      </label>

      <div className="tuning-summary">
        <Gauge size={18} />
        <span>
          {params.player_track_imgsz}px jogadores · campo/{params.pitch_every_n_frames} · bola/{params.ball_track_every_n_frames}
        </span>
      </div>
    </div>
  );
}
