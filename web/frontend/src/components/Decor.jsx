import { PIPELINE_STAGES } from "../constants.js";

export function PitchSchematic() {
  return (
    <div className="pitch-schematic" aria-hidden="true">
      <span className="midline" />
      <span className="center-circle" />
      <span className="box box-left" />
      <span className="box box-right" />
      <span className="player-dot dot-a" />
      <span className="player-dot dot-b" />
      <span className="player-dot dot-c" />
      <span className="ball-dot" />
    </div>
  );
}

export function PipelineStrip({ running, complete }) {
  return (
    <div className="pipeline-strip" aria-label="Fluxo de processamento">
      {PIPELINE_STAGES.map((stage, index) => {
        const isActive = running && index <= 2;
        const isDone = complete || (running && index < 2);
        return (
          <div className={`pipeline-stage ${isActive ? "active" : ""} ${isDone ? "done" : ""}`} key={stage}>
            <span />
            <strong>{stage}</strong>
          </div>
        );
      })}
    </div>
  );
}
