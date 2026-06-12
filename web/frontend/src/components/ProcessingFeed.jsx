import { statusLabel } from "../helpers.js";
import { processingEvents } from "../jobEvents.js";

export default function ProcessingFeed({ job, percent, running, complete }) {
  const events = processingEvents(job);

  return (
    <section className="processing-feed" data-testid="logs" aria-label="Relato do processamento">
      <div className="feed-visual">
        <div className={`analysis-orb ${running ? "active" : ""} ${complete ? "complete" : ""}`}>
          <span className="orb-ring ring-a" />
          <span className="orb-ring ring-b" />
          <span className="orb-scan" />
          <strong>{percent}%</strong>
          <small>{complete ? "pronto" : running ? "a analisar" : "em espera"}</small>
        </div>
        <div className="feed-caption">
          <span>Relato tático</span>
          <strong>{statusLabel(job?.status)}</strong>
        </div>
      </div>

      <ol className="event-list">
        {events.map((event, index) => (
          <li className={`event-card ${event.tone}`} key={`${event.title}-${index}`}>
            <span className="event-dot" />
            <div>
              <strong>{event.title}</strong>
              <p>{event.body}</p>
            </div>
            <small>{event.meta}</small>
          </li>
        ))}
      </ol>
    </section>
  );
}
