import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App, { StatusPanel } from "./App.jsx";

const readySystem = {
  ready: true,
  models: [
    { key: "player_segmentation", path: "models/active/yolo11m_seg_players.pt", exists: true, size_bytes: 10 }
  ],
  cuda: { available: true, device_name: "RTX", error: null },
  defaults: {
    mode: "RADAR",
    device: "cuda",
    debug: false,
    player_track_imgsz: 1024,
    pitch_every_n_frames: 5,
    ball_track_imgsz: 960,
    ball_track_every_n_frames: 2,
    ball_track_conf: 0.25,
    ball_max_hold_frames: 3
  },
  modes: ["RADAR", "PLAYER_DETECTION"],
  devices: ["cuda", "cpu"]
};

describe("FootAR frontend", () => {
  beforeEach(() => {
    global.fetch = vi.fn();
    window.URL.createObjectURL = vi.fn(() => "blob:clip");
    window.URL.revokeObjectURL = vi.fn();
    window.localStorage.clear();
    delete document.documentElement.dataset.theme;
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("disables processing when preflight reports missing models", async () => {
    global.fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        ...readySystem,
        ready: false,
        models: [
          {
            key: "ball_detection",
            path: "models/active/ball_y11m_1280_footar_best.pt",
            exists: false,
            size_bytes: null
          }
        ]
      })
    });

    render(<App />);

    await waitFor(() => expect(screen.getByRole("button", { name: /Processar/i })).toBeDisabled());
    fireEvent.click(screen.getByRole("button", { name: /Avançado/i }));

    await waitFor(() => expect(screen.getByText("Modelos em falta")).toBeInTheDocument());
    expect(screen.getByRole("button", { name: /Processar/i })).toBeDisabled();
  });

  it("renders progress for running jobs", () => {
    render(
      <StatusPanel
        job={{
          job_id: "job-1",
          status: "running",
          progress: 0.42,
          processed_frames: 42,
          total_frames: 100,
          logs: [
            "rendering Radar in frame: 9.6ms",
            'FOOTAR_EVENT {"event":"progress","processed_frames":42,"total_frames":100,"progress":0.42}'
          ]
        }}
        onCancel={() => {}}
      />
    );

    expect(screen.getAllByText("42%").length).toBeGreaterThan(0);
    expect(screen.getByText("42 / 100 fotogramas")).toBeInTheDocument();
    expect(screen.getByTestId("logs")).toHaveTextContent("42 de 100 fotogramas analisados");
    expect(screen.getByTestId("logs")).toHaveTextContent("A compor a vista tática do lance");
    expect(screen.getByTestId("logs")).not.toHaveTextContent("FOOTAR_EVENT");
    expect(screen.getByTestId("logs")).not.toHaveTextContent("9.6ms");
  });

  it("renders error state", () => {
    render(
      <StatusPanel
        job={{
          job_id: "job-2",
          status: "failed",
          progress: 0.2,
          processed_frames: 20,
          total_frames: 100,
          error: "Processor exited with code 1",
          logs: []
        }}
        onCancel={() => {}}
      />
    );

    expect(screen.getAllByText("Falhou").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Processador terminou com código 1").length).toBeGreaterThan(0);
  });

  it("renders output preview and download when job succeeds", () => {
    render(
      <StatusPanel
        job={{
          job_id: "job-3",
          status: "succeeded",
          progress: 1,
          processed_frames: 100,
          total_frames: 100,
          output_url: "/api/jobs/job-3/output",
          logs: []
        }}
        onCancel={() => {}}
      />
    );

    expect(screen.getAllByText("100%").length).toBeGreaterThan(0);
    expect(screen.getByRole("link", { name: /Descarregar/i })).toHaveAttribute("href", "/api/jobs/job-3/output");
  });

  it("submits jobs with live mode when the toggle is enabled", async () => {
    global.fetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => readySystem
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ job_id: "job-live" })
      });

    render(<App />);

    await waitFor(() => expect(global.fetch).toHaveBeenCalledWith("/api/system"));

    const file = new File(["video bytes"], "clip.mp4", { type: "video/mp4" });
    fireEvent.change(screen.getByLabelText("Selecionar vídeo"), { target: { files: [file] } });
    await waitFor(() => expect(screen.getByRole("button", { name: /Processar/i })).not.toBeDisabled());
    fireEvent.click(screen.getByLabelText("Ativar processamento Live"));
    fireEvent.click(screen.getByRole("button", { name: /Processar/i }));

    await waitFor(() => expect(global.fetch).toHaveBeenCalledWith("/api/jobs", expect.any(Object)));
    const [, request] = global.fetch.mock.calls.find(([url]) => url === "/api/jobs");
    expect(request.body.get("live")).toBe("true");
  });

  it("toggles between white and dark themes", async () => {
    global.fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => readySystem
    });

    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: /Avançado/i }));
    const toggle = await screen.findByRole("button", { name: /Ativar tema claro/i });
    await waitFor(() => expect(document.documentElement).toHaveAttribute("data-theme", "dark"));

    fireEvent.click(toggle);

    await waitFor(() => expect(document.documentElement).toHaveAttribute("data-theme", "light"));
    expect(screen.getByRole("button", { name: /Ativar tema escuro/i })).toBeInTheDocument();
  });

  it("renders the statistics section when the job has stats", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        fps: 25,
        possession: {
          frames_analyzed: 100,
          loose_pct: 10,
          team: { "0": { pct: 64 }, "1": { pct: 36 } },
          top_players: []
        },
        players: [
          { tracker_id: 7, team: 0, distance_km: 1.23, max_speed_kmh: 28.4, possession_seconds: 42.1 }
        ],
        heatmaps: {
          global: "global.png",
          ball: "ball.png",
          team: { "0": "team_0.png", "1": "team_1.png" },
          players: [{ tracker_id: 7, team: 0, samples: 10, file: "player_7.png" }]
        }
      })
    });

    render(
      <StatusPanel
        job={{
          job_id: "job-9",
          status: "succeeded",
          progress: 1,
          processed_frames: 100,
          total_frames: 100,
          output_url: "/api/jobs/job-9/output",
          stats_ready: true,
          stats_url: "/api/jobs/job-9/stats",
          logs: []
        }}
        onCancel={() => {}}
      />
    );

    await waitFor(() => expect(global.fetch).toHaveBeenCalledWith("/api/jobs/job-9/stats"));

    expect(await screen.findByText(/64%/)).toBeInTheDocument();
    expect(screen.getByText("1.23")).toBeInTheDocument();
    expect(screen.getAllByText("Equipa A").length).toBeGreaterThan(0);

    const image = screen.getByAltText(/Heatmap/i);
    expect(image).toHaveAttribute("src", "/api/jobs/job-9/heatmap/global.png");

    // Dropdown custom: abrir e escolher a opcao "Equipa B".
    fireEvent.click(screen.getByRole("button", { name: "Selecionar heatmap" }));
    fireEvent.click(screen.getByRole("option", { name: "Equipa B" }));
    expect(screen.getByAltText(/Heatmap/i)).toHaveAttribute("src", "/api/jobs/job-9/heatmap/team_1.png");
  });
});
