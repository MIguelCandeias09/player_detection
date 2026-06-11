import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fetchStats } from "./api.js";

describe("fetchStats", () => {
  beforeEach(() => {
    global.fetch = vi.fn();
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("requests the stats endpoint and returns json", async () => {
    global.fetch.mockResolvedValue({ ok: true, json: async () => ({ fps: 25 }) });

    const data = await fetchStats("job-1");

    expect(global.fetch).toHaveBeenCalledWith("/api/jobs/job-1/stats");
    expect(data.fps).toBe(25);
  });

  it("throws on non-ok response", async () => {
    global.fetch.mockResolvedValue({ ok: false });

    await expect(fetchStats("job-1")).rejects.toThrow();
  });
});
