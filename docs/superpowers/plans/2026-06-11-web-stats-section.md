# Web Statistics Section Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface the possession, distance/speed and heatmap statistics computed during RADAR processing in the web UI, as a collapsible "Estatísticas" section below the result video.

**Architecture:** The inference processor (`src/main_seg.py`, heavy env with cv2/torch) writes `stats.json` + heatmap PNGs into a per-job `stats/` directory at the end of processing. The FastAPI backend (light env) serves those files via new endpoints. The React frontend fetches the JSON on job success and renders a collapsible section with a possession bar, a players table and a heatmap viewer.

**Tech Stack:** Python (FastAPI, numpy, opencv, supervision), React 18 + Vite, pytest, vitest + @testing-library/react.

---

## Key conventions

- **Run backend/light tests** (anything importing only `web.backend.*` or `sports.stats_export`) with any Python that has `fastapi`+`pytest`. Command from `player_detection/`:
  `python -m pytest tests/<file>::<test> -v`
- **Run tracker tests** (importing `sports.possession_tracker` / `distance_tracker` / `heatmap_tracker`) with the **inference venv** (has `numpy`, `supervision`, `opencv`). Example from `web/README.md`:
  `& C:\VS-Projects\FootAR\FootAR_old\.venv\Scripts\python.exe -m pytest tests/<file> -v`
  Tracker test files import `sports.*` via the `sys.path` shim used in `tests/test_interpolation.py:13`.
- **Run frontend tests** from `player_detection/web/frontend/`: `npm test` (runs `vitest run`). For a single file: `npx vitest run src/<file>`.

## Canonical `stats.json` shape (authoritative — every task must match)

```json
{
  "fps": 25.0,
  "possession": {
    "frames_analyzed": 1234,
    "loose_pct": 12.3,
    "team": { "0": { "pct": 64.0 }, "1": { "pct": 36.0 } },
    "top_players": [ { "tracker_id": 7, "team": 0, "seconds": 42.1 } ]
  },
  "players": [
    { "tracker_id": 7, "team": 0, "distance_km": 1.23, "max_speed_kmh": 28.4, "possession_seconds": 42.1 }
  ],
  "heatmaps": {
    "global": "global.png",
    "ball": "ball.png",
    "team": { "0": "team_0.png", "1": "team_1.png" },
    "players": [ { "tracker_id": 7, "team": 0, "samples": 210, "file": "player_7.png" } ]
  }
}
```

## File structure

| File | Responsibility | Action |
|---|---|---|
| `src/sports/possession_tracker.py` | `to_dict()`, `player_seconds()` | Modify |
| `src/sports/distance_tracker.py` | `to_dict()` | Modify |
| `src/sports/heatmap_tracker.py` | `save_heatmaps()`, `player_team_map()` | Modify |
| `src/sports/stats_export.py` | pure `build_stats_payload()` + `write_stats()` (no heavy deps) | Create |
| `src/main_seg.py` | thread `stats_output_dir`/`headless` into `run_radar`; gate `show()`; emit `stats` event | Modify |
| `web/backend/jobs.py` | `JobRecord.stats_dir`, `stats_json_path`, `stats_ready`/`stats_url`, command flag | Modify |
| `web/backend/app.py` | create `stats/` dir; `/stats` + `/heatmap/{name}` endpoints | Modify |
| `web/frontend/src/api.js` | `fetchStats()` | Modify |
| `web/frontend/src/App.jsx` | `StatsSection` component wired into `StatusPanel` | Modify |
| `web/frontend/src/styles.css` | `.stats-*` styles | Modify |
| `tests/test_stats_export.py` | unit tests for the pure export helpers | Create |
| `tests/test_stats_trackers.py` | unit tests for tracker `to_dict()` | Create |
| `tests/test_web_backend.py` | endpoint + command tests | Modify |
| `web/frontend/src/api.test.jsx` | `fetchStats` test | Create |
| `web/frontend/src/App.test.jsx` | `StatsSection` render test | Modify |

---

## Task 1: `PossessionTracker.to_dict()` + `player_seconds()`

**Files:**
- Modify: `src/sports/possession_tracker.py`
- Test: `tests/test_stats_trackers.py`

- [ ] **Step 1: Write the failing test** (create `tests/test_stats_trackers.py`)

```python
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from sports.possession_tracker import PossessionTracker
from sports.distance_tracker import DistanceTracker


def test_possession_to_dict_reports_team_and_top_players():
    t = PossessionTracker(fps=10.0)
    t._total_frames = 100
    t._loose_frames = 20
    t._frames_team[0] = 60
    t._frames_team[1] = 20
    t._frames_player[7] = 30
    t._player_team[7] = 0

    d = t.to_dict()

    assert d["frames_analyzed"] == 100
    assert d["loose_pct"] == 20.0
    assert d["team"]["0"]["pct"] == 75.0
    assert d["team"]["1"]["pct"] == 25.0
    assert d["top_players"][0] == {"tracker_id": 7, "team": 0, "seconds": 3.0}


def test_possession_player_seconds_uses_fps():
    t = PossessionTracker(fps=10.0)
    t._frames_player[7] = 30
    t._frames_player[9] = 5

    assert t.player_seconds() == {7: 3.0, 9: 0.5}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& C:\VS-Projects\FootAR\FootAR_old\.venv\Scripts\python.exe -m pytest tests/test_stats_trackers.py -v`
Expected: FAIL with `AttributeError: 'PossessionTracker' object has no attribute 'to_dict'`

- [ ] **Step 3: Add the methods** — insert after `print_report` (before `_record_frame`) in `src/sports/possession_tracker.py`:

```python
    def to_dict(self) -> dict:
        """Serializa o relatorio de posse para JSON (frontend web)."""
        top_players = []
        if self._frames_player:
            ranked = sorted(
                self._frames_player.items(),
                key=lambda item: (-item[1], item[0]),
            )[:5]
            for tracker_id, frames in ranked:
                seconds = frames / self.fps if self.fps else 0.0
                top_players.append({
                    "tracker_id": tracker_id,
                    "team": self._player_team.get(tracker_id),
                    "seconds": round(seconds, 1),
                })
        return {
            "frames_analyzed": self._total_frames,
            "loose_pct": round(self._pct(self._loose_frames), 1),
            "team": {
                "0": {"pct": round(self._possession_pct(self._frames_team[0]), 1)},
                "1": {"pct": round(self._possession_pct(self._frames_team[1]), 1)},
            },
            "top_players": top_players,
        }

    def player_seconds(self) -> dict:
        """Tempo (s) com bola por tracker_id, para juntar a tabela de jogadores."""
        if not self.fps:
            return {tracker_id: 0.0 for tracker_id in self._frames_player}
        return {
            tracker_id: frames / self.fps
            for tracker_id, frames in self._frames_player.items()
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& C:\VS-Projects\FootAR\FootAR_old\.venv\Scripts\python.exe -m pytest tests/test_stats_trackers.py -v`
Expected: PASS (2 passed; `DistanceTracker` import unused for now is fine)

- [ ] **Step 5: Commit**

```bash
git add src/sports/possession_tracker.py tests/test_stats_trackers.py
git commit -m "feat(stats): serialize possession tracker to dict"
```

---

## Task 2: `DistanceTracker.to_dict()`

**Files:**
- Modify: `src/sports/distance_tracker.py`
- Test: `tests/test_stats_trackers.py`

- [ ] **Step 1: Add the failing test** (append to `tests/test_stats_trackers.py`)

```python
def test_distance_to_dict_sorted_by_tracker_id():
    t = DistanceTracker(fps=25.0)
    t._total_distance_m[7] = 1234.0
    t._max_speed_kmh[7] = 28.44
    t._total_distance_m[3] = 500.0

    rows = t.to_dict()

    assert [r["tracker_id"] for r in rows] == [3, 7]
    assert {"tracker_id": 7, "distance_km": 1.234, "max_speed_kmh": 28.4} in rows
    assert rows[0] == {"tracker_id": 3, "distance_km": 0.5, "max_speed_kmh": 0.0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& C:\VS-Projects\FootAR\FootAR_old\.venv\Scripts\python.exe -m pytest tests/test_stats_trackers.py::test_distance_to_dict_sorted_by_tracker_id -v`
Expected: FAIL with `AttributeError: 'DistanceTracker' object has no attribute 'to_dict'`

- [ ] **Step 3: Add the method** — insert after `print_report` (before the `# Internos` comment) in `src/sports/distance_tracker.py`:

```python
    def to_dict(self) -> list:
        """Serializa distancia/velocidade por tracker_id (frontend web)."""
        rows = []
        for tid in sorted(self._total_distance_m.keys()):
            rows.append({
                "tracker_id": tid,
                "distance_km": round(self._total_distance_m[tid] / 1000.0, 3),
                "max_speed_kmh": round(self._max_speed_kmh.get(tid, 0.0), 1),
            })
        return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& C:\VS-Projects\FootAR\FootAR_old\.venv\Scripts\python.exe -m pytest tests/test_stats_trackers.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/sports/distance_tracker.py tests/test_stats_trackers.py
git commit -m "feat(stats): serialize distance tracker to dict"
```

---

## Task 3: `HeatmapTracker.save_heatmaps()` + `player_team_map()`

**Files:**
- Modify: `src/sports/heatmap_tracker.py`
- Test: `tests/test_stats_trackers.py`

- [ ] **Step 1: Add the failing test** (append to `tests/test_stats_trackers.py`)

```python
import numpy as np

from sports.heatmap_tracker import HeatmapTracker


def test_heatmap_save_writes_pngs_and_manifest(tmp_path):
    t = HeatmapTracker()
    # Simula presencas: 1 jogador da equipa 0 com algumas amostras.
    t._grid_all[10, 10] = 5.0
    t._grids[0][10, 10] = 5.0
    t._grids_player[7][10, 10] = 5.0
    t._samples[0] = 5
    t._samples_player[7] = 5
    t._player_team[7] = 0

    manifest = t.save_heatmaps(str(tmp_path))

    assert (tmp_path / "global.png").exists()
    assert (tmp_path / "ball.png").exists()
    assert (tmp_path / "team_0.png").exists()
    assert (tmp_path / "team_1.png").exists()
    assert (tmp_path / "player_7.png").exists()
    assert manifest["global"] == "global.png"
    assert manifest["team"]["0"] == "team_0.png"
    assert manifest["players"][0] == {
        "tracker_id": 7, "team": 0, "samples": 5, "file": "player_7.png"
    }
    assert t.player_team_map() == {7: 0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& C:\VS-Projects\FootAR\FootAR_old\.venv\Scripts\python.exe -m pytest tests/test_stats_trackers.py::test_heatmap_save_writes_pngs_and_manifest -v`
Expected: FAIL with `AttributeError: 'HeatmapTracker' object has no attribute 'save_heatmaps'`

- [ ] **Step 3: Add `os` import + the methods** — at the top of `src/sports/heatmap_tracker.py`, the imports start with `from collections import defaultdict`. Add `import os` as the first line. Then insert these methods after `list_players` (before `print_report`):

```python
    def player_team_map(self) -> dict:
        """Mapa tracker_id -> equipa, para juntar a tabela de jogadores."""
        return dict(self._player_team)

    def save_heatmaps(self, out_dir: str) -> dict:
        """Renderiza e grava os PNGs dos heatmaps; devolve o manifesto.

        Escreve sempre os agregados (global, bola, equipa 0/1) e um PNG por
        jogador com amostras. Devolve o bloco "heatmaps" do stats.json.
        """
        os.makedirs(out_dir, exist_ok=True)

        cv2.imwrite(os.path.join(out_dir, "global.png"), self.render_global())
        cv2.imwrite(os.path.join(out_dir, "ball.png"), self.render_ball())
        cv2.imwrite(os.path.join(out_dir, "team_0.png"), self.render_team(0))
        cv2.imwrite(os.path.join(out_dir, "team_1.png"), self.render_team(1))

        manifest = {
            "global": "global.png",
            "ball": "ball.png",
            "team": {"0": "team_0.png", "1": "team_1.png"},
            "players": [],
        }
        for entry in self.list_players():
            tracker_id = entry["tracker_id"]
            filename = f"player_{tracker_id}.png"
            cv2.imwrite(os.path.join(out_dir, filename), self.render_player(tracker_id))
            manifest["players"].append({
                "tracker_id": tracker_id,
                "team": entry["team"],
                "samples": entry["samples"],
                "file": filename,
            })
        return manifest
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& C:\VS-Projects\FootAR\FootAR_old\.venv\Scripts\python.exe -m pytest tests/test_stats_trackers.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/sports/heatmap_tracker.py tests/test_stats_trackers.py
git commit -m "feat(stats): save heatmap PNGs and expose manifest"
```

---

## Task 4: Pure export helpers `src/sports/stats_export.py`

This module has NO heavy deps (only `json`, `os`) so it is fast to unit-test with stub trackers.

**Files:**
- Create: `src/sports/stats_export.py`
- Test: `tests/test_stats_export.py`

- [ ] **Step 1: Write the failing test** (create `tests/test_stats_export.py`)

```python
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from sports.stats_export import build_stats_payload, write_stats


class _StubPossession:
    def to_dict(self):
        return {
            "frames_analyzed": 10,
            "loose_pct": 0.0,
            "team": {"0": {"pct": 50.0}, "1": {"pct": 50.0}},
            "top_players": [],
        }

    def player_seconds(self):
        return {7: 42.1, 9: 0.0}


class _StubDistance:
    def to_dict(self):
        return [
            {"tracker_id": 7, "distance_km": 1.2, "max_speed_kmh": 20.0},
            {"tracker_id": 9, "distance_km": 3.4, "max_speed_kmh": 25.0},
        ]


class _StubHeatmap:
    def save_heatmaps(self, out_dir):
        with open(os.path.join(out_dir, "global.png"), "wb"):
            pass
        return {
            "global": "global.png",
            "ball": "ball.png",
            "team": {"0": "team_0.png", "1": "team_1.png"},
            "players": [],
        }

    def player_team_map(self):
        return {7: 0}


_MANIFEST = {
    "global": "global.png",
    "ball": "ball.png",
    "team": {"0": "team_0.png", "1": "team_1.png"},
    "players": [],
}


def test_build_stats_payload_merges_and_sorts_by_distance():
    payload = build_stats_payload(25.0, _StubPossession(), _StubDistance(), _MANIFEST, {7: 0})

    assert payload["fps"] == 25.0
    assert payload["heatmaps"] is _MANIFEST
    assert [p["tracker_id"] for p in payload["players"]] == [9, 7]

    p7 = next(p for p in payload["players"] if p["tracker_id"] == 7)
    assert p7 == {
        "tracker_id": 7, "team": 0, "distance_km": 1.2,
        "max_speed_kmh": 20.0, "possession_seconds": 42.1,
    }
    p9 = next(p for p in payload["players"] if p["tracker_id"] == 9)
    assert p9["team"] is None


def test_write_stats_writes_json_file(tmp_path):
    path = write_stats(str(tmp_path), 25.0, _StubPossession(), _StubDistance(), _StubHeatmap())

    assert path == os.path.join(str(tmp_path), "stats.json")
    data = json.loads((tmp_path / "stats.json").read_text(encoding="utf-8"))
    assert data["players"][0]["tracker_id"] == 9
    assert data["heatmaps"]["global"] == "global.png"


def test_write_stats_returns_none_without_dir():
    assert write_stats(None, 25.0, _StubPossession(), _StubDistance(), _StubHeatmap()) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_stats_export.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sports.stats_export'`

- [ ] **Step 3: Create `src/sports/stats_export.py`**

```python
"""Helpers puros (sem deps pesadas) para exportar estatisticas para JSON.

Mantidos fora do main_seg para serem testaveis sem importar torch/ultralytics.
"""
import json
import os


def build_stats_payload(fps, possession_tracker, distance_tracker, heatmap_manifest, team_map):
    """Junta os tres trackers num unico payload (ver stats.json canonico)."""
    possession_seconds = possession_tracker.player_seconds()

    players = []
    for row in distance_tracker.to_dict():
        tracker_id = row["tracker_id"]
        players.append({
            "tracker_id": tracker_id,
            "team": team_map.get(tracker_id),
            "distance_km": row["distance_km"],
            "max_speed_kmh": row["max_speed_kmh"],
            "possession_seconds": round(float(possession_seconds.get(tracker_id, 0.0)), 1),
        })
    players.sort(key=lambda r: r["distance_km"], reverse=True)

    return {
        "fps": fps,
        "possession": possession_tracker.to_dict(),
        "players": players,
        "heatmaps": heatmap_manifest,
    }


def write_stats(stats_output_dir, fps, possession_tracker, distance_tracker, heatmap_tracker, team_map=None):
    """Grava PNGs + stats.json. Devolve o caminho do JSON, ou None se sem dir."""
    if not stats_output_dir:
        return None

    os.makedirs(stats_output_dir, exist_ok=True)
    manifest = heatmap_tracker.save_heatmaps(stats_output_dir)
    resolved_team_map = team_map if team_map is not None else heatmap_tracker.player_team_map()
    payload = build_stats_payload(fps, possession_tracker, distance_tracker, manifest, resolved_team_map)

    path = os.path.join(stats_output_dir, "stats.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
    return path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_stats_export.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/sports/stats_export.py tests/test_stats_export.py
git commit -m "feat(stats): pure stats export helpers"
```

---

## Task 5: Wire stats export into `src/main_seg.py`

No new unit test (the module imports torch/ultralytics; the pure logic is already covered by Task 4). Verified via `--help` and the backend command test (Task 6).

**Files:**
- Modify: `src/main_seg.py`

- [ ] **Step 1: Import the helper** — near the other `from sports...` imports (around line 30-32), add:

```python
from sports.stats_export import write_stats
```

- [ ] **Step 2: Extend `run_radar` signature** — change the `def run_radar(...)` signature (starts at line 628) to add three params before the `) -> Iterator[np.ndarray]:` line:

```python
    ball_max_hold_frames: int = DEFAULT_BALL_MAX_HOLD_FRAMES,
    stats_output_dir: str = None,
    headless: bool = False,
    structured_logs: bool = False,
) -> Iterator[np.ndarray]:
```

- [ ] **Step 3: Replace the post-loop report block** — at the end of `run_radar` (currently lines 1110-1113):

```python
    distance_tracker.print_report()
    possession_tracker.print_report()
    heatmap_tracker.print_report()
    heatmap_tracker.show()
```

Replace with:

```python
    distance_tracker.print_report()
    possession_tracker.print_report()
    heatmap_tracker.print_report()

    if stats_output_dir:
        try:
            saved = write_stats(
                stats_output_dir,
                video_info.fps,
                possession_tracker,
                distance_tracker,
                heatmap_tracker,
            )
            if saved:
                emit_structured_event(structured_logs, 'stats', path='stats.json')
        except Exception as exc:
            print(f'[stats] Falha a exportar estatisticas: {exc}')

    if not headless:
        heatmap_tracker.show()
```

- [ ] **Step 4: Add `stats_output_dir` to `main()` signature** — in `def main(...)` (line 1116), after `debug_output_dir: str = None,` (line 1130) add:

```python
    stats_output_dir: str = None,
```

- [ ] **Step 5: Pass params into the `run_radar` call** — in `main()` the RADAR branch calls `run_radar(...)` (around line 1149-1158). Add three keyword args before the closing `)`:

```python
            ball_max_hold_frames=ball_max_hold_frames,
            stats_output_dir=stats_output_dir,
            headless=not preview,
            structured_logs=structured_logs,
        )
```

- [ ] **Step 6: Add the CLI argument** — after the `--debug_output_dir` argument (line 1352) add:

```python
    parser.add_argument('--stats_output_dir', type=str, required=False, help='Directory for stats.json and heatmap PNGs')
```

- [ ] **Step 7: Pass the CLI arg into both `main()` calls** — there are two `main(...)` invocations in `__main__` (after lines ~1404 and ~1429). In each, immediately after the `debug_output_dir=args.debug_output_dir,` line, add:

```python
                    stats_output_dir=args.stats_output_dir,
```

(match the existing indentation at each call site).

- [ ] **Step 8: Verify the CLI exposes the flag**

Run: `& C:\VS-Projects\FootAR\FootAR_old\.venv\Scripts\python.exe src/main_seg.py --help`
Expected: output contains `--stats_output_dir`

- [ ] **Step 9: Commit**

```bash
git add src/main_seg.py
git commit -m "feat(stats): write stats.json from run_radar and gate cv2 window"
```

---

## Task 6: Backend job record + processing command

**Files:**
- Modify: `web/backend/jobs.py`
- Test: `tests/test_web_backend.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_web_backend.py`)

```python
def test_build_processing_command_includes_stats_dir(tmp_path):
    manager = JobManager()
    stats_dir = tmp_path / "stats"
    job = manager.create_job(
        input_filename="clip.mp4",
        input_path=tmp_path / "input.mp4",
        output_path=tmp_path / "output.mp4",
        debug_output_dir=tmp_path / "debug",
        params=ProcessingParams.from_raw(device="cpu"),
        stats_dir=stats_dir,
        job_id="job-stats",
    )

    command = build_processing_command(job)

    assert command[command.index("--stats_output_dir") + 1] == str(stats_dir)


def test_snapshot_reports_stats_ready(tmp_path):
    manager = JobManager()
    stats_dir = tmp_path / "stats"
    stats_dir.mkdir()
    (stats_dir / "stats.json").write_text('{"fps": 25.0, "players": []}', encoding="utf-8")
    job = manager.create_job(
        input_filename="clip.mp4",
        input_path=tmp_path / "input.mp4",
        output_path=tmp_path / "output.mp4",
        debug_output_dir=tmp_path / "debug",
        params=ProcessingParams(),
        stats_dir=stats_dir,
        job_id="job-snap",
    )
    job.status = "succeeded"

    snapshot = manager.snapshot("job-snap")
    assert snapshot["stats_ready"] is True
    assert snapshot["stats_url"] == "/api/jobs/job-snap/stats"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_web_backend.py::test_build_processing_command_includes_stats_dir tests/test_web_backend.py::test_snapshot_reports_stats_ready -v`
Expected: FAIL — `create_job()` got an unexpected keyword argument `stats_dir`

- [ ] **Step 3a: Add the field to `JobRecord`** — in `web/backend/jobs.py`, after `live_frame_dir: Path | None = None` (line 122) add:

```python
    stats_dir: Path | None = None
```

- [ ] **Step 3b: Expose stats in `to_public_dict`** — inside `JobRecord.to_public_dict` (line 135), after the `live_enabled = ...` line (139) add:

```python
        stats_ready = bool(self.stats_json_path and self.stats_json_path.exists())
```

and inside the returned dict, after the `"live_stream_url": ...` entry (line 156) add:

```python
            "stats_ready": stats_ready,
            "stats_url": f"/api/jobs/{self.id}/stats" if stats_ready else None,
```

- [ ] **Step 3c: Add the `stats_json_path` property** — after the existing `live_frame_path` property (lines 161-163) add:

```python
    @property
    def stats_json_path(self) -> Path | None:
        return self.stats_dir / "stats.json" if self.stats_dir is not None else None
```

- [ ] **Step 3d: Accept `stats_dir` in `JobManager.create_job`** — add the parameter (after `live_frame_dir: Path | None = None,`, line 260) and pass it to `JobRecord`:

```python
        live_frame_dir: Path | None = None,
        stats_dir: Path | None = None,
        job_id: str | None = None,
```

and in the `JobRecord(...)` construction (after `live_frame_dir=live_frame_dir,`, line 271):

```python
            live_frame_dir=live_frame_dir,
            stats_dir=stats_dir,
```

- [ ] **Step 3e: Add the command flag** — in `build_processing_command` (line 166), after the `if job.live_frame_dir is not None:` block (lines 197-205) add:

```python
    if job.stats_dir is not None:
        command.extend(["--stats_output_dir", str(job.stats_dir)])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_web_backend.py -v`
Expected: PASS (all, including the two new tests)

- [ ] **Step 5: Commit**

```bash
git add web/backend/jobs.py tests/test_web_backend.py
git commit -m "feat(stats): track stats dir on jobs and pass to processor"
```

---

## Task 7: Backend `/stats` and `/heatmap/{name}` endpoints

**Files:**
- Modify: `web/backend/app.py`
- Test: `tests/test_web_backend.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_web_backend.py`; add `import json` at the top of the file if not already present — it is not, so add it after the `from pathlib import Path` line)

```python
def test_stats_endpoint_serves_json(monkeypatch, tmp_path):
    manager = JobManager()
    stats_dir = tmp_path / "stats"
    stats_dir.mkdir()
    (stats_dir / "stats.json").write_text('{"fps": 25.0, "players": []}', encoding="utf-8")
    manager.create_job(
        input_filename="c.mp4",
        input_path=tmp_path / "i.mp4",
        output_path=tmp_path / "o.mp4",
        debug_output_dir=tmp_path / "d",
        params=ProcessingParams(),
        stats_dir=stats_dir,
        job_id="job-st",
    )
    monkeypatch.setattr(api, "JOB_MANAGER", manager)

    client = TestClient(api.app)
    response = client.get("/api/jobs/job-st/stats")

    assert response.status_code == 200
    assert response.json()["fps"] == 25.0


def test_stats_endpoint_404_without_file(monkeypatch, tmp_path):
    manager = JobManager()
    manager.create_job(
        input_filename="c.mp4",
        input_path=tmp_path / "i.mp4",
        output_path=tmp_path / "o.mp4",
        debug_output_dir=tmp_path / "d",
        params=ProcessingParams(),
        stats_dir=tmp_path / "stats",
        job_id="job-no-stats",
    )
    monkeypatch.setattr(api, "JOB_MANAGER", manager)

    client = TestClient(api.app)
    assert client.get("/api/jobs/job-no-stats/stats").status_code == 404


def test_heatmap_endpoint_validates_name(monkeypatch, tmp_path):
    manager = JobManager()
    stats_dir = tmp_path / "stats"
    stats_dir.mkdir()
    (stats_dir / "stats.json").write_text(
        json.dumps({
            "heatmaps": {
                "global": "global.png", "ball": "ball.png",
                "team": {"0": "team_0.png", "1": "team_1.png"}, "players": [],
            }
        }),
        encoding="utf-8",
    )
    (stats_dir / "global.png").write_bytes(b"PNGDATA")
    manager.create_job(
        input_filename="c.mp4",
        input_path=tmp_path / "i.mp4",
        output_path=tmp_path / "o.mp4",
        debug_output_dir=tmp_path / "d",
        params=ProcessingParams(),
        stats_dir=stats_dir,
        job_id="job-hm",
    )
    monkeypatch.setattr(api, "JOB_MANAGER", manager)

    client = TestClient(api.app)
    ok = client.get("/api/jobs/job-hm/heatmap/global.png")
    bad = client.get("/api/jobs/job-hm/heatmap/secret.txt")

    assert ok.status_code == 200
    assert ok.content == b"PNGDATA"
    assert ok.headers["content-type"] == "image/png"
    assert bad.status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_web_backend.py::test_stats_endpoint_serves_json tests/test_web_backend.py::test_heatmap_endpoint_validates_name -v`
Expected: FAIL with 404 (routes not defined) / KeyError

- [ ] **Step 3a: Add the manifest helper** — in `web/backend/app.py`, after the `safe_filename` function (lines 187-189) add:

```python
def allowed_heatmap_files(stats_path: Path) -> set[str]:
    try:
        data = json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()

    heatmaps = data.get("heatmaps") or {}
    names: set[str] = set()
    for key in ("global", "ball"):
        if heatmaps.get(key):
            names.add(heatmaps[key])
    for value in (heatmaps.get("team") or {}).values():
        if value:
            names.add(value)
    for player in heatmaps.get("players") or []:
        if player.get("file"):
            names.add(player["file"])
    return names
```

- [ ] **Step 3b: Add the endpoints** — after `get_job_output` (lines 313-320) add:

```python
@app.get("/api/jobs/{job_id}/stats")
def get_job_stats(job_id: str) -> FileResponse:
    job = JOB_MANAGER.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    stats_path = job.stats_json_path
    if stats_path is None or not stats_path.exists():
        raise HTTPException(status_code=404, detail="Statistics are not available")
    return FileResponse(stats_path, media_type="application/json")


@app.get("/api/jobs/{job_id}/heatmap/{name}")
def get_job_heatmap(job_id: str, name: str) -> FileResponse:
    job = JOB_MANAGER.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    stats_path = job.stats_json_path
    if stats_path is None or not stats_path.exists():
        raise HTTPException(status_code=404, detail="Statistics are not available")
    if name not in allowed_heatmap_files(stats_path):
        raise HTTPException(status_code=404, detail="Heatmap not found")
    image_path = job.stats_dir / name
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Heatmap image is missing")
    return FileResponse(image_path, media_type="image/png")
```

- [ ] **Step 3c: Create the stats dir on job creation** — in `create_job` (line 210), next to `live_frame_dir` setup (lines 272-274), after `debug_dir = result_dir / "debug"` (line 266) add:

```python
    stats_dir = result_dir / "stats"
```

and after `result_dir.mkdir(parents=True, exist_ok=True)` (line 268) add:

```python
    stats_dir.mkdir(parents=True, exist_ok=True)
```

and in the `JOB_MANAGER.create_job(...)` call (lines 283-291), after `live_frame_dir=live_frame_dir,` add:

```python
        stats_dir=stats_dir,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_web_backend.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add web/backend/app.py tests/test_web_backend.py
git commit -m "feat(stats): serve stats.json and heatmap PNGs"
```

---

## Task 8: Frontend `fetchStats`

**Files:**
- Modify: `web/frontend/src/api.js`
- Test: `web/frontend/src/api.test.jsx`

- [ ] **Step 1: Write the failing test** (create `web/frontend/src/api.test.jsx`)

```jsx
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
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `web/frontend/`): `npx vitest run src/api.test.jsx`
Expected: FAIL — `fetchStats` is not exported

- [ ] **Step 3: Add `fetchStats`** — append to `web/frontend/src/api.js`:

```js
export async function fetchStats(jobId) {
  const response = await fetch(`/api/jobs/${jobId}/stats`);
  if (!response.ok) {
    throw new Error("Nao foi possivel obter as estatisticas");
  }
  return response.json();
}
```

- [ ] **Step 4: Run test to verify it passes**

Run (from `web/frontend/`): `npx vitest run src/api.test.jsx`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add web/frontend/src/api.js web/frontend/src/api.test.jsx
git commit -m "feat(stats): frontend fetchStats helper"
```

---

## Task 9: `StatsSection` component

**Files:**
- Modify: `web/frontend/src/App.jsx`
- Test: `web/frontend/src/App.test.jsx`

- [ ] **Step 1: Write the failing test** (append a new `it(...)` inside the `describe("FootAR frontend", ...)` block in `web/frontend/src/App.test.jsx`)

```jsx
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
    // "Equipa A" aparece na celula da tabela E na opcao do seletor -> getAllByText.
    expect(screen.getAllByText("Equipa A").length).toBeGreaterThan(0);

    const image = screen.getByAltText(/Heatmap/i);
    expect(image).toHaveAttribute("src", "/api/jobs/job-9/heatmap/global.png");

    fireEvent.change(screen.getByLabelText("Selecionar heatmap"), { target: { value: "team1" } });
    expect(screen.getByAltText(/Heatmap/i)).toHaveAttribute("src", "/api/jobs/job-9/heatmap/team_1.png");
  });
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `web/frontend/`): `npx vitest run src/App.test.jsx`
Expected: FAIL — no element with text `/64%/` (StatsSection not rendered)

- [ ] **Step 3a: Import `fetchStats`** — in `web/frontend/src/App.jsx`, change the api import (line 23) to include `fetchStats`:

```jsx
import { cancelJob, createJob, fetchJob, fetchStats, fetchSystem } from "./api.js";
```

- [ ] **Step 3b: Add helpers + the component** — add these above the `StatusPanel` definition (before line 636) in `web/frontend/src/App.jsx`:

```jsx
function teamLabel(team) {
  if (team === 0) return "Equipa A";
  if (team === 1) return "Equipa B";
  return "—";
}

function buildHeatmapOptions(stats) {
  if (!stats?.heatmaps) return [];
  const heatmaps = stats.heatmaps;
  const options = [
    { key: "global", label: "Global", file: heatmaps.global },
    { key: "team0", label: "Equipa A", file: heatmaps.team?.["0"] },
    { key: "team1", label: "Equipa B", file: heatmaps.team?.["1"] },
    { key: "ball", label: "Bola", file: heatmaps.ball }
  ].filter((option) => option.file);

  (heatmaps.players || []).forEach((player) => {
    options.push({
      key: `player_${player.tracker_id}`,
      label: `Jogador ${player.tracker_id}`,
      file: player.file
    });
  });
  return options;
}

function StatsSection({ job }) {
  const jobId = job?.job_id;
  const ready = job?.status === "succeeded" && Boolean(job?.stats_ready);
  const [stats, setStats] = useState(null);
  const [error, setError] = useState("");
  const [heatmap, setHeatmap] = useState("global");

  useEffect(() => {
    if (!ready || !jobId) return undefined;
    let active = true;
    fetchStats(jobId)
      .then((data) => {
        if (active) setStats(data);
      })
      .catch((err) => {
        if (active) setError(err.message);
      });
    return () => {
      active = false;
    };
  }, [ready, jobId]);

  if (!ready) return null;

  const possession = stats?.possession;
  const team0 = possession?.team?.["0"]?.pct ?? 0;
  const team1 = possession?.team?.["1"]?.pct ?? 0;
  const heatmapOptions = buildHeatmapOptions(stats);
  const selected = heatmapOptions.find((option) => option.key === heatmap) || heatmapOptions[0];
  const heatmapUrl = selected ? `/api/jobs/${jobId}/heatmap/${selected.file}` : null;

  return (
    <details className="panel stats-panel" data-testid="stats">
      <summary className="readiness-summary">
        <div>
          <h2>Estatísticas</h2>
        </div>
        <ChevronDown className="summary-chevron" size={18} />
      </summary>

      <div className="stats-body">
        {error ? <p className="notice error">{localizeTechnicalText(error)}</p> : null}

        {stats ? (
          <>
            <div className="possession-block">
              <div className="possession-bar">
                <span className="poss-team team-a" style={{ width: `${team0}%` }}>
                  A {team0}%
                </span>
                <span className="poss-team team-b" style={{ width: `${team1}%` }}>
                  B {team1}%
                </span>
              </div>
              <small>Bola solta: {possession?.loose_pct ?? 0}%</small>
            </div>

            <table className="stats-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Equipa</th>
                  <th>Dist. (km)</th>
                  <th>Vel. máx (km/h)</th>
                  <th>Posse (s)</th>
                </tr>
              </thead>
              <tbody>
                {(stats.players || []).map((player) => (
                  <tr key={player.tracker_id}>
                    <td>{player.tracker_id}</td>
                    <td>{teamLabel(player.team)}</td>
                    <td>{player.distance_km}</td>
                    <td>{player.max_speed_kmh}</td>
                    <td>{player.possession_seconds}</td>
                  </tr>
                ))}
              </tbody>
            </table>

            <div className="heatmap-viewer">
              <select
                aria-label="Selecionar heatmap"
                value={selected ? selected.key : "global"}
                onChange={(event) => setHeatmap(event.target.value)}
              >
                {heatmapOptions.map((option) => (
                  <option value={option.key} key={option.key}>
                    {option.label}
                  </option>
                ))}
              </select>
              {heatmapUrl ? (
                <img className="heatmap-image" src={heatmapUrl} alt={`Heatmap ${selected.label}`} />
              ) : null}
            </div>
          </>
        ) : (
          <p className="muted">A carregar estatísticas…</p>
        )}
      </div>
    </details>
  );
}
```

- [ ] **Step 3c: Render `StatsSection` inside `StatusPanel`** — in `StatusPanel` (line 636), after the `output-block` block closes (line 710, the `) : null}` that ends the `{outputUrl ? (...)` expression) and before `<ProcessingFeed .../>` (line 712), add:

```jsx
      <StatsSection job={job} />
```

- [ ] **Step 4: Run test to verify it passes**

Run (from `web/frontend/`): `npx vitest run src/App.test.jsx`
Expected: PASS (all, including the new statistics test)

- [ ] **Step 5: Commit**

```bash
git add web/frontend/src/App.jsx web/frontend/src/App.test.jsx
git commit -m "feat(stats): statistics section with possession, table and heatmaps"
```

---

## Task 10: Styles + full verification

**Files:**
- Modify: `web/frontend/src/styles.css`

- [ ] **Step 1: Append the styles** to `web/frontend/src/styles.css` (uses the existing CSS variables; check the top of the file for the exact variable names — `--panel`, `--text`, `--accent` etc. — and adjust if the names differ):

```css
.stats-panel {
  margin-top: 16px;
}

.stats-body {
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding-top: 12px;
}

.possession-block {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.possession-bar {
  display: flex;
  height: 28px;
  border-radius: 999px;
  overflow: hidden;
  background: rgba(255, 255, 255, 0.08);
}

.poss-team {
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: 600;
  color: #fff;
  white-space: nowrap;
  min-width: 0;
}

.poss-team.team-a {
  background: #2563eb;
}

.poss-team.team-b {
  background: #dc2626;
}

.stats-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

.stats-table th,
.stats-table td {
  text-align: left;
  padding: 6px 10px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}

.stats-table th {
  font-weight: 600;
  opacity: 0.7;
}

.heatmap-viewer {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.heatmap-image {
  width: 100%;
  border-radius: 12px;
  display: block;
}
```

- [ ] **Step 2: Run the full frontend suite**

Run (from `web/frontend/`): `npm test`
Expected: PASS (all suites green)

- [ ] **Step 3: Run the full backend + export suites**

Run (from `player_detection/`): `python -m pytest tests/test_web_backend.py tests/test_stats_export.py -v`
Expected: PASS

- [ ] **Step 4: Run the tracker suite** (inference venv)

Run: `& C:\VS-Projects\FootAR\FootAR_old\.venv\Scripts\python.exe -m pytest tests/test_stats_trackers.py -v`
Expected: PASS

- [ ] **Step 5: Manual end-to-end smoke check** (optional but recommended)

Start the backend (`python -m web.backend`) and frontend (`npm run dev`), process a short clip in RADAR mode, and confirm: job succeeds, the "Estatísticas" dropdown appears under the video, possession bar + table populate, and the heatmap selector swaps images. Confirm the processor did NOT open a blocking cv2 window during the web run.

- [ ] **Step 6: Commit**

```bash
git add web/frontend/src/styles.css
git commit -m "feat(stats): styles for the statistics section"
```

---

## Self-review notes

- **Spec coverage:** possession (Task 1, 9), distance/speed (Task 2, 9), heatmaps incl. per-player/ball/global (Task 3, 9), JSON contract (Task 4), processor wiring + cv2 bug gate (Task 5), backend record/command (Task 6), endpoints with manifest validation (Task 7), collapsible "Estatísticas" dropdown under the video (Task 9), styles (Task 10), tests at every layer. All spec sections mapped.
- **Type consistency:** `stats_dir` (Path) / `stats_json_path` / `stats_ready` / `stats_url` used consistently across `jobs.py`, `app.py`, tests; `--stats_output_dir` flag name identical in `main_seg.py` argparse and `jobs.build_processing_command`; `build_stats_payload`/`write_stats` signatures identical between `stats_export.py`, its tests, and the `run_radar` call site; heatmap manifest keys (`global`/`ball`/`team`/`players`/`file`) identical across tracker, export, backend whitelist, and frontend `buildHeatmapOptions`.
- **Decisions applied (from brainstorming defaults):** players table sorted by distance desc; players with no team shown as "—" (`team: null`); default heatmap = Global.
