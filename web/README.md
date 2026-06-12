# FootAR Web UI

Local React + FastAPI wrapper for `src/main_seg.py`.

## Run

From `player_detection/`:

```powershell
python -m web.backend
```

The backend itself only needs `fastapi`, `uvicorn`, and `python-multipart`. The
inference processor (`src/main_seg.py`) runs in a separate Python that must have
`torch`, `supervision`, and `ultralytics` installed.

The processor Python is resolved in this order:

1. `FOOTAR_PYTHON` environment variable
2. `FOOTAR_PYTHON` entry in `player_detection/.env` (copy `.env.example` to `.env`)
3. Auto-detected known venvs (project `.venv`, `FootAR_V2/.venv`, `FootAR_old/.venv`)
4. The backend's own Python (fallback)

Recommended setup — create a `.env` once per machine:

```powershell
Copy-Item .env.example .env
# edit .env and point FOOTAR_PYTHON at your inference venv
```

Or override per-session:

```powershell
$env:FOOTAR_PYTHON="C:\VS-Projects\FootAR\FootAR_old\.venv\Scripts\python.exe"
python -m web.backend
```

From `player_detection/web/frontend/`:

```powershell
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

## Required model files

The API preflight checks the model paths currently referenced by `src/main_seg.py`:

- `models/active/yolo11m_seg_players.pt`
- `models/active/pitch_v11m_640_footar_best.pt`
- `models/active/ball_y11m_1280_footar_best.pt`

If any are missing, the UI disables processing and shows the missing paths.

## API

- `GET /api/system`
- `POST /api/jobs`
- `GET /api/jobs/{job_id}`
- `POST /api/jobs/{job_id}/cancel`
- `GET /api/jobs/{job_id}/live-frame`
- `GET /api/jobs/{job_id}/live-stream`
- `GET /api/jobs/{job_id}/output`
