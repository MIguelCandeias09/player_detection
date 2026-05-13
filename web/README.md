# FootAR Web UI

Local React + FastAPI wrapper for `src/main_seg.py`.

## Run

From `player_detection/`:

```powershell
python -m web.backend
```

Run the backend with the same Python environment used for inference. It must have
`torch`, `supervision`, `ultralytics`, `fastapi`, `uvicorn`, and
`python-multipart` installed. On this machine the working CUDA environment is:

```powershell
C:\VS-Projects\FootAR\FootAR_old\.venv\Scripts\python.exe -m web.backend
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
- `GET /api/jobs/{job_id}/output`
