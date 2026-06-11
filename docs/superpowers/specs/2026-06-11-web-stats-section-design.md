# Design: Secção de Estatísticas na Web

**Data:** 2026-06-11
**Estado:** Aprovado (aguarda revisão do spec)

## Contexto e problema

As estatísticas de jogo já são calculadas durante o processamento em modo `RADAR`
por três trackers em `src/sports/`:

- `PossessionTracker` — posse por equipa (T0/T1 %), bola solta %, top jogadores por
  tempo com bola.
- `DistanceTracker` — distância percorrida (km) e velocidade máxima (km/h) por jogador.
- `HeatmapTracker` — heatmaps por equipa, por jogador, global e da bola (imagens do campo).

No fim de `run_radar` (`src/main_seg.py:1110-1113`) estes dados **só vão para o stdout**
(`print_report()`) e para uma janela cv2 (`heatmap_tracker.show()`). Nada é persistido em
ficheiro nem chega ao backend web. O frontend (`web/frontend/src/App.jsx`) não tem qualquer
secção de estatísticas.

Há ainda um **bug latente**: `heatmap_tracker.show()` faz `cv2.waitKey(0)` (bloqueante).
No subprocesso do backend web isto pode pendurar o job ou falhar.

## Objetivo

Surgir as estatísticas (posse + distâncias/velocidades + heatmaps) no frontend web, numa
secção colapsável "Estatísticas" por baixo do vídeo de resultado, visível quando o job conclui.

## Arquitetura

Feature vertical em três camadas, ligadas por ficheiros em disco (o backend web corre num
ambiente Python separado, sem as dependências pesadas de inferência — cv2/torch — pelo que a
renderização tem de acontecer no subprocesso do processador).

```
Processador (main_seg.py, env de inferência)
  └─ escreve {result_dir}/stats/stats.json + *.png
Backend web (FastAPI, env leve)
  └─ serve stats.json e PNGs via endpoints
Frontend (React)
  └─ <StatsSection> colapsável: posse, tabela jogadores, visualizador de heatmaps
```

### A. Contrato de dados

No fim do job, o processador escreve em `{result_dir}/stats/`:

**`stats.json`** (fonte única de verdade):

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

**PNGs** dos heatmaps na mesma pasta:
- Agregados sempre: `global.png`, `ball.png`, `team_0.png`, `team_1.png`.
- Um por jogador com amostras: `player_{tracker_id}.png`.

Notas:
- `team` usa as chaves `"0"`/`"1"` (equipas de campo). Guarda-redes (2/3), árbitros e bola não
  entram nas linhas de equipa/jogador do heatmap, conforme a lógica já existente.
- A lista `players` é construída no momento da exportação, juntando distância+velocidade
  (`DistanceTracker`) com tempo-de-posse e equipa (mapa `player→team` do `PossessionTracker`/
  `HeatmapTracker`). Jogadores sem equipa atribuída (ex.: só vistos pelo DistanceTracker, como
  GKs) ficam com `team: null`.

### B. Processador (`src/sports/*` + `src/main_seg.py`)

- `PossessionTracker.to_dict()` — devolve o bloco `possession` acima (deriva das estruturas
  internas que hoje alimentam `print_report()`).
- `DistanceTracker.to_dict()` — devolve `[{tracker_id, distance_km, max_speed_kmh}]`.
- `HeatmapTracker.save_heatmaps(out_dir)` — escreve os PNGs (reaproveita `render()`/`render_team`/
  `render_player`/`render_ball`/`render_global` existentes) e devolve o manifesto `heatmaps` +
  o mapa `player→team` e `player→samples` (de `list_players()`).
- `run_radar` passa a aceitar `stats_output_dir: str | None` e `headless: bool`. No fim do
  gerador:
  - constrói o `stats.json` combinado e escreve-o;
  - chama `heatmap_tracker.save_heatmaps(...)`;
  - `heatmap_tracker.show()` **só corre se `headless is False`** (corrige o bug de bloqueio).
- `main()`:
  - passa `stats_output_dir = debug_output_dir` e `headless = structured_logs` ao `run_radar`
    (no fluxo web, `structured_logs=True` ⇒ headless);
  - emite `FOOTAR_EVENT {"event":"stats","path": "stats/stats.json"}` no fim.

A escrita de `stats.json`/PNGs é defensiva (try/except com log): uma falha na exportação de
estatísticas não deve fazer o job falhar — o vídeo já está produzido.

### C. Backend (`web/backend/`)

- `JobRecord` ganha `stats_dir: Path | None` (= `debug_output_dir`); `live_frame_path`-style
  helper `stats_json_path`.
- `to_public_dict()` expõe `stats_ready` (true quando `stats.json` existe) e
  `stats_url = /api/jobs/{id}/stats` quando pronto.
- Endpoints novos em `app.py`:
  - `GET /api/jobs/{id}/stats` → conteúdo de `stats.json` (404 se não existir).
  - `GET /api/jobs/{id}/heatmap/{name}` → `FileResponse` do PNG. `name` validado: apenas
    nomes presentes no manifesto de `stats.json` (evita path traversal); media type `image/png`,
    `Cache-Control: no-store` desnecessário (ficheiro imutável por job).

### D. Frontend (`web/frontend/src/`)

- `api.js`: `fetchStats(jobId)` → GET `/api/jobs/{jobId}/stats`.
- Novo componente `StatsSection` em `App.jsx` (ou ficheiro próprio se crescer):
  - `<details className="panel stats-panel">` colapsável, `<summary>` "Estatísticas",
    renderizado dentro de `StatusPanel`, logo a seguir ao bloco `output-block` do vídeo,
    quando `job.status === "succeeded" && job.stats_ready`.
  - Faz fetch dos stats uma vez quando o job passa a `succeeded` (guardado em estado local).
  - **Barra de posse**: T0 vs T1 (largura proporcional) + etiqueta de % bola solta.
  - **Tabela de jogadores**: colunas ID · equipa (chip de cor) · distância (km) · vel. máx.
    (km/h) · tempo com bola (s). Ordenada por distância desc.
  - **Visualizador de heatmap**: `<select>` com Global / Equipa A / Equipa B / Bola / Jogador N
    → `<img src={/api/jobs/{id}/heatmap/{file}}>`. Default: Global.
- Rótulos: "Equipa A" (T0) / "Equipa B" (T1). Reutiliza o estilo `readiness-details`/`panel`
  e adiciona estilos `stats-*` em `styles.css`.

### E. Tratamento de erros

- Processador: exportação de stats é best-effort; falha não derruba o job (vídeo já existe).
- Backend: endpoints devolvem 404 limpo quando stats/PNG ausentes; `name` de heatmap validado
  contra o manifesto.
- Frontend: se `fetchStats` falhar ou `stats_ready` for false, a secção "Estatísticas" não
  aparece (ou mostra estado vazio discreto); o resto do painel funciona na mesma.

### F. Testes

- Trackers (`tests/`): unit tests de `to_dict()` do `PossessionTracker` e `DistanceTracker` com
  dados sintéticos; teste de `save_heatmaps()` a escrever PNGs + manifesto numa tmp dir.
- Backend (`tests/test_web_backend.py`): com uma pasta `stats/` falsa, validar
  `GET /stats` (JSON) e `GET /heatmap/{name}` (PNG + rejeição de nome inválido), e `stats_ready`
  no snapshot do job.
- Frontend (`App.test.jsx`): render do `StatsSection` com `fetchStats` mockado — asserir barra
  de posse, linhas da tabela e troca de heatmap no seletor.

## Decisões e trade-offs

- **PNGs pré-renderizados no processador** (em vez de render on-demand no backend): inevitável,
  porque o backend web não tem cv2/torch. Custo: ~2 MB de PNGs por job; render rápido (grid
  120×68). Aceite.
- **Heatmaps por jogador para todos os jogadores com amostras** (não só top-N): o render é
  barato e o número de jogadores é limitado; simplifica a UI (seletor lista todos).
- **Secção inline colapsável** (não um separador/rota dedicada): pedido do utilizador; mantém o
  fluxo de página única e o padrão `<details>` já usado no `SystemPanel`.

## Fora de âmbito (v-next)

- Persistência/histórico de jobs entre reinícios do backend (atualmente em memória).
- Nomes/cores reais de equipas e números de camisola (jogadores identificados por `tracker_id`).
- Estatísticas ao vivo durante o processamento (só ficam disponíveis no fim).
- Exportação CSV/PDF das estatísticas.
