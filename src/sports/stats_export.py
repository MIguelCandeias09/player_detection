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
