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
