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


def test_distance_to_dict_sorted_by_tracker_id():
    t = DistanceTracker(fps=25.0)
    t._total_distance_m[7] = 1234.0
    t._max_speed_kmh[7] = 28.44
    t._total_distance_m[3] = 500.0

    rows = t.to_dict()

    assert [r["tracker_id"] for r in rows] == [3, 7]
    assert {"tracker_id": 7, "distance_km": 1.234, "max_speed_kmh": 28.4} in rows
    assert rows[0] == {"tracker_id": 3, "distance_km": 0.5, "max_speed_kmh": 0.0}
