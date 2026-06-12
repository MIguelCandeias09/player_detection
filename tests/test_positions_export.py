import os
import sys

import numpy as np
import pytest
import supervision as sv

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from sports.configs.soccer import SoccerPitchConfiguration
from sports.positions_export import PositionsRecorder

_VERTS = np.array(SoccerPitchConfiguration().vertices, dtype=np.float32)
# Vertices espalhados e nao-colineares, todos com coords > 1 (passam a mascara)
_PICKED = [6, 9, 12, 16, 17, 20]


def identity_keypoints(shift_px: float = 0.0) -> sv.KeyPoints:
    """Keypoints cujos pixels coincidem com os vertices (homografia = identidade),
    opcionalmente deslocados em x para simular um pan de camara."""
    xy = np.zeros((1, len(_VERTS), 2), dtype=np.float32)
    conf = np.zeros((1, len(_VERTS)), dtype=np.float32)
    for i in _PICKED:
        xy[0, i] = _VERTS[i] - np.array([shift_px, 0.0], dtype=np.float32)
        conf[0, i] = 0.9
    return sv.KeyPoints(xy=xy, confidence=conf)


def detection_at(x_cm: float, y_cm: float, tracker_id: int) -> sv.Detections:
    """Detecao cujo anchor BOTTOM_CENTER cai no "pixel" (x_cm, y_cm)."""
    return sv.Detections(
        xyxy=np.array([[x_cm - 20, y_cm - 80, x_cm + 20, y_cm]], dtype=np.float32),
        class_id=np.array([2]),
        tracker_id=np.array([tracker_id]),
    )


def recorded_position(recorder: PositionsRecorder, frame_index: int, tracker_id: int):
    for frame in recorder.to_payload()["frames"]:
        if frame["i"] == frame_index:
            for player in frame["players"]:
                if player["id"] == tracker_id:
                    return player["x"], player["y"]
    raise AssertionError(f"tracker {tracker_id} sem posicao no frame {frame_index}")


def test_history_reset_after_long_gap():
    recorder = PositionsRecorder(fps=25.0, smooth_window=5)
    keypoints = identity_keypoints()
    team = np.array([0])

    # 5 frames seguidos no mesmo sitio: janela de suavizacao cheia
    for frame_index in range(1, 6):
        recorder.update(frame_index, detection_at(1000.0, 1000.0, 7), keypoints, team)

    # Reaparece ~8 s depois no centro do campo: posicoes antigas nao podem
    # puxar a media (senao o jogador "desliza" desde o sitio antigo)
    recorder.update(200, detection_at(6000.0, 3500.0, 7), keypoints, team)

    x, y = recorded_position(recorder, 200, 7)
    assert x == pytest.approx(52.5, abs=0.3)
    assert y == pytest.approx(34.0, abs=0.3)


def test_keypoints_interpolated_between_pitch_detections():
    """Jogador parado durante um pan de camara tem de ficar parado nos dados.

    O pitch detection so corre a cada N frames; entre detecoes os keypoints sao
    reutilizados (congelados). Sem interpolacao, a posicao projetada "desliza"
    com o pan e salta quando chega a detecao fresca (serrote).
    """
    recorder = PositionsRecorder(fps=25.0, smooth_window=1)
    team = np.array([0])
    ka = identity_keypoints()
    kb = identity_keypoints(shift_px=600.0)

    # Jogador parado em (6000, 3500) cm; o pixel acompanha o pan da camara.
    # Frames 1..10 reutilizam os keypoints de ka (detecao fresca so no 1 e no 11).
    recorder.update(1, detection_at(6000.0, 3500.0, 7), ka, team)
    for frame_index in range(2, 11):
        shift = 600.0 * (frame_index - 1) / 10.0
        recorder.update(frame_index, detection_at(6000.0 - shift, 3500.0, 7), ka, team)
    recorder.update(11, detection_at(5400.0, 3500.0, 7), kb, team)

    for frame_index in (1, 6, 11):
        x, y = recorded_position(recorder, frame_index, 7)
        assert x == pytest.approx(52.5, abs=0.3), f"frame {frame_index}: x={x}"
        assert y == pytest.approx(34.0, abs=0.3), f"frame {frame_index}: y={y}"


def test_roster_uses_majority_team_vote():
    """A equipa de cada tracker e decidida por voto maioritario, nao pelo
    ultimo frame visto (a classificacao oscila pontualmente)."""
    recorder = PositionsRecorder(fps=25.0, smooth_window=1)
    keypoints = identity_keypoints()

    recorder.update(1, detection_at(6000.0, 3500.0, 7), keypoints, np.array([0]))
    recorder.update(2, detection_at(6000.0, 3500.0, 7), keypoints, np.array([0]))
    recorder.update(3, detection_at(6000.0, 3500.0, 7), keypoints, np.array([1]))
    recorder.update(4, detection_at(6000.0, 3500.0, 7), keypoints, np.array([0]))

    payload = recorder.to_payload()
    assert payload["roster"] == [{"id": 7, "t": 0, "c": 2}]


def test_frame_players_carry_only_id_and_position():
    """Equipa/classe vivem no roster; as entradas por frame ficam compactas."""
    recorder = PositionsRecorder(fps=25.0, smooth_window=1)
    recorder.update(1, detection_at(6000.0, 3500.0, 7), identity_keypoints(), np.array([0]))

    payload = recorder.to_payload()
    assert set(payload["frames"][0]["players"][0].keys()) == {"id", "x", "y"}


def test_short_lived_id_is_stitched_to_dead_track():
    """Um id novo que nasce pouco depois e perto de um id morto e o mesmo
    jogador re-identificado pelo tracker: herda o id antigo no payload."""
    recorder = PositionsRecorder(fps=25.0, smooth_window=1)
    keypoints = identity_keypoints()
    team = np.array([0])

    # tid 7 vive frames 1..5 e morre em (6000, 3500)
    for frame_index in range(1, 6):
        recorder.update(frame_index, detection_at(6000.0, 3500.0, 7), keypoints, team)
    # tid 99 nasce 0.8 s depois a ~0.9 m
    for frame_index in range(25, 30):
        recorder.update(frame_index, detection_at(6100.0, 3500.0, 99), keypoints, team)

    payload = recorder.to_payload()
    ids = {p["id"] for frame in payload["frames"] for p in frame["players"]}
    assert ids == {7}
    assert payload["roster"] == [{"id": 7, "t": 0, "c": 2}]


def test_distant_new_id_stays_separate():
    recorder = PositionsRecorder(fps=25.0, smooth_window=1)
    keypoints = identity_keypoints()
    team = np.array([0])

    for frame_index in range(1, 6):
        recorder.update(frame_index, detection_at(6000.0, 3500.0, 7), keypoints, team)
    # Nasce a ~26 m do id morto: jogador diferente, nao herda
    for frame_index in range(25, 30):
        recorder.update(frame_index, detection_at(9000.0, 3500.0, 99), keypoints, team)

    payload = recorder.to_payload()
    ids = {p["id"] for frame in payload["frames"] for p in frame["players"]}
    assert ids == {7, 99}


def test_consecutive_frames_keep_smoothing():
    recorder = PositionsRecorder(fps=25.0, smooth_window=5)
    keypoints = identity_keypoints()
    team = np.array([0])

    recorder.update(1, detection_at(1000.0, 3500.0, 7), keypoints, team)
    recorder.update(2, detection_at(2000.0, 3500.0, 7), keypoints, team)

    # Sem gap, a media móvel mantem-se: (1000+2000)/2 = 1500 cm -> 13.13 m
    x, _ = recorded_position(recorder, 2, 7)
    assert x == pytest.approx(13.13, abs=0.1)
