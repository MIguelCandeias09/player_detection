import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from sports.common.view import ViewTransformer


def _affine_target(points):
    # Mapeamento "ground truth" estilo pixels->cm do campo: escala 6x + translacao
    return points * 6.0 + np.array([1000.0, 500.0], dtype=np.float32)


def test_outlier_correspondence_does_not_skew_projection():
    # Grelha 4x2 de keypoints "detetados" (pixels) com correspondencias exatas
    source = np.array(
        [
            [100.0, 200.0], [600.0, 200.0], [1100.0, 200.0], [1600.0, 200.0],
            [100.0, 700.0], [600.0, 700.0], [1100.0, 700.0], [1600.0, 700.0],
        ],
        dtype=np.float32,
    )
    target = _affine_target(source).astype(np.float32)

    # Um keypoint mal detetado: pixel deslocado 800px, rotulo (target) correto
    corrupted = source.copy()
    corrupted[7] += np.array([800.0, -300.0], dtype=np.float32)

    transformer = ViewTransformer(source=corrupted, target=target)

    probe = np.array([[850.0, 450.0]], dtype=np.float32)
    expected = _affine_target(probe)
    error_cm = float(np.linalg.norm(transformer.transform_points(points=probe) - expected))

    # Com rejeicao de outliers o erro no centro deve ser desprezavel (<1 m)
    assert error_cm < 100.0, f"projecao desviada {error_cm:.1f} cm pelo outlier"


def test_exact_homography_with_four_points():
    # Com apenas 4 pontos nao ha redundancia: mapeamento exato, sem RANSAC
    source = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]], dtype=np.float32)
    target = source * 10.0

    transformer = ViewTransformer(source=source, target=target)
    out = transformer.transform_points(points=np.array([[50.0, 50.0]], dtype=np.float32))

    assert np.allclose(out, [[500.0, 500.0]], atol=1e-2)
