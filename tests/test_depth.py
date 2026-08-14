import numpy as np

from open_vocab_grasping.perception.depth import depth_buffer_to_meters, meters_to_depth_buffer


def test_opengl_depth_buffer_roundtrip() -> None:
    metric = np.array([0.05, 0.10, 0.5, 1.2, 3.0])
    buffer = meters_to_depth_buffer(metric, 0.05, 3.0)
    recovered = depth_buffer_to_meters(buffer, 0.05, 3.0)
    np.testing.assert_allclose(recovered, metric, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(buffer[[0, -1]], [0.0, 1.0], atol=1e-12)

