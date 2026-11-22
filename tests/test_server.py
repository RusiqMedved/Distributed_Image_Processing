import os
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
import io
import numpy as np
from server.worker import app as worker_app

from server.gateway import app

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(BASE_DIR, "res")

# Global client is fine for negative tests (they error out before needing the LoadBalancer)
client = TestClient(app)


# Patch the underlying httpx library directly so the lifespan doesn't overwrite it
@patch("httpx.AsyncClient.post", new_callable=AsyncMock)
def test_process_image_success(mock_post):
    # 1. Setup the fake worker response
    mock_post.return_value.status_code = 200
    # Provide a 100x100 matrix (10,000 bytes) so vertical merging doesn't crash on margin calculations
    mock_post.return_value.content = b"\x00" * 10000
    mock_post.return_value.headers = {
        "X-Computation-Time-ms": "50.0",
        "X-Row-Count": "100",
        "X-Col-Count": "100",
        "X-Dtype": "uint8"
    }
    # 2. Add the missing method that gateway.py expects
    mock_post.return_value.raise_for_status = MagicMock()

    img_path = os.path.join(RES_DIR, "test.jpg")

    # 3. Using a 'with' block forces FastAPI to run the lifespan!
    # This initializes _load_balancer so it isn't None.
    with TestClient(app) as live_client:
        with open(img_path, "rb") as f:
            response = live_client.post(
                "/process",
                files={"file": ("test.jpg", f, "image/jpeg")},
                data={"filter_type": "laplace", "chunks": 2, "lb_strategy": "round_robin"}
            )

    # 4. Assert success
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"

def test_worker_compute_pipeline_chunk_stack():
    """Tests /compute-pipeline-chunk directly on worker.py in stack mode."""
    dummy_chunk = np.random.randint(0, 256, size=(32, 32), dtype=np.uint8)
    headers = {
        "X-Row-Count": "32",
        "X-Col-Count": "32",
        "X-Dtype": "uint8",
        "X-Filters": "original,sobel,laplace",
        "X-Mode": "stack",
        "X-Raw-Float": "true",
    }

    with TestClient(worker_app) as w_client:
        response = w_client.post(
            "/compute-pipeline-chunk",
            content=dummy_chunk.tobytes(),
            headers=headers,
        )

    assert response.status_code == 200
    assert response.headers["X-Row-Count"] == "32"
    assert response.headers["X-Col-Count"] == "32"
    assert response.headers["X-Channel-Count"] == "3"
    assert response.headers["X-Dtype"] == "float32"

    tensor = np.frombuffer(response.content, dtype=np.float32).reshape((32, 32, 3))
    assert tensor.shape == (32, 32, 3)


@patch("httpx.AsyncClient.post", new_callable=AsyncMock)
def test_gateway_pipeline_npz_success(mock_post):
    """Tests /pipeline on gateway.py returning a compressed .npz 3D tensor."""
    fake_tensor_chunk = np.zeros((20, 20, 2), dtype=np.float32)

    mock_post.return_value.status_code = 200
    mock_post.return_value.content = fake_tensor_chunk.tobytes()
    mock_post.return_value.headers = {
        "X-Computation-Time-ms": "25.0",
        "X-Row-Count": "20",
        "X-Col-Count": "20",
        "X-Channel-Count": "2",
        "X-Dtype": "float32",
    }
    mock_post.return_value.raise_for_status = MagicMock()

    img_path = os.path.join(RES_DIR, "test.jpg")
    with TestClient(app) as live_client:
        with open(img_path, "rb") as f:
            response = live_client.post(
                "/pipeline",
                files={"file": ("test.jpg", f, "image/jpeg")},
                data={
                    "filters": "original,laplace",
                    "mode": "stack",
                    "chunks": 2,
                    "output_format": "npz",
                    "lb_strategy": "round_robin",
                },
            )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"

    with np.load(io.BytesIO(response.content)) as archive:
        assert "tensor" in archive
        assert archive["tensor"].dtype == np.float32
        assert archive["tensor"].ndim == 3
        assert archive["tensor"].shape[2] == 2
        assert archive["channel_names"].tolist() == ["original", "laplace"]


def test_gateway_pipeline_validation_errors():
    """Verifies HTTP 400 on unknown filters or invalid stack + jpeg combinations."""
    img_path = os.path.join(RES_DIR, "test.jpg")
    with TestClient(app) as live_client:
        with open(img_path, "rb") as f:
            bad_filter_resp = live_client.post(
                "/pipeline",
                files={"file": ("test.jpg", f, "image/jpeg")},
                data={"filters": "gaussian,not_a_filter", "mode": "sequential", "output_format": "npz"},
            )
        assert bad_filter_resp.status_code == 400

        with open(img_path, "rb") as f:
            bad_format_resp = live_client.post(
                "/pipeline",
                files={"file": ("test.jpg", f, "image/jpeg")},
                data={"filters": "original,sobel", "mode": "stack", "output_format": "jpeg"},
            )
        assert bad_format_resp.status_code == 400