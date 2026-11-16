import pytest
from fastapi.testclient import TestClient
from server.main import app
import io

client = TestClient(app)

def test_process_image_success():
    img_path = "tests/res/test.jpg"
    with open(img_path, "rb") as f:
        response = client.post(
            "/process",
            files={"file": ("test.jpg", f, "image/jpeg")},
            data={"filter_type": "laplace", "chunks": 2}
        )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert len(response.content) > 0

def test_process_image_invalid_filter():
    img_path = "tests/res/test.jpg"
    with open(img_path, "rb") as f:
        response = client.post(
            "/process",
            files={"file": ("test.jpg", f, "image/jpeg")},
            data={"filter_type": "invalid_filter", "chunks": 1}
        )
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid filter type requested."

def test_process_image_non_image_file():
    fake_file = io.BytesIO(b"print('malicious code')")
    response = client.post(
        "/process",
        files={"file": ("script.py", fake_file, "text/x-python")},
        data={"filter_type": "sobel", "chunks": 1}
    )
    assert response.status_code == 400
    assert "Only images are allowed" in response.json()["detail"]

def test_process_image_corrupted_data():
    garbage_bytes = io.BytesIO(b"this is not a real image format")
    response = client.post(
        "/process",
        files={"file": ("fake.jpg", garbage_bytes, "image/jpeg")},
        data={"filter_type": "gaussian", "chunks": 1}
    )
    assert response.status_code == 400
    assert "corrupted" in response.json()["detail"]