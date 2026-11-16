import os
import pytest
import requests
from unittest.mock import patch, Mock
from client.client import send_image


@patch("client.client.requests.post")
def test_send_image_success(mock_post, tmp_path):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.content = b"fake_image_bytes"
    mock_response.raise_for_status = Mock()
    mock_post.return_value = mock_response

    input_path = "tests/res/test.jpg"
    output_path = tmp_path / "mock_out.jpg"

    send_image(input_path, "gaussian", str(output_path))

    mock_post.assert_called_once()

    assert os.path.exists(output_path)
    with open(output_path, "rb") as f:
        assert f.read() == b"fake_image_bytes"


def test_send_image_file_not_found(capsys):
    send_image("nonexistent_image.jpg", "sobel", "out.jpg")

    captured = capsys.readouterr()
    assert "Error: Input file 'nonexistent_image.jpg' not found." in captured.out


@patch("client.client.requests.post")
@patch("client.client.time.sleep") # Mock sleep so the test runs instantly
def test_send_image_retry_success(mock_sleep, mock_post, tmp_path):
    # Setup a successful mock response
    mock_success = Mock()
    mock_success.status_code = 200
    mock_success.content = b"fake_image_bytes"
    mock_success.raise_for_status = Mock()

    # Make the mock fail the first time (ConnectionError), then succeed the second time
    mock_post.side_effect = [requests.exceptions.ConnectionError("Server offline"), mock_success]

    input_path = "tests/res/test.jpg"
    output_path = tmp_path / "mock_out.jpg"

    # Call the function
    send_image(input_path, "gaussian", str(output_path), chunks=4, max_retries=3, timeout=5)

    # Assertions
    assert mock_post.call_count == 2 # Proves the retry loop worked!
    mock_sleep.assert_called_once()  # Proves exponential backoff triggered once
    assert os.path.exists(output_path)