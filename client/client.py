import io
import os
import requests
import numpy as np
import time

def send_image(input_path: str,
               filter_type: str,
               output_path: str,
               chunks: int = 4,
               max_retries: int = 3,
               timeout: int = 15):

    url = "http://127.0.0.1:8000/process"

    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        return

    print(f"Sending {input_path} to server (Filter: {filter_type}, Chunks: {chunks})...")

    for attempt in range(1, max_retries + 1):
        try:
            with open(input_path, 'rb') as img_file:
                files_ = {'file': img_file}
                data_ = {'filter_type': filter_type, 'chunks': chunks}

                response = requests.post(url, files=files_, data=data_, timeout=timeout)
                response.raise_for_status()

            with open(output_path, 'wb') as out_file:
                out_file.write(response.content)

            print(f"Success! Processed image saved to {output_path}")
            return

        except requests.exceptions.Timeout:
            print(f"Attempt {attempt}/{max_retries} failed: Request timed out after {timeout} seconds.")
        except requests.exceptions.ConnectionError:
            print(f"Attempt {attempt}/{max_retries} failed: Could not connect to the server. Is it running?")
        except requests.exceptions.HTTPError as e:
            print(f"Attempt {attempt}/{max_retries} failed: HTTP error: {e}")
            err_response = e.response
            if err_response is not None:
                print(f"Server message: {err_response.text}")
                if 400 <= err_response.status_code < 500:
                    print("Client error detected. Aborting retries.")
                    break
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt}/{max_retries} failed: Network error: {e}")

        if attempt < max_retries:
            wait_time = 2 ** attempt
            print(f"Retrying in {wait_time} seconds...\n")
            time.sleep(wait_time)
        else:
            print("Max retries reached. Image processing failed.")
def request_pipeline(
    image_path: str,
    filters: list[str],
    mode: str = "stack",
    chunks: int = 4,
    output_format: str = "npz",
    output_path: str | None = None,
    lb_strategy: str = "least_connections",
    url: str = "http://127.0.0.1:8000/pipeline",
    timeout: int = 30,
):
    """
    Sends an image to /pipeline.
    - When output_format == 'npz': returns (tensor, channel_names) as NumPy arrays
      (and optionally writes the .npz file to output_path).
    - When output_format is 'jpeg' or 'png': saves the image to output_path and returns raw bytes.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input file '{image_path}' not found.")

    filters_str = ",".join(filters)
    data = {
        "filters": filters_str,
        "mode": mode,
        "chunks": str(chunks),
        "output_format": output_format,
        "lb_strategy": lb_strategy,
    }

    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
        response = requests.post(url, files=files, data=data, timeout=timeout)

    response.raise_for_status()

    if output_path:
        with open(output_path, "wb") as out_f:
            out_f.write(response.content)

    if output_format.lower() == "npz":
        with np.load(io.BytesIO(response.content)) as archive:
            tensor = archive["tensor"]
            channel_names = archive["channel_names"].tolist()
        return tensor, channel_names

    return response.content

