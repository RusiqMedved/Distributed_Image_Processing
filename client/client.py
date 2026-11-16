import requests
import os
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
            if response is not None:
                print(f"Server message: {response.text}")

            if response.status_code == 400:
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


