import requests
import time
import os

import time
import requests

# URL = "http://127.0.0.1:8000/process"
# TEST_IMAGE = "test_image.jpg"  # Ensure you have a large test image in the directory
# FILTER = "canny"  # Use a heavy filter to generate CPU load
#
# strategies = ["round_robin", "least_connections", "weighted"]
# chunk_counts = [4, 8, 12, 16]
#
# print(f"{'Chunks':<8} | {'Strategy':<20} | {'Time (s)':<10}")
# print("-" * 45)
#
# for chunks in chunk_counts:
#     for strategy in strategies:
#         with open(TEST_IMAGE, "rb") as f:
#             files = {"file": f}
#             data = {
#                 "filter_type": FILTER,
#                 "chunks": chunks,
#                 "lb_strategy": strategy
#             }
#
#             start_time = time.perf_counter()
#             response = requests.post(URL, files=files, data=data)
#             elapsed = time.perf_counter() - start_time
#
#             if response.status_code == 200:
#                 print(f"{chunks:<8} | {strategy:<20} | {elapsed:.3f}s")
#             else:
#                 print(f"{chunks:<8} | {strategy:<20} | FAILED ({response.status_code})")

# def run_benchmark(input_path: str, filter_type: str, runs_per_config: int = 5):
#     url = "http://127.0.0.1:8000/process"
#     worker_configs = [1, 2, 4, 8]
#
#     if not os.path.exists(input_path):
#         print(f"Error: {input_path} not found.")
#         return
#
#     print(f"Starting benchmark for '{filter_type}' filter using {input_path}")
#     print(f"Averaging {runs_per_config} runs per configuration...\n")
#     print(f"{'Workers (N)':<15} | {'Avg Time (s)':<15} | {'Speedup (Sp)':<15}")
#     print("-" * 50)
#
#     avg_times = {}
#
#     for workers in worker_configs:
#         total_time = 0.0
#
#         for _ in range(runs_per_config):
#             with open(input_path, 'rb') as img_file:
#                 files = {'file': img_file}
#                 data = {'filter_type': filter_type, 'chunks': workers}
#
#                 start_time = time.perf_counter()
#                 response = requests.post(url, files=files, data=data)
#                 end_time = time.perf_counter()
#
#                 response.raise_for_status()
#                 total_time += (end_time - start_time)
#
#         avg_time = total_time / runs_per_config
#         avg_times[workers] = avg_time
#
#         t1 = avg_times[1]
#         speedup = t1 / avg_time
#
#         print(f"{workers:<15} | {avg_time:<15.4f} | {speedup:<15.4f}")
#
#     print("\nBenchmark complete. Use these Sp values to calculate Amdahl's Law theoretical limits.")
#
#
# if __name__ == "__main__":
#     project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     test_image = os.path.join(project_root, "tests", "res", "test.jpg")
#     run_benchmark(test_image, filter_type="sobel", runs_per_config=3)