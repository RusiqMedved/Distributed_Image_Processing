import os
import numpy as np
from PIL import Image
from server.image_math import (
    save_filter_result, laplace_filter, gaussian_blur, sobel_detection, canny_edge_detection,
    image_sharpening, laplacian_enhancement, emboss_effect, binarize_and_sketch, harris_corner_detection
)
from server.processing import (
    calculate_pipeline_margin,
    chunk_matrix_horizontally,
    merge_chunks_vertically,
    run_seq_pipeline,
    run_par_stack,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "res", "test.jpg")


def _run_filter_test(filter_func, tmp_path, filename):
    output_file = tmp_path / filename
    save_filter_result(INPUT_FILE, filter_func, str(output_file))
    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_laplace_creates_file(tmp_path):
    _run_filter_test(laplace_filter, tmp_path, "laplace_out.jpg")


def test_canny_creates_file(tmp_path):
    _run_filter_test(canny_edge_detection, tmp_path, "canny_out.jpg")


def test_gaussian_creates_file(tmp_path):
    _run_filter_test(gaussian_blur, tmp_path, "gaussian_out.jpg")


def test_sobel_creates_file(tmp_path):
    _run_filter_test(sobel_detection, tmp_path, "sobel_out.jpg")


def test_sharpen_creates_file(tmp_path):
    _run_filter_test(image_sharpening, tmp_path, "sharpen_out.jpg")


def test_laplace_enhance_creates_file(tmp_path):
    _run_filter_test(laplacian_enhancement, tmp_path, "laplace_enhance_out.jpg")


def test_emboss_creates_file(tmp_path):
    _run_filter_test(emboss_effect, tmp_path, "emboss_out.jpg")


def test_sketch_creates_file(tmp_path):
    _run_filter_test(binarize_and_sketch, tmp_path, "sketch_out.jpg")


def test_harris_creates_file(tmp_path):
    _run_filter_test(harris_corner_detection, tmp_path, "harris_out.jpg")


def test_sequential_pipeline_zero_seam():
    """
    Proves cumulative margin math eliminates chunk border seams:
    4-chunk sequential execution must match 1-chunk execution in float32.
    """
    img = Image.open(INPUT_FILE).convert("L")
    img_matrix = np.array(img)

    pipeline = ["gaussian", "sharpen", "sobel"]
    margin = calculate_pipeline_margin(pipeline, mode="sequential")
    assert margin == 4  # 2 (gaussian) + 1 (sharpen) + 1 (sobel)

    single_res = run_seq_pipeline(img_matrix, pipeline, final_raw_float=True)

    chunks_info = chunk_matrix_horizontally(img_matrix, num_chunks=4, margin=margin)
    processed_chunks = [
        (run_seq_pipeline(chunk, pipeline, final_raw_float=True), top, bot)
        for chunk, top, bot in chunks_info
    ]
    stitched_res = merge_chunks_vertically(processed_chunks)

    assert stitched_res.dtype == np.float32
    assert stitched_res.shape == single_res.shape
    assert np.allclose(single_res, stitched_res, atol=1e-5)


def test_parallel_stack_tensor_precision_and_shape():
    """
    Verifies 3D (H, W, C) float32 tensor stitching and that negative derivatives
    (e.g., Laplace) are preserved without uint8 clipping.
    """
    img = Image.open(INPUT_FILE).convert("L")
    img_matrix = np.array(img)

    filters = ["original", "sobel", "laplace", "harris"]
    margin = calculate_pipeline_margin(filters, mode="stack")
    assert margin == 3  # max(0, 1, 1, 3)

    single_tensor = run_par_stack(img_matrix, filters)

    chunks_info = chunk_matrix_horizontally(img_matrix, num_chunks=4, margin=margin)
    processed_chunks = [
        (run_par_stack(chunk, filters), top, bot)
        for chunk, top, bot in chunks_info
    ]
    stitched_tensor = merge_chunks_vertically(processed_chunks)

    assert stitched_tensor.dtype == np.float32
    assert stitched_tensor.shape == (img_matrix.shape[0], img_matrix.shape[1], len(filters))
    assert np.allclose(single_tensor, stitched_tensor, atol=1e-4)

    # Channel 0 ("original") should be normalized to [0.0, 1.0]
    assert stitched_tensor[:, :, 0].min() >= 0.0
    assert stitched_tensor[:, :, 0].max() <= 1.0

    # Channel 2 ("laplace") must preserve negative second-derivative values
    assert stitched_tensor[:, :, 2].min() < 0.0

