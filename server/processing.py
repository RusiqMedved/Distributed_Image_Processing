import numpy as np


def chunk_matrix_horizontally(img_matrix: np.ndarray, num_chunks: int, margin: int):
    """
    Splits a 2D numpy array into horizontal bands with overlapping margins.
    Returns a list of tuples: (chunk_data, top_margin_size, bottom_margin_size)
    """
    height = img_matrix.shape[0]
    chunk_height = height // num_chunks
    chunks = []

    for i in range(num_chunks):
        start_row = i * chunk_height
        end_row = height if i == num_chunks - 1 else (i + 1) * chunk_height

        start_overlap = max(0, start_row - margin)
        end_overlap = min(height, end_row + margin)

        chunk = img_matrix[start_overlap:end_overlap, :].copy()

        top_margin = start_row - start_overlap
        bottom_margin = end_overlap - end_row

        chunks.append((chunk, top_margin, bottom_margin))

    return chunks


def merge_chunks_vertically(processed_chunks_info: list) -> np.ndarray:
    """
    Strips the overlapping margins from processed chunks and stitches them back together.
    """
    stripped_chunks = []

    for chunk, top_margin, bottom_margin in processed_chunks_info:
        keep_start = top_margin
        keep_end = chunk.shape[0] - bottom_margin

        core_data = chunk[keep_start:keep_end, :]
        stripped_chunks.append(core_data)

    return np.vstack(stripped_chunks)

