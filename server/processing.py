import numpy as np
from server.image_math import FILTER_REGISTRY


def apply_filter_to_chunk(task_data: tuple) -> tuple:
    """Worker helper function executed inside the ProcessPool."""
    chunk, top_margin, bottom_margin, filter_type = task_data

    entry = FILTER_REGISTRY.get(filter_type)
    if entry is None:
        raise ValueError(f"Invalid filter: {filter_type}")

    processed = entry["func"](chunk)
    return (processed, top_margin, bottom_margin)


def chunk_matrix_horizontally(img_matrix: np.ndarray, num_chunks: int, margin: int) -> list:
    """
    Splits a 2D numpy array into balanced horizontal bands with overlapping margins.
    Returns a list of tuples: (chunk_data, top_margin_size, bottom_margin_size)
    """
    height = img_matrix.shape[0]
    num_chunks = max(1, min(num_chunks, height))
    chunks = []

    for i in range(num_chunks):
        start_row = (i * height) // num_chunks
        end_row = ((i + 1) * height) // num_chunks

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
        keep_end = chunk.shape[0] - bottom_margin if bottom_margin > 0 else chunk.shape[0]

        core_data = chunk[keep_start:keep_end, :]
        stripped_chunks.append(core_data)

    return np.concatenate(stripped_chunks, axis=0)

def calculate_pipeline_margin(filter_names: list[str], mode:str)->int:
    if not filter_names:
        raise ValueError("Filter list cannot be empty.")

    radii = []
    for name in filter_names:
        entry = FILTER_REGISTRY.get(name)
        if entry is None:
            raise ValueError(f"Invalid filter: {name}")
        radii.append(entry["radius"])

    if mode == "sequential":
        return sum(radii)
    elif mode == "stack":
        return max(radii)
    else:
        raise ValueError(f"Invalid execution mode: {mode}")

def run_seq_pipeline(
        chunk: np.ndarray,
        filter_names: list[str],
        final_raw_float: bool = False
)->np.ndarray:
    if not filter_names:
        raise ValueError("Filter list cannot be empty.")

    cur = chunk
    last_id = len(filter_names) - 1

    for i,name in enumerate(filter_names):
        entry = FILTER_REGISTRY.get(name)
        if entry is None:
            raise ValueError(f"Invalid filter: {name}")

        use_raw_float = final_raw_float if i == last_id else True
        cur = entry["func"](cur, raw_float=use_raw_float)

    return cur

def run_par_stack(chunk: np.ndarray, filter_names: list[str])->np.ndarray:
    if not filter_names:
        raise ValueError("Filter list cannot be empty.")

    levels = []
    for name in filter_names:
        entry = FILTER_REGISTRY.get(name)
        if entry is None:
            raise ValueError(f"Invalid filter: {name}")
        feature_map = entry["func"](chunk, raw_float=True)
        levels.append(feature_map)

    return np.stack(levels, axis=-1).astype(np.float32)