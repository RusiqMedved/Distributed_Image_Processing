from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
import asyncio
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from server.image_math import laplace_filter, gaussian_blur, sobbel_detection, canny_edge_detection
from server.processing import chunk_matrix_horizontally, merge_chunks_vertically

app = FastAPI()
proc_pool = ProcessPoolExecutor()

manager = Manager()
total_images_processed = manager.Value('i', 0)  # 'i' означава integer
stats_lock = manager.Lock()

MAX_FILE_SIZE = 10 * 1024 * 1024


def apply_filter_to_chunk(task_data):
    """Worker helper function executed inside the ProcessPool."""
    chunk, top_margin, bottom_margin, filter_type = task_data

    if filter_type == "laplace":
        processed = laplace_filter(chunk)
    elif filter_type == "gaussian":
        processed = gaussian_blur(chunk)
    elif filter_type == "sobel":
        processed = sobbel_detection(chunk)
    elif filter_type == "canny":
        processed = canny_edge_detection(chunk)
    else:
        raise ValueError("Invalid filter")

    return (processed, top_margin, bottom_margin)


@app.post("/process")
async def process_image(
        file: UploadFile = File(...),
        filter_type: str = Form(...),
        chunks: int = Form(1, ge=1, le=16)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only images are allowed.")

    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413,
                            detail=f"File too large. Maximum size is {MAX_FILE_SIZE / 1024 / 1024} MB.")

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('L')
        img_matrix = np.array(img)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image file data.")

    if filter_type not in ["laplace", "gaussian", "sobel", "canny"]:
        raise HTTPException(status_code=400, detail="Invalid filter type requested.")

    margin = 2 if filter_type == "gaussian" else 1

    if chunks > img_matrix.shape[0]:
        chunks = img_matrix.shape[0]

    chunks_info = chunk_matrix_horizontally(img_matrix, num_chunks=chunks, margin=margin)
    tasks = [(chunk, top, bottom, filter_type) for chunk, top, bottom in chunks_info]

    loop = asyncio.get_running_loop()

    try:
        futures = [
            loop.run_in_executor(proc_pool, apply_filter_to_chunk, task)
            for task in tasks
        ]
        processed_chunks_info = await asyncio.gather(*futures)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error during parallel processing.")

    result_matrix = merge_chunks_vertically(processed_chunks_info)

    with stats_lock:
        total_images_processed.value += 1
        current_count = total_images_processed.value

    print(f"Server stats: Processed images count: {current_count}")

    result_img = Image.fromarray(result_matrix)
    buf = io.BytesIO()
    result_img.save(buf, format="JPEG")

    return Response(content=buf.getvalue(), media_type="image/jpeg")
