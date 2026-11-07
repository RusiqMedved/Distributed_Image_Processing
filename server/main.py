from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
import numpy as np
from PIL import Image
import io
import asyncio
from concurrent.futures import ProcessPoolExecutor

from server.image_math import laplace_filter, gaussian_blur, sobbel_detection
from server.processing import chunk_matrix_horizontally, merge_chunks_vertically
app = FastAPI()

proc_pool = ProcessPoolExecutor()


def apply_filter_to_chunk(task_data):
    """Worker helper function executed inside the ProcessPool."""
    chunk, top_margin, bottom_margin, filter_type = task_data

    if filter_type == "laplace":
        processed = laplace_filter(chunk)
    elif filter_type == "gaussian":
        processed = gaussian_blur(chunk)
    elif filter_type == "sobel":
        processed = sobbel_detection(chunk)
    else:
        raise ValueError("Invalid filter")

    return (processed, top_margin, bottom_margin)
@app.post("/process")
async def process_image(file: UploadFile = File(...),
                        filter_type: str = Form(...),
                        chunks: int = Form(1)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    img_matrix = np.array(img)

    if filter_type not in ["laplace", "gaussian", "sobel"]:
        return Response(content="Invalid filter", status_code=400)

    margin = 2 if filter_type == "gaussian" else 1

    chunks_info = chunk_matrix_horizontally(img_matrix, num_chunks=chunks, margin=margin)

    tasks = [(chunk, top, bottom, filter_type) for chunk, top, bottom in chunks_info]

    loop = asyncio.get_running_loop()
    futures = [
        loop.run_in_executor(proc_pool, apply_filter_to_chunk, task)
        for task in tasks
    ]

    processed_chunks_info = await asyncio.gather(*futures)
    result_matrix = merge_chunks_vertically(processed_chunks_info)

    result_img = Image.fromarray(result_matrix)
    buf = io.BytesIO()
    result_img.save(buf, format="JPEG")

    return Response(content=buf.getvalue(), media_type="image/jpeg")

