import io
import time
import asyncio
import httpx
import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from contextlib import asynccontextmanager
from server.image_math import FILTER_REGISTRY
from server.processing import (
    calculate_pipeline_margin,
    chunk_matrix_horizontally,
    merge_chunks_vertically,
)
from server.load_balancer import LoadBalancer

WORKER_URLS = [
    "http://worker1:8000", "http://worker2:8000",
    "http://worker3:8000", "http://worker4:8000"
]

_http_client: httpx.AsyncClient | None = None
_load_balancer: LoadBalancer | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http_client,_load_balancer
    _http_client = httpx.AsyncClient(timeout=30.0)
    _load_balancer = LoadBalancer(WORKER_URLS)
    yield
    await _http_client.aclose()

app = FastAPI(lifespan=lifespan)

async def dispatch_chunk(chunk: np.ndarray, top_margin: int, bottom_margin: int,
                         filter_type: str, strategy: str, max_retries: int = 3) ->tuple:
    chunk_b = chunk.tobytes()
    headers = {
        "X-Row-Count": str(chunk.shape[0]),
        "X-Col-Count": str(chunk.shape[1]),
        "X-Dtype": str(chunk.dtype),
        "X-Filter": filter_type
    }

    for att in range(max_retries):
        worker = await _load_balancer.get_worker(strategy)
        start_t = time.perf_counter()
        calc_t_ms = 0.0

        try:
            response = await _http_client.post(
                f"{worker.url}/compute-chunk",
                content=chunk_b,
                headers=headers
            )
            response.raise_for_status()
            calc_t_ms = float(response.headers.get("X-Computation-Time-ms", 0))
            if calc_t_ms == 0:
                calc_t_ms = (time.perf_counter() - start_t) * 1000

            res_dtype = np.dtype(response.headers["X-Dtype"])
            res_rows = int(response.headers["X-Row-Count"])
            res_cols = int(response.headers["X-Col-Count"])

            flat_arr = np.frombuffer(response.content, dtype=res_dtype)
            proc_matrix = flat_arr.reshape((res_rows,res_cols))

            return (proc_matrix,top_margin,bottom_margin)

        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            print(f"Worker {worker.url} failed: {e}. Retrying...")
            worker.is_healthy = False
        finally:
            await _load_balancer.release_worker(worker,calc_t_ms)

    raise HTTPException(status_code=502, detail="All chunk dispatch attempts failed.")

async def dispatch_pipeline_chunk(
    chunk: np.ndarray,
    top_margin: int,
    bottom_margin: int,
    filters_str: str,
    mode: str,
    raw_float: bool,
    strategy: str,
    max_retries: int = 3,
) -> tuple:
    chunk_b = chunk.tobytes()
    headers = {
        "X-Row-Count": str(chunk.shape[0]),
        "X-Col-Count": str(chunk.shape[1]),
        "X-Dtype": str(chunk.dtype),
        "X-Filters": filters_str,
        "X-Mode": mode,
        "X-Raw-Float": "true" if raw_float else "false",
    }

    for _ in range(max_retries):
        worker = await _load_balancer.get_worker(strategy)
        start_t = time.perf_counter()
        calc_t_ms = 0.0

        try:
            response = await _http_client.post(
                f"{worker.url}/compute-pipeline-chunk",
                content=chunk_b,
                headers=headers,
            )
            response.raise_for_status()
            calc_t_ms = float(response.headers.get("X-Computation-Time-ms", 0))
            if calc_t_ms == 0:
                calc_t_ms = (time.perf_counter() - start_t) * 1000

            res_dtype = np.dtype(response.headers["X-Dtype"])
            res_rows = int(response.headers["X-Row-Count"])
            res_cols = int(response.headers["X-Col-Count"])
            res_channels = response.headers.get("X-Channel-Count")

            flat_arr = np.frombuffer(response.content, dtype=res_dtype)
            if res_channels is not None:
                proc_matrix = flat_arr.reshape((res_rows, res_cols, int(res_channels)))
            else:
                proc_matrix = flat_arr.reshape((res_rows, res_cols))

            return (proc_matrix, top_margin, bottom_margin)

        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            print(f"Worker {worker.url} failed on pipeline chunk: {e}. Retrying...")
            worker.is_healthy = False
        finally:
            await _load_balancer.release_worker(worker, calc_t_ms)

    raise HTTPException(status_code=502, detail="All pipeline chunk dispatch attempts failed.")

@app.post("/process")
async def process_image(
        file: UploadFile = File(...),
        filter_type: str  =Form(),
        chunks: int = Form(1,ge=1,le=16),
        lb_strategy: str = Form("least_connections")
):
    if filter_type not in FILTER_REGISTRY:
        raise HTTPException(status_code=400, detail="Invalid filter type.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only images are allowed.")

    if filter_type not in FILTER_REGISTRY:
        raise HTTPException(status_code=400, detail="Invalid filter type.")

    img_b = await file.read()
    try:
        img = Image.open(io.BytesIO(img_b)).convert('L')
        img_matrix = np.array(img)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image data.")

    margin = FILTER_REGISTRY[filter_type]["radius"]
    chunks_info = chunk_matrix_horizontally(img_matrix, num_chunks=chunks, margin=margin)

    tasks = [dispatch_chunk(chunk,top,bot,filter_type,lb_strategy) for chunk,top,bot in chunks_info]
    try:
        processed_chunks_info = await asyncio.gather(*tasks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    result_matrix = merge_chunks_vertically(processed_chunks_info)

    result_img = Image.fromarray(result_matrix)
    buf = io.BytesIO()
    result_img.save(buf, format="JPEG")

    return Response(content=buf.getvalue(), media_type="image/jpeg")

@app.post("/pipeline")
async def process_pipeline(
    file: UploadFile = File(...),
    filters: str = Form(...),
    mode: str = Form("sequential"),
    chunks: int = Form(1, ge=1, le=16),
    output_format: str = Form("npz"),
    lb_strategy: str = Form("least_connections"),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only images are allowed.")

    filter_list = [f.strip() for f in filters.split(",") if f.strip()]
    if not filter_list:
        raise HTTPException(status_code=400, detail="Filter list cannot be empty.")

    invalid_filters = [f for f in filter_list if f not in FILTER_REGISTRY]
    if invalid_filters:
        raise HTTPException(status_code=400, detail=f"Invalid filters: {invalid_filters}")

    if mode not in ("sequential", "stack"):
        raise HTTPException(status_code=400, detail="Mode must be 'sequential' or 'stack'.")

    output_format = output_format.lower()
    if output_format not in ("jpeg", "png", "npz"):
        raise HTTPException(status_code=400, detail="output_format must be 'jpeg', 'png', or 'npz'.")

    if mode == "stack" and output_format in ("jpeg", "png") and len(filter_list) != 3:
        raise HTTPException(
            status_code=400,
            detail="Stack mode requires output_format='npz' unless exactly 3 filters are selected for RGB output.",
        )

    img_b = await file.read()
    try:
        img = Image.open(io.BytesIO(img_b)).convert("L")
        img_matrix = np.array(img)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image data.")

    margin = calculate_pipeline_margin(filter_list, mode)
    chunks_info = chunk_matrix_horizontally(img_matrix, num_chunks=chunks, margin=margin)

    normalized_filters_str = ",".join(filter_list)
    use_raw_float = output_format == "npz"

    tasks = [
        dispatch_pipeline_chunk(
            chunk, top, bot, normalized_filters_str, mode, use_raw_float, lb_strategy
        )
        for chunk, top, bot in chunks_info
    ]

    try:
        processed_chunks_info = await asyncio.gather(*tasks)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    result_matrix = merge_chunks_vertically(processed_chunks_info)

    if output_format == "npz":
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            tensor=result_matrix.astype(np.float32),
            channel_names=np.array(filter_list),
            mode=np.array(mode),
        )
        return Response(
            content=buf.getvalue(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": 'attachment; filename="pipeline_output.npz"'},
        )

    # JPEG or PNG serialization
    if result_matrix.ndim == 3:
        rgb_uint8 = np.clip(result_matrix, 0, 255).astype(np.uint8)
        result_img = Image.fromarray(rgb_uint8, mode="RGB")
    else:
        result_img = Image.fromarray(np.clip(result_matrix, 0, 255).astype(np.uint8))

    buf = io.BytesIO()
    pil_format = "PNG" if output_format == "png" else "JPEG"
    media_type = "image/png" if output_format == "png" else "image/jpeg"
    result_img.save(buf, format=pil_format)

    return Response(content=buf.getvalue(), media_type=media_type)