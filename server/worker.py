import os
import time
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response, Header, HTTPException
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from server.image_math import FILTER_REGISTRY
from server.processing import run_seq_pipeline, run_par_stack

_proc_pool: ProcessPoolExecutor | None = None

def execute_filter(img_matrix: np.ndarray, filter_name: str) -> np.ndarray:
    entry = FILTER_REGISTRY.get(filter_name)
    if not entry:
        raise ValueError(f"Invalid filter: {filter_name}")
    return entry["func"](img_matrix)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _proc_pool
    _proc_pool = ProcessPoolExecutor()
    yield
    if _proc_pool:
        _proc_pool.shutdown(wait=True)

app = FastAPI(lifespan = lifespan)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "cpu_count": os.cpu_count() or 1
    }

@app.post("/compute-chunk")
async def compute_chunk(
    request: Request,
    x_row_count: int = Header(..., alias="X-Row-Count"),
    x_col_count: int = Header(..., alias="X-Col-Count"),
    x_dtype: str = Header(..., alias="X-Dtype"),
    x_filter: str = Header(..., alias="X-Filter")
):
    if x_filter not in FILTER_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown filter: {x_filter}")

    raw_b = await request.body()
    try:
        flat_arr = np.frombuffer(raw_b,dtype=np.dtype(x_dtype))
        img_matrix = flat_arr.reshape((x_row_count,x_col_count))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Matrix reconstruction failed: {e}")

    if x_filter not in FILTER_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown filter: {x_filter}")

    loop = asyncio.get_running_loop()
    start_time = time.perf_counter()

    try:
        processed_matrix = await loop.run_in_executor(
            _proc_pool,
            execute_filter,
            img_matrix,
            x_filter
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Worker computation failed")

    calc_time_ms = (time.perf_counter() - start_time) * 1000

    return Response(
        content=processed_matrix.tobytes(),
        media_type="application/octet-stream",
        headers={
            "X-Computation-Time-ms": f"{calc_time_ms:.2f}",
            "X-Row-Count": str(processed_matrix.shape[0]),
            "X-Col-Count": str(processed_matrix.shape[1]),
            "X-Dtype": str(processed_matrix.dtype)
        }
    )

@app.post("/compute-pipeline-chunk")
async def compute_pipeline_chunk(
request: Request,
    x_row_count: int = Header(..., alias="X-Row-Count"),
    x_col_count: int = Header(..., alias="X-Col-Count"),
    x_dtype: str = Header(..., alias="X-Dtype"),
    x_filters: str = Header(..., alias="X-Filters"),
    x_mode: str = Header(..., alias="X-Mode"),
    x_raw_float: bool = Header(False, alias="X-Raw-Float"),
):
    raw_b = await request.body()
    try:
        flat_arr = np.frombuffer(raw_b, dtype=np.dtype(x_dtype))
        img_matrix = flat_arr.reshape((x_row_count, x_col_count))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Matrix reconstruction failed: {e}")
    filter_names = [f.strip() for f in x_filters.split(",") if f.strip()]

    if not filter_names:
        raise HTTPException(status_code=400, detail="X-Filters header cannot be empty.")

    invalid_filters = [f for f in filter_names if f not in FILTER_REGISTRY]
    if invalid_filters:
        raise HTTPException(status_code=400, detail=f"Unknown filters: {invalid_filters}")

    if x_mode not in ("sequential", "stack"):
        raise HTTPException(status_code=400, detail=f"Invalid mode: {x_mode}")

    loop = asyncio.get_running_loop()
    start_time = time.perf_counter()

    try:
        if x_mode == "sequential":
            processed_matrix = await loop.run_in_executor(
                _proc_pool,
                run_seq_pipeline,
                img_matrix,
                filter_names,
                x_raw_float,
            )
        else:
            processed_matrix = await loop.run_in_executor(
                _proc_pool,
                run_par_stack,
                img_matrix,
                filter_names,
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Worker pipeline computation failed: {e}")

    calc_time_ms = (time.perf_counter() - start_time) * 1000

    response_headers = {
        "X-Computation-Time-ms": f"{calc_time_ms:.2f}",
        "X-Row-Count": str(processed_matrix.shape[0]),
        "X-Col-Count": str(processed_matrix.shape[1]),
        "X-Dtype": str(processed_matrix.dtype),
    }

    if processed_matrix.ndim == 3:
        response_headers["X-Channel-Count"] = str(processed_matrix.shape[2])

    return Response(
        content=processed_matrix.tobytes(),
        media_type="application/octet-stream",
        headers=response_headers,
    )