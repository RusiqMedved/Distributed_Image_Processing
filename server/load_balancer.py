import asyncio
from typing import List


class WorkerNode():
    def __init__(self,url: str):
       self.url = url
       self.is_healthy = True
       self.active_connections = 0
       self.avg_latency_ms = 100.0
       self.total_processed = 0

class LoadBalancer:
    def __init__(self,worker_urls: List[str]):
        self.workers = [WorkerNode(url) for url in worker_urls]
        self.lock = asyncio.Lock()
        self.cur_index = 0

    async def get_worker(self, strategy: str) -> WorkerNode:
        async with self.lock:
            healthy_workers = [w for w in self.workers if w.is_healthy]
            if not healthy_workers:
                raise RuntimeError("Cluster failure: No healthy workers available")

            if strategy == "round_robin":
                worker = healthy_workers[self.cur_index % len(healthy_workers)]
                self.cur_index +=1
            elif strategy == "least_connections":
                worker = min(healthy_workers, key=lambda w: w.active_connections)
            elif strategy == "weighted":
                worker = min(healthy_workers, key=lambda w: (w.active_connections + 1) * w.avg_latency_ms)
            else:
                raise ValueError(f"Unknown routing strategy: {strategy}")

            worker.active_connections += 1
            return worker

    async def release_worker(self, worker : WorkerNode, measured_latency_ms : float):
        async with self.lock:
            worker.active_connections = max(0,worker.active_connections-1)
            worker.total_processed +=1
            worker.avg_latency_ms = (0.8 * worker.avg_latency_ms) + (0.2 * measured_latency_ms)



