"""Recommendation API over a serving bundle (src/serving.py), plus the demo page in web/.

Usage:
    BUNDLE_DIR=models/serve/full-main-6ep uvicorn src.serve:app --port 8000

Endpoints:
    POST /recommend           {customer_id, k, repeat?} -> top-k articles with their source and per-stage timings
    GET  /customers/sample    random customer ids (with a purchase history unless include_new=true)
    GET  /customers/{id}      profile and most recent purchases as of the cutoff
    GET  /health              liveness plus what is loaded
"""

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import faiss
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.evaluate import merge_repeat
from src.serving import MAX_REPEAT, ServingBundle, augmented_queries

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
state: dict = {}
# Handlers are `async def` although they are CPU-bound: each takes about a millisecond, so running them in order on
# the event loop beats FastAPI's thread pool, where concurrent requests contend for the GIL and the tail grows.


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One request = one small forward pass and one search; extra threads only add contention under concurrency.
    torch.set_num_threads(1)
    faiss.omp_set_num_threads(1)
    start = time.perf_counter()
    state["bundle"] = ServingBundle(os.environ.get("BUNDLE_DIR", "bundle"))
    state["load_seconds"] = round(time.perf_counter() - start, 2)
    state["with_history"] = np.flatnonzero(state["bundle"].n_prior > 0)
    state["articles"] = article_records(state["bundle"])
    yield
    state.clear()


app = FastAPI(title="Two-tower recommendations", lifespan=lifespan)


class RecommendRequest(BaseModel):
    customer_id: str
    k: int = Field(12, ge=1, le=100)
    repeat: int | None = Field(None, ge=0, le=MAX_REPEAT, description="past purchases first; default from the bundle")


def article_records(bundle: ServingBundle) -> list[dict]:
    """Display fields per item row, built once: a pandas row lookup per item costs more than the model."""
    a = bundle.articles
    return [{"article_id": f"{aid:010d}", "name": name, "type": typ, "group": group, "colour": colour,
             "department": dept, "description": desc}
            for aid, name, typ, group, colour, dept, desc in zip(
                a["article_id"].tolist(), a["prod_name"], a["product_type_name"], a["product_group_name"],
                a["colour_group_name"], a["index_group_name"], a["detail_desc"])]


def article(row: int) -> dict:
    return state["articles"][row]


def customer_index(bundle: ServingBundle, customer_id: str) -> int:
    idx = bundle.lookup(customer_id)
    if idx is None:
        raise HTTPException(404, f"unknown customer_id {customer_id!r}")
    return idx


@app.post("/recommend")
async def recommend(req: RecommendRequest) -> dict:
    bundle: ServingBundle = state["bundle"]
    t0 = time.perf_counter()
    idx = np.array([customer_index(bundle, req.customer_id)])
    repeat = min(bundle.repeat if req.repeat is None else req.repeat, req.k)
    user_vec = bundle.user_vectors(idx)
    t1 = time.perf_counter()
    _, ranked = bundle.search(user_vec, req.k + repeat)
    t2 = time.perf_counter()
    own = bundle.repeat_lists(idx, repeat)
    rows = merge_repeat(ranked, own, repeat, req.k)[0] if repeat else ranked[0, :req.k]
    scores = bundle.item_vectors[rows] @ augmented_queries(user_vec)[0]  # blended score, also for past purchases
    past = set(own[0].tolist())
    items = [{**article(r), "source": "repeat" if r in past else "model", "score": round(float(s), 4)}
             for r, s in zip(rows.tolist(), scores)]
    t3 = time.perf_counter()
    return {
        "customer_id": req.customer_id, "as_of": bundle.config["cutoff_date"], "items": items,
        "timing_ms": {"user_tower": round((t1 - t0) * 1000, 2), "search": round((t2 - t1) * 1000, 2),
                      "merge": round((t3 - t2) * 1000, 2), "total": round((t3 - t0) * 1000, 2)},
    }


@app.get("/customers/sample")
async def sample_customers(n: int = 8, include_new: bool = False) -> dict:
    bundle: ServingBundle = state["bundle"]
    pool = np.arange(bundle.n_customers) if include_new else state["with_history"]
    picked = np.random.default_rng().choice(pool, min(n, 1000, len(pool)), replace=False)
    return {"customer_ids": [bundle.customer_ids[i].decode() for i in picked]}


@app.get("/customers/{customer_id}")
async def customer(customer_id: str) -> dict:
    bundle: ServingBundle = state["bundle"]
    idx = customer_index(bundle, customer_id)
    n_prior = int(bundle.n_prior[idx])
    recent = bundle.hist_rows[idx][::-1]  # newest first
    return {
        "customer_id": customer_id, "as_of": bundle.config["cutoff_date"], "purchases": n_prior,
        "days_since_last": int(bundle.cutoff_day - bundle.last_day[idx]) if n_prior else None,
        "recent": [article(r) for r in recent[recent >= 0].tolist()],
    }


@app.get("/health")
async def health() -> dict:
    bundle: ServingBundle = state["bundle"]
    return {"status": "ok", "model": bundle.config["run_name"], "as_of": bundle.config["cutoff_date"],
            "index": bundle.index_config, "repeat": bundle.repeat, "alpha": bundle.alpha,
            "customers": bundle.n_customers, "items": bundle.n_items, "load_seconds": state["load_seconds"],
            "test_metrics": bundle.config.get("test_metrics")}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")
