"""FastAPI: endpoints only — wires modules, no heavy logic. Thin switchboard."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.storage import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Stemmer", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
