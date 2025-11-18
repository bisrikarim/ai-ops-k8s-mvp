"""Collector service – enriches Alertmanager webhooks, stores embeddings, and publishes to NATS."""

import json
import os
from uuid import uuid4

import httpx
from fastapi import FastAPI, status
from pydantic import BaseModel, Field
from nats.aio.client import Client as NATS

app = FastAPI(title="AI-Ops Collector")

# Config
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
CHROMA_URL = os.getenv("CHROMA_URL", "http://chromadb.ai-ops:8000")
NATS_URL = os.getenv("NATS_URL", "nats://nats.ai-ops.svc.cluster.local:4222")
NATS_SUBJECT = os.getenv("NATS_SUBJECT", "observability.raw")


class AlertIn(BaseModel):
    status: str
    alerts: list[dict]
    groupLabels: dict = Field(default_factory=dict)
    commonLabels: dict = Field(default_factory=dict)
    commonAnnotations: dict = Field(default_factory=dict)
    receiver: str | None = None


class NATSClient:
    _nc: NATS | None = None

    @classmethod
    async def get(cls) -> NATS:
        if cls._nc and cls._nc.is_connected:
            return cls._nc
        cls._nc = NATS()
        await cls._nc.connect(servers=[NATS_URL])
        return cls._nc


async def generate_embedding(text: str) -> list[float]:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text},
        )
        r.raise_for_status()
        return r.json()["embedding"]


async def upsert_chroma(vector: list[float], metadata: dict):
    collection = "alerts"
    async with httpx.AsyncClient(timeout=30) as client:
        await client.post(f"{CHROMA_URL}/api/v1/collections", json={"name": collection})
        await client.post(
            f"{CHROMA_URL}/api/v1/collections/{collection}/points",
            json={
                "ids": [metadata["id"]],
                "embeddings": [vector],
                "metadatas": [metadata],
            },
        )


def summarize(alert: AlertIn) -> tuple[str, dict]:
    first = alert.alerts[0]
    labels = first.get("labels", {})
    annotations = first.get("annotations", {})
    summary = (
        f"{labels.get('alertname')} in ns {labels.get('namespace')} – "
        f"status {alert.status}. Msg: {annotations.get('description','')}"
    )
    meta = {
        "id": str(uuid4()),
        "alertname": labels.get("alertname"),
        "namespace": labels.get("namespace"),
        "status": alert.status,
        "labels": json.dumps(labels),
        "annotations": json.dumps(annotations),
    }
    return summary, meta


@app.post("/alert", status_code=status.HTTP_202_ACCEPTED)
async def handle_alert(alert: AlertIn):
    summary, meta = summarize(alert)

    # 1) Embedding
    vector = await generate_embedding(summary)

    # 2) Store in ChromaDB
    await upsert_chroma(vector, meta)

    # 3) Publish on NATS
    nc = await NATSClient.get()
    await nc.publish(NATS_SUBJECT, json.dumps({"summary": summary, **meta}).encode())

    return {"status": "queued"}
