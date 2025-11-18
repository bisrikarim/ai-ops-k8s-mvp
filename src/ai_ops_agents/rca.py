"""RCA service – consumes alerts from NATS, queries similar incidents, prompts LLM, publishes diagnosis."""

import asyncio
import json
import os

import httpx
from nats.aio.client import Client as NATS


# Config
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
CHROMA_URL = os.getenv("CHROMA_URL", "http://localhost:8000")
NATS_URL = os.getenv("NATS_URL", "nats://localhost:4222")
NATS_SUBJECT_IN = os.getenv("NATS_SUBJECT_IN", "observability.raw")
NATS_SUBJECT_OUT = os.getenv("NATS_SUBJECT_OUT", "ai.rca")


async def generate_embedding(text: str) -> list[float]:
    """Generate embedding for similarity search."""
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text},
        )
        r.raise_for_status()
        return r.json()["embedding"]


async def query_similar(vector: list[float], top_k: int = 5) -> list[dict]:
    """Query ChromaDB for similar past incidents."""
    collection = "alerts"
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            r = await client.post(
                f"{CHROMA_URL}/api/v1/collections/{collection}/query",
                json={"query_embeddings": [vector], "n_results": top_k},
            )
            r.raise_for_status()
            data = r.json()
            # Chroma returns { ids, distances, metadatas }
            results = []
            if "metadatas" in data and data["metadatas"]:
                for meta_list in data["metadatas"]:
                    results.extend(meta_list)
            return results
        except httpx.HTTPStatusError:
            # collection might be empty or not exist yet
            return []


async def prompt_llm(alert_summary: str, similar: list[dict]) -> str:
    """Ask Mistral for root cause hypothesis."""
    context = "\n".join(
        [f"- {s.get('alertname')} in {s.get('namespace')}: {s.get('labels','')}"
         for s in similar[:3]]
    )
    prompt = f"""Tu es un expert SRE Kubernetes. Voici une alerte actuelle :

{alert_summary}

Incidents similaires passés :
{context if context else "(aucun historique)"}

Quelle est la cause probable ? Réponds en 2-3 phrases concises."""

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": "mistral", "prompt": prompt, "stream": False},
        )
        r.raise_for_status()
        return r.json()["response"]


async def message_handler(msg):
    """Process incoming alert message from NATS."""
    data = json.loads(msg.data.decode())
    summary = data.get("summary", "")
    print(f"[RCA] Received: {summary}")

    # 1) Generate embedding for similarity search
    vector = await generate_embedding(summary)

    # 2) Query similar incidents
    similar = await query_similar(vector, top_k=5)
    print(f"[RCA] Found {len(similar)} similar incidents")

    # 3) Prompt LLM
    diagnosis = await prompt_llm(summary, similar)
    print(f"[RCA] Diagnosis: {diagnosis}")

    # 4) Publish result
    nc = msg._client
    rca_msg = {
        "alert_id": data.get("id"),
        "summary": summary,
        "diagnosis": diagnosis,
        "similar_count": len(similar),
    }
    await nc.publish(NATS_SUBJECT_OUT, json.dumps(rca_msg).encode())
    print(f"[RCA] Published to {NATS_SUBJECT_OUT}")


async def run_rca_agent():
    """Main loop: subscribe to NATS and process alerts."""
    nc = NATS()
    await nc.connect(servers=[NATS_URL])
    print(f"[RCA] Connected to NATS, subscribing to {NATS_SUBJECT_IN}")

    await nc.subscribe(NATS_SUBJECT_IN, cb=message_handler)

    # Keep running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        await nc.close()


if __name__ == "__main__":
    asyncio.run(run_rca_agent())
