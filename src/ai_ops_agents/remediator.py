"""Remediator service – consumes RCA results, generates K8s patches, proposes GitOps commits."""

import asyncio
import json
import os
from datetime import datetime

from nats.aio.client import Client as NATS


# Config
NATS_URL = os.getenv("NATS_URL", "nats://localhost:4222")
NATS_SUBJECT_IN = os.getenv("NATS_SUBJECT_IN", "ai.rca")
GIT_REPO_PATH = os.getenv("GIT_REPO_PATH", "/tmp/k8s-manifests")


def generate_patch(rca_data: dict) -> str:
    """Generate a Kustomize patch or manifest based on RCA diagnosis."""
    
    alert_id = rca_data.get("alert_id", "unknown")
    summary = rca_data.get("summary", "")
    diagnosis = rca_data.get("diagnosis", "")
    
    # Simple heuristic: if diagnosis mentions "crash" or "loop", suggest rollback
    action = "rollback"
    if "mémoire" in diagnosis.lower() or "memory" in diagnosis.lower():
        action = "increase_memory"
    elif "cpu" in diagnosis.lower():
        action = "increase_cpu"
    
    timestamp = datetime.utcnow().isoformat()
    
    if action == "rollback":
        patch = f"""# Auto-generated remediation for alert {alert_id}
# Timestamp: {timestamp}
# Diagnosis: {diagnosis[:100]}...

apiVersion: apps/v1
kind: Deployment
metadata:
  name: <deployment-name>  # À compléter selon le pod
  namespace: default
spec:
  template:
    metadata:
      annotations:
        kubectl.kubernetes.io/restartedAt: "{timestamp}"
    spec:
      containers:
      - name: app
        image: <previous-image>  # Rollback vers image stable précédente
"""
    
    elif action == "increase_memory":
        patch = f"""# Auto-generated remediation for alert {alert_id}
# Timestamp: {timestamp}

apiVersion: apps/v1
kind: Deployment
metadata:
  name: <deployment-name>
  namespace: default
spec:
  template:
    spec:
      containers:
      - name: app
        resources:
          limits:
            memory: "512Mi"  # Augmenté depuis 256Mi
          requests:
            memory: "256Mi"
"""
    
    else:
        patch = f"""# Manual intervention required for alert {alert_id}
# Diagnosis: {diagnosis}
"""
    
    return patch


async def message_handler(msg):
    """Process incoming RCA messages and generate remediation."""
    
    data = json.loads(msg.data.decode())
    alert_id = data.get("alert_id", "unknown")
    summary = data.get("summary", "")
    diagnosis = data.get("diagnosis", "")
    
    print(f"\n[REMEDIATION] Received RCA for alert {alert_id}")
    print(f"[REMEDIATION] Summary: {summary}")
    print(f"[REMEDIATION] Diagnosis: {diagnosis[:150]}...")
    
    # Generate patch
    patch = generate_patch(data)
    
    print(f"\n[REMEDIATION] Generated patch:\n{patch}")
    
    # Simulate Git commit (for MVP, just log it)
    commit_msg = f"fix: auto-remediation for {alert_id}\n\n{diagnosis[:200]}"
    
    print(f"\n[REMEDIATION] Would commit with message:")
    print(f"---\n{commit_msg}\n---")
    
    # In production: git add, commit, push, open PR
    # For MVP: just display the action
    print(f"[REMEDIATION] ✅ Remediation proposal ready for review\n")


async def run_remediator_agent():
    """Main loop: subscribe to ai.rca and generate remediations."""
    
    nc = NATS()
    await nc.connect(servers=[NATS_URL])
    print(f"[REMEDIATION] Connected to NATS, subscribing to {NATS_SUBJECT_IN}")
    
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
    asyncio.run(run_remediator_agent())
