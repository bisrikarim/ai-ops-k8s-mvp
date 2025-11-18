# AI-Ops MVP : Système de diagnostic et remédiation automatique pour Kubernetes

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/downloads/)
[![Kubernetes](https://img.shields.io/badge/kubernetes-%23326ce5.svg?style=for-the-badge&logo=kubernetes&logoColor=white)](https://kubernetes.io/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![LLM](https://img.shields.io/badge/Mistral%20AI-FA520F?logo=mistral-ai&logoColor=fff)](https://ollama.ai/)

## Vue d'ensemble

Ce projet implémente une chaîne complète d'observabilité intelligente pour Kubernetes, utilisant des modèles de langage (LLM) pour analyser automatiquement les incidents et proposer des correctifs.

### Qu'est-ce que ce système fait concrètement ?

Lorsqu'un problème survient dans votre cluster Kubernetes (un pod qui crashloop, une application qui consomme trop de mémoire, etc.), le système :

1. **Détecte** l'anomalie via Prometheus
2. **Collecte** les informations contextuelles (logs, métriques, événements)
3. **Analyse** la cause probable en interrogeant un modèle d'IA local (Mistral-7B)
4. **Propose** un correctif Kubernetes prêt à appliquer
5. **Apprend** de chaque incident pour améliorer ses futurs diagnostics

Le tout fonctionne en local, sans dépendance à des services cloud payants.

---

## Architecture du système

Le MVP est composé de 6 briques principales :

### Couche Observabilité
- **Prometheus** : collecte des métriques du cluster
- **Alertmanager** : gère les alertes et envoie des webhooks
- **Loki** : agrège les logs des pods
- **Grafana** : visualisation (optionnel pour le MVP)

### Couche Données & Traitement
- **NATS JetStream** : bus de messages temps réel entre les agents
- **ChromaDB** : base de données vectorielle pour stocker les empreintes d'incidents
- **Ollama** : serveur d'inférence pour les modèles LLM locaux (Mistral-7B + nomic-embed-text)

### Couche Intelligence (3 agents Python)
1. **Collector** : reçoit les webhooks Alertmanager, génère des embeddings, stocke dans Chroma, publie sur NATS
2. **RCA (Root Cause Analysis)** : consomme les alertes, recherche des incidents similaires, interroge le LLM, publie le diagnostic
3. **Remediator** : consomme les diagnostics, génère des patches Kubernetes, prépare des commits Git

### Flux de données

```
Alert → Prometheus → Alertmanager → Collector → NATS
                                         ↓
                                     ChromaDB
                                         
NATS → RCA → LLM (Mistral) → Diagnostic → NATS

NATS → Remediator → Patch K8s → Proposition Git
```

---

## Prérequis

### Matériel recommandé
- **CPU** : 4 vCPU minimum (8 recommandé)
- **RAM** : 8 Go minimum (16 Go recommandé)
- **Disque** : 50 Go d'espace libre
- **OS** : Ubuntu 24.04 (natif ou WSL 2)

### Logiciels requis
- Ubuntu 24.04 LTS
- Python 3.12+
- Git
- Accès sudo

---

## Installation complète

### Étape 1 : Préparation de l'environnement

#### 1.1 Mise à jour du système

```bash
sudo apt update && sudo apt upgrade -y
```

#### 1.2 Installation des dépendances de base

```bash
sudo apt install -y curl wget git build-essential python3-pip python3-venv
```

#### 1.3 Configuration WSL (si applicable)

Si vous utilisez WSL 2, configurez les ressources allouées en créant le fichier `C:\Users\<VotreUser>\.wslconfig` sous Windows :

```ini
[wsl2]
processors=8
memory=16GB
swap=4GB
localhostForwarding=true
```

Redémarrez WSL après modification :

```powershell
wsl --shutdown
wsl
```

Activez systemd dans `/etc/wsl.conf` (dans Ubuntu WSL) :

```bash
sudo nano /etc/wsl.conf
```

Ajoutez :

```ini
[boot]
systemd=true
```

Redémarrez à nouveau WSL.

---

### Étape 2 : Installation de Kubernetes (k3s)

k3s est une distribution Kubernetes légère, idéale pour un environnement de développement ou un MVP.

```bash
curl -sfL https://get.k3s.io | sh -s - \
  --write-kubeconfig-mode 644 \
  --disable traefik \
  --disable local-storage
```

Cette commande :
- Installe k3s en tant que service systemd
- Configure kubectl automatiquement
- Désactive Traefik et le provisionneur de stockage par défaut (nous utiliserons local-path)

Vérifiez que le cluster est opérationnel :

```bash
sudo k3s kubectl get nodes
```

Vous devriez voir votre nœud en état `Ready`.

Configurez kubectl pour l'utilisateur courant :

```bash
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown $(id -u):$(id -g) ~/.kube/config
```

Vérifiez :

```bash
kubectl get nodes
```

---

### Étape 3 : Installation de Helm

Helm est un gestionnaire de paquets pour Kubernetes qui simplifie l'installation d'applications complexes.

```bash
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```

Vérifiez l'installation :

```bash
helm version
```

---

### Étape 4 : Installation du provisionneur de stockage

k3s nécessite un provisionneur de volumes pour les applications stateful comme ChromaDB.

```bash
kubectl apply -f https://raw.githubusercontent.com/rancher/local-path-provisioner/master/deploy/local-path-storage.yaml
```

Vérifiez que le StorageClass est créé :

```bash
kubectl get storageclass
```

Vous devriez voir `local-path (default)`.

---

### Étape 5 : Déploiement de la stack Observabilité

#### 5.1 Création du namespace

```bash
kubectl create namespace observability
```

#### 5.2 Ajout des dépôts Helm

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update
```

#### 5.3 Installation de Prometheus + Alertmanager + Grafana

```bash
helm install kps prometheus-community/kube-prometheus-stack \
  -n observability \
  --set prometheus-node-exporter.enabled=false
```

Note : Nous désactivons node-exporter car il n'est pas compatible avec WSL 2 (problème de montage des volumes host).

Attendez que tous les pods démarrent (cela peut prendre 2-3 minutes) :

```bash
kubectl get pods -n observability -w
```

Appuyez sur Ctrl+C une fois que tous les pods sont `Running`.

#### 5.4 Installation de Loki + Promtail

```bash
helm install loki grafana/loki-stack \
  --set promtail.enabled=true \
  -n observability
```

Vérifiez :

```bash
kubectl get pods -n observability | grep loki
```

---

### Étape 6 : Déploiement de l'infrastructure AI-Ops

#### 6.1 Création du namespace

```bash
kubectl create namespace ai-ops
```

#### 6.2 Installation de NATS JetStream

```bash
helm repo add nats https://nats-io.github.io/k8s/helm/charts
helm repo update

helm install nats nats/nats -n ai-ops \
  --set nats.jetstream.enabled=true \
  --set nats.jetstream.memStorage.enabled=true \
  --set nats.jetstream.fileStorage.enabled=false \
  --set cluster.enabled=false \
  --set auth.enabled=false
```

Cette configuration :
- Active JetStream (système de messaging avec persistance)
- Utilise le stockage en mémoire (suffisant pour un MVP)
- Mode single-node (pas de cluster NATS)
- Désactive l'authentification (sécurisé en prod avec TLS + users)

Vérifiez :

```bash
kubectl get pods -n ai-ops
```

Vous devriez voir `nats-0` et `nats-box-xxx` en état `Running`.

#### 6.3 Déploiement de ChromaDB

Créez le fichier `chromadb.yaml` :

```bash
cat > chromadb.yaml <<'EOF'
apiVersion: v1
kind: Service
metadata:
  name: chromadb
  namespace: ai-ops
  labels:
    app: chromadb
spec:
  ports:
    - port: 8000
      targetPort: 8000
  selector:
    app: chromadb
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: chromadb
  namespace: ai-ops
spec:
  serviceName: chromadb
  replicas: 1
  selector:
    matchLabels:
      app: chromadb
  template:
    metadata:
      labels:
        app: chromadb
    spec:
      containers:
        - name: chromadb
          image: ghcr.io/chroma-core/chroma:0.4.24
          ports:
            - containerPort: 8000
          env:
            - name: IS_PERSISTENT
              value: "true"
          volumeMounts:
            - name: data
              mountPath: /chroma/chroma
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 5Gi
EOF
```

Appliquez :

```bash
kubectl apply -f chromadb.yaml
```

Vérifiez :

```bash
kubectl get pods -n ai-ops -l app=chromadb
kubectl get pvc -n ai-ops
```

Le pod doit être `Running` et le PVC `Bound`.

---

### Étape 7 : Installation d'Ollama et des modèles LLM

Ollama est un serveur local qui permet d'exécuter des modèles LLM (comme Mistral) sans API externe.

#### 7.1 Installation d'Ollama

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

Cette commande :
- Installe le binaire `ollama`
- Configure le service systemd
- Démarre le serveur sur `http://localhost:11434`

Vérifiez que le service est actif :

```bash
systemctl status ollama
```

#### 7.2 Téléchargement des modèles

**Mistral-7B** (modèle de raisonnement pour l'analyse) :

```bash
ollama pull mistral
```

Ce téléchargement fait environ 4 Go (version quantisée). Patientez quelques minutes.

**nomic-embed-text** (modèle d'embeddings pour la recherche vectorielle) :

```bash
ollama pull nomic-embed-text
```

Ce modèle fait environ 270 Mo.

Vérifiez que les modèles sont disponibles :

```bash
ollama list
```

Vous devriez voir :

```
NAME                    ID              SIZE
mistral:latest          ...             4.1 GB
nomic-embed-text:latest ...             274 MB
```

Testez l'API :

```bash
curl http://localhost:11434/api/generate \
  -d '{"model": "mistral", "prompt": "Hello", "stream": false}'
```

Vous devriez recevoir une réponse JSON.

---

### Étape 8 : Développement et installation des agents Python

#### 8.1 Installation de Poetry

Poetry est un gestionnaire de dépendances Python moderne qui gère les environnements virtuels.

```bash
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
```

Vérifiez :

```bash
poetry --version
```

#### 8.2 Création du projet

```bash
mkdir -p ~/ai-ops-agents && cd ~/ai-ops-agents
git init
poetry new --name ai_ops_agents app
cd app
```

#### 8.3 Configuration de Poetry

Modifiez `pyproject.toml` pour corriger la contrainte Python :

```bash
nano pyproject.toml
```

Remplacez la section `[tool.poetry]` par :

```toml
[tool.poetry]
name = "ai-ops-agents"
version = "0.1.0"
description = "AI-Ops agents for Kubernetes incident management"
authors = ["Votre Nom <votre.email@example.com>"]
packages = [{ include = "ai_ops_agents", from = "src" }]

[tool.poetry.dependencies]
python = ">=3.12,<4.0"
langchain = "0.2.0"
chromadb = "^1.3.4"
nats-py = "^2.12.0"
kubernetes = "^34.1.0"
pydantic = "^2.12.4"
gitpython = "^3.1.45"
python-dotenv = "^1.2.1"
fastapi = "^0.121.2"
uvicorn = {extras = ["standard"], version = "^0.38.0"}
typer = "^0.20.0"
httpx = "^0.28.0"

[build-system]
requires = ["poetry-core>=2.0.0,<3.0.0"]
build-backend = "poetry.core.masonry.api"
```

Installez les dépendances :

```bash
poetry install
```

#### 8.4 Création de la structure du projet

```bash
mkdir -p src/ai_ops_agents
touch src/ai_ops_agents/__init__.py
```

#### 8.5 Implémentation de l'agent Collector

Créez `src/ai_ops_agents/collector.py` :

```bash
cat > src/ai_ops_agents/collector.py <<'EOF'
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
NATS_URL = os.getenv("NATS_URL", "nats://nats.ai-ops:4222")
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
    vector = await generate_embedding(summary)
    await upsert_chroma(vector, meta)
    nc = await NATSClient.get()
    await nc.publish(NATS_SUBJECT, json.dumps({"summary": summary, **meta}).encode())
    return {"status": "queued"}
EOF
```

#### 8.6 Implémentation de l'agent RCA

Créez `src/ai_ops_agents/rca.py` :

```bash
cat > src/ai_ops_agents/rca.py <<'EOF'
"""RCA service – consumes alerts from NATS, queries similar incidents, prompts LLM, publishes diagnosis."""

import asyncio
import json
import os

import httpx
from nats.aio.client import Client as NATS

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
CHROMA_URL = os.getenv("CHROMA_URL", "http://localhost:8000")
NATS_URL = os.getenv("NATS_URL", "nats://localhost:4222")
NATS_SUBJECT_IN = os.getenv("NATS_SUBJECT_IN", "observability.raw")
NATS_SUBJECT_OUT = os.getenv("NATS_SUBJECT_OUT", "ai.rca")


async def generate_embedding(text: str) -> list[float]:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text},
        )
        r.raise_for_status()
        return r.json()["embedding"]


async def query_similar(vector: list[float], top_k: int = 5) -> list[dict]:
    collection = "alerts"
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            r = await client.post(
                f"{CHROMA_URL}/api/v1/collections/{collection}/query",
                json={"query_embeddings": [vector], "n_results": top_k},
            )
            r.raise_for_status()
            data = r.json()
            results = []
            if "metadatas" in data and data["metadatas"]:
                for meta_list in data["metadatas"]:
                    results.extend(meta_list)
            return results
        except httpx.HTTPStatusError:
            return []


async def prompt_llm(alert_summary: str, similar: list[dict]) -> str:
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
    data = json.loads(msg.data.decode())
    summary = data.get("summary", "")
    print(f"[RCA] Received: {summary}")

    vector = await generate_embedding(summary)
    similar = await query_similar(vector, top_k=5)
    print(f"[RCA] Found {len(similar)} similar incidents")

    diagnosis = await prompt_llm(summary, similar)
    print(f"[RCA] Diagnosis: {diagnosis}")

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
    nc = NATS()
    await nc.connect(servers=[NATS_URL])
    print(f"[RCA] Connected to NATS, subscribing to {NATS_SUBJECT_IN}")

    await nc.subscribe(NATS_SUBJECT_IN, cb=message_handler)

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        await nc.close()


if __name__ == "__main__":
    asyncio.run(run_rca_agent())
EOF
```

#### 8.7 Implémentation de l'agent Remediator

Créez `src/ai_ops_agents/remediator.py` :

```bash
cat > src/ai_ops_agents/remediator.py <<'EOF'
"""Remediator service – consumes RCA results, generates K8s patches, proposes GitOps commits."""

import asyncio
import json
import os
from datetime import datetime

from nats.aio.client import Client as NATS

NATS_URL = os.getenv("NATS_URL", "nats://localhost:4222")
NATS_SUBJECT_IN = os.getenv("NATS_SUBJECT_IN", "ai.rca")


def generate_patch(rca_data: dict) -> str:
    alert_id = rca_data.get("alert_id", "unknown")
    diagnosis = rca_data.get("diagnosis", "")
    
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
  name: <deployment-name>
  namespace: default
spec:
  template:
    metadata:
      annotations:
        kubectl.kubernetes.io/restartedAt: "{timestamp}"
    spec:
      containers:
      - name: app
        image: <previous-image>
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
            memory: "512Mi"
          requests:
            memory: "256Mi"
"""
    
    else:
        patch = f"""# Manual intervention required for alert {alert_id}
# Diagnosis: {diagnosis}
"""
    
    return patch


async def message_handler(msg):
    data = json.loads(msg.data.decode())
    alert_id = data.get("alert_id", "unknown")
    summary = data.get("summary", "")
    diagnosis = data.get("diagnosis", "")
    
    print(f"\n[REMEDIATION] Received RCA for alert {alert_id}")
    print(f"[REMEDIATION] Summary: {summary}")
    print(f"[REMEDIATION] Diagnosis: {diagnosis[:150]}...")
    
    patch = generate_patch(data)
    
    print(f"\n[REMEDIATION] Generated patch:\n{patch}")
    
    commit_msg = f"fix: auto-remediation for {alert_id}\n\n{diagnosis[:200]}"
    
    print(f"\n[REMEDIATION] Would commit with message:")
    print(f"---\n{commit_msg}\n---")
    print(f"[REMEDIATION] Remediation proposal ready for review\n")


async def run_remediator_agent():
    nc = NATS()
    await nc.connect(servers=[NATS_URL])
    print(f"[REMEDIATION] Connected to NATS, subscribing to {NATS_SUBJECT_IN}")
    
    await nc.subscribe(NATS_SUBJECT_IN, cb=message_handler)
    
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        await nc.close()


if __name__ == "__main__":
    asyncio.run(run_remediator_agent())
EOF
```

#### 8.8 Création du CLI

Créez `src/ai_ops_agents/cli.py` :

```bash
cat > src/ai_ops_agents/cli.py <<'EOF'
import typer, uvicorn, asyncio
from .collector import app as collector_app
from .rca import run_rca_agent
from .remediator import run_remediator_agent

cli = typer.Typer()

@cli.command()
def collector(port: int = 8080):
    """Lance l'agent collector (webhook Alertmanager)."""
    uvicorn.run(
        collector_app,
        host="0.0.0.0",
        port=port,
        log_level="info",
    )

@cli.command()
def rca():
    """Lance l'agent RCA (consomme NATS, appelle LLM, publie diagnostic)."""
    asyncio.run(run_rca_agent())

@cli.command()
def remediator():
    """Lance l'agent remediator (consomme RCA, génère patches K8s)."""
    asyncio.run(run_remediator_agent())

if __name__ == "__main__":
    cli()
EOF
```

---

### Étape 9 : Configuration d'Alertmanager

#### 9.1 Récupération de l'IP WSL

Votre Collector tournera en local et doit être accessible depuis le cluster. Récupérez votre IP WSL :

```bash
ip addr show eth0 | grep 'inet ' | awk '{print $2}' | cut -d/ -f1
```

Notez cette IP (par exemple : `172.25.232.235`).

#### 9.2 Configuration du webhook

Récupérez la configuration actuelle :

```bash
kubectl -n observability get secret alertmanager-kps-kube-prometheus-stack-alertmanager \
  -o jsonpath='{.data.alertmanager\.yaml}' | base64 -d > /tmp/alertmanager.yaml
```

Éditez le fichier :

```bash
nano /tmp/alertmanager.yaml
```

Remplacez le contenu par (en adaptant l'IP) :

```yaml
global:
  resolve_timeout: 5m
inhibit_rules:
- equal:
  - namespace
  - alertname
  source_matchers:
  - severity = critical
  target_matchers:
  - severity =~ warning|info
receivers:
- name: "null"
- name: "ai-ops-webhook"
  webhook_configs:
  - url: 'http://172.25.232.235:8081/alert'
    send_resolved: false
route:
  group_by:
  - namespace
  - alertname
  group_interval: 5m
  group_wait: 30s
  receiver: "ai-ops-webhook"
  repeat_interval: 5m
  routes:
  - matchers:
    - alertname = "Watchdog"
    receiver: "null"
templates:
- /etc/alertmanager/config/*.tmpl
```

Appliquez :

```bash
kubectl -n observability create secret generic alertmanager-kps-kube-prometheus-stack-alertmanager \
  --from-file=alertmanager.yaml=/tmp/alertmanager.yaml \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl -n observability delete pod -l app.kubernetes.io/name=alertmanager
```

Attendez que le pod redémarre :

```bash
kubectl -n observability get pods -l app.kubernetes.io/name=alertmanager
```

#### 9.3 Création d'une règle d'alerte de test

Créez une règle qui détectera les CrashLoopBackOff :

```bash
cat <<EOF | kubectl apply -f -
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: pod-crashloop-alerts
  namespace: observability
  labels:
    prometheus: kube-prometheus-stack
    release: kps
spec:
  groups:
  - name: pod-alerts
    interval: 30s
    rules:
    - alert: PodCrashLooping
      expr: rate(kube_pod_container_status_restarts_total[5m]) > 0
      for: 1m
      labels:
        severity: critical
      annotations:
        description: "Pod {{ \$labels.namespace }}/{{ \$labels.pod }} (container {{ \$labels.container }}) is restarting repeatedly"
        summary: "CrashLoopBackOff detected in {{ \$labels.namespace }}"
EOF
```

---

### Étape 10 : Création d'un pod de test

Déployez un pod qui crashloop volontairement :

```bash
kubectl create namespace test-aiops

cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: crashloop-demo
  namespace: test-aiops
  labels:
    app: crashloop-demo
spec:
  replicas: 1
  selector:
    matchLabels:
      app: crashloop-demo
  template:
    metadata:
      labels:
        app: crashloop-demo
    spec:
      containers:
      - name: app
        image: busybox
        command: ["sh", "-c", "echo 'Starting app...'; sleep 5; exit 1"]
        resources:
          limits:
            memory: "64Mi"
          requests:
            memory: "32Mi"
EOF
```

Vérifiez qu'il crashloop :

```bash
kubectl get pods -n test-aiops
```

Vous devriez voir le pod en état `CrashLoopBackOff` ou `Error` avec des restarts croissants.

---

## Utilisation du système

### Lancement des agents

Vous avez besoin de 5 terminaux en parallèle :

#### Terminal 1 : Port-forward ChromaDB

```bash
kubectl -n ai-ops port-forward svc/chromadb 8000:8000
```

#### Terminal 2 : Port-forward NATS

```bash
kubectl -n ai-ops port-forward svc/nats 4222:4222
```

#### Terminal 3 : Agent Collector

```bash
cd ~/ai-ops-agents/app
source ~/.cache/pypoetry/virtualenvs/ai-ops-agents-*/bin/activate
CHROMA_URL=http://localhost:8000 NATS_URL=nats://localhost:4222 \
  python -m ai_ops_agents.cli collector --port 8081
```

#### Terminal 4 : Agent RCA

```bash
cd ~/ai-ops-agents/app
source ~/.cache/pypoetry/virtualenvs/ai-ops-agents-*/bin/activate
CHROMA_URL=http://localhost:8000 NATS_URL=nats://localhost:4222 \
  python -m ai_ops_agents.cli rca
```

#### Terminal 5 : Agent Remediator

```bash
cd ~/ai-ops-agents/app
source ~/.cache/pypoetry/virtualenvs/ai-ops-agents-*/bin/activate
NATS_URL=nats://localhost:4222 \
  python -m ai_ops_agents.cli remediator
```

---

### Observation du système en action

Une fois les 5 terminaux lancés, le système est opérationnel. Le pod `crashloop-demo` continue de crasher, et après environ 1-2 minutes :

1. **Prometheus** détecte l'anomalie
2. **Alertmanager** envoie un webhook vers le Collector
3. Le **Collector** affiche :
   ```
   INFO: 172.25.232.235:xxxxx - "POST /alert HTTP/1.1" 202 Accepted
   ```

4. L'agent **RCA** affiche :
   ```
   [RCA] Received: PodCrashLooping in ns test-aiops...
   [RCA] Found 0 similar incidents
   [RCA] Diagnosis: <analyse de Mistral>
   [RCA] Published to ai.rca
   ```

5. L'agent **Remediator** affiche :
   ```
   [REMEDIATION] Received RCA for alert ...
   [REMEDIATION] Generated patch:
   <patch Kubernetes>
   [REMEDIATION] Remediation proposal ready for review
   ```

---

## Validation du système

### Test manuel via curl

Vous pouvez simuler une alerte sans attendre Prometheus :

```bash
curl -X POST http://localhost:8081/alert \
  -H 'Content-Type: application/json' \
  -d '{
    "status": "firing",
    "alerts": [{
      "labels": {
        "alertname": "TestAlert",
        "namespace": "default",
        "pod": "test-pod"
      },
      "annotations": {
        "description": "Ceci est un test"
      }
    }]
  }'
```

Vous devriez recevoir :

```json
{"status":"queued"}
```

Et observer l'activité dans les terminaux RCA et Remediator.

### Accès aux interfaces web

#### Prometheus

```bash
kubectl port-forward -n observability svc/kps-kube-prometheus-stack-prometheus 9090:9090
```

Puis ouvrez http://localhost:9090 dans votre navigateur.

#### Alertmanager

```bash
kubectl port-forward -n observability svc/kps-kube-prometheus-stack-alertmanager 9093:9093
```

Puis ouvrez http://localhost:9093

#### Grafana

```bash
kubectl port-forward -n observability svc/kps-grafana 3000:80
```

Récupérez le mot de passe admin :

```bash
kubectl -n observability get secret kps-grafana -o jsonpath="{.data.admin-password}" | base64 -d; echo
```

Ouvrez http://localhost:3000 (utilisateur : `admin`).

---

## Dépannage

### Le Collector ne reçoit pas d'alertes

**Symptômes** : Aucune ligne dans le terminal Collector après 2-3 minutes.

**Causes possibles** :
1. Alertmanager n'a pas redémarré après la config
2. L'IP WSL a changé
3. Le firewall bloque les connexions

**Solutions** :

Vérifiez l'état d'Alertmanager :

```bash
kubectl -n observability logs -l app.kubernetes.io/name=alertmanager --tail=50
```

Cherchez des erreurs de connexion vers votre IP.

Vérifiez que votre IP est toujours la même :

```bash
ip addr show eth0 | grep 'inet '
```

Si elle a changé, remettez à jour la config Alertmanager (Étape 9.2).

### Ollama retourne des erreurs

**Symptômes** : Le RCA affiche des exceptions `httpcore.ConnectError`.

**Causes** : Ollama n'est pas démarré ou le port 11434 est bloqué.

**Solutions** :

Vérifiez le service :

```bash
systemctl status ollama
```

Si inactif :

```bash
sudo systemctl start ollama
```

Testez l'API :

```bash
curl http://localhost:11434/api/generate -d '{"model":"mistral","prompt":"test","stream":false}'
```

### ChromaDB ou NATS inaccessibles

**Symptômes** : Les agents affichent `Connection refused`.

**Causes** : Les port-forwards ne sont pas actifs.

**Solutions** :

Relancez les port-forwards dans deux terminaux dédiés :

```bash
# Terminal 1
kubectl -n ai-ops port-forward svc/chromadb 8000:8000

# Terminal 2
kubectl -n ai-ops port-forward svc/nats 4222:4222
```

### Les pods Prometheus/Grafana restent en Pending

**Cause** : Ressources insuffisantes ou StorageClass manquant.

**Solutions** :

Vérifiez les ressources WSL (`.wslconfig`) et redémarrez WSL si nécessaire.

Vérifiez que le provisionneur de stockage est actif :

```bash
kubectl get storageclass
kubectl -n kube-system get pods | grep local-path
```

---

## Architecture de production

Ce MVP est conçu pour un environnement de développement/test. Pour une utilisation en production, considérez :

### Sécurité
- Activation de l'authentification NATS (TLS + users/tokens)
- Mise en place de NetworkPolicies Kubernetes
- Rotation des credentials ChromaDB
- Restriction RBAC pour les agents

### Haute disponibilité
- Déploiement de NATS en mode cluster (3 nœuds)
- ChromaDB avec réplication (ou migration vers Weaviate/Milvus)
- Plusieurs réplicas des agents avec un LoadBalancer

### Scalabilité
- Utilisation d'un modèle LLM plus performant (Mixtral 8x7B) sur GPU
- Fine-tuning de Mistral sur vos incidents historiques
- Intégration d'un cache Redis pour les embeddings

### Observabilité des agents
- Métriques Prometheus pour chaque agent (latence, erreurs, throughput)
- Dashboards Grafana dédiés
- Alertes si les agents deviennent injoignables

### GitOps réel
- Installation de FluxCD ou ArgoCD
- Configuration d'un dépôt Git pour les manifests
- Automatisation des commits/PR depuis le Remediator

---

## Prochaines étapes recommandées

### Court terme (1-2 semaines)
1. **Tester sur des incidents réels** : laissez tourner le système et collectez des feedbacks
2. **Améliorer les prompts** : ajustez les questions envoyées à Mistral pour affiner les diagnostics
3. **Créer des dashboards Grafana** : visualisez les incidents traités, le temps de résolution, etc.

### Moyen terme (1 mois)
1. **Packager les agents en images Docker** : créez des Dockerfiles et poussez les images dans un registry
2. **Déployer les agents dans le cluster** : éliminez les port-forwards et lancez tout dans Kubernetes
3. **Intégrer un bot Slack/Teams** : notifications temps réel des diagnostics

### Long terme (3-6 mois)
1. **Collecter des données pour le fine-tuning** : rassemblez 100-200 incidents avec leurs résolutions
2. **Fine-tuner Mistral** : entraînez le modèle sur vos patterns spécifiques
3. **Automatiser l'application des correctifs** : après validation, laissez le Remediator appliquer directement certains patchs non-critiques

---

## Ressources supplémentaires

### Documentation officielle
- [Kubernetes](https://kubernetes.io/docs/)
- [Prometheus Operator](https://prometheus-operator.dev/)
- [NATS JetStream](https://docs.nats.io/nats-concepts/jetstream)
- [ChromaDB](https://docs.trychroma.com/)
- [Ollama](https://ollama.ai/)
- [LangChain](https://python.langchain.com/)

### Communauté
- Groupe Kubernetes Slack : #sig-observability
- Discord Ollama : https://discord.gg/ollama
- Forum ChromaDB : https://github.com/chroma-core/chroma/discussions

---

## Contributions

Ce projet est un MVP de démonstration. Les contributions sont les bienvenues :

- Amélioration des prompts LLM
- Ajout de nouveaux types de remediations
- Support de modèles LLM alternatifs (Llama 3, Gemma, etc.)
- Intégration d'autres sources d'observabilité (OpenTelemetry, Jaeger)

---

## Licence

Ce projet est fourni à des fins éducatives et de démonstration. Adaptez-le à vos besoins.

---

**Auteur** : MVP développé en collaboration avec Claude (Anthropic)  
**Date** : Novembre 2025  
**Version** : 1.0.0

