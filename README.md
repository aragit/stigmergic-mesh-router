# Stigmergic Mesh Router

> **An Enterprise-Grade, Multi-Agent Swarm Architecture for Self-Organizing LLM Inference Routing**

The `stigmergic-mesh-router` implements **Digital Stigmergy** — a bio-inspired mechanism of indirect coordination where autonomous router and worker agents communicate by depositing $4\\text{D}$ pheromone signals into a shared environmental memory field.

Rather than relying on rigid, top‑down load-balancing rules, the mesh achieves **emergent, dynamic traffic optimization** across Velocity ($V$), Latency ($L$), Stability ($S$), and Cost ($C$), incorporating real-time feedback decay, multi-tenant ingress governance, persistent state checkpointing, and GitOps orchestration.

---

## 🐝 Swarm Architecture vs. Traditional Routing

| Dimension | Traditional LLM Load Balancer | Stigmergic Mesh Router (Swarm) |
| :--- | :--- | :--- |
| **Control Paradigm** | Centralized, top-down rule engine / Round-Robin | Decentralized, emergent multi-agent self-organization |
| **Agent Coordination** | Direct RPC / synchronous health checks | Indirect stigmergic signaling via shared $4\\text{D}$ Memory Field |
| **Adaptability** | Hardcoded failovers & static weight matrices | Real-time pheromone reinforcement & continuous decay ($\\gamma$) |
| **State Dynamics** | Stateless or rigid session persistence | Dynamic environmental memory with warm-start boot hydration |

---

## 📐 Conceptual Swarm Stigmergy Loop

```text
               ┌────────────────────────────────────────────────────────┐
               │        ENVIRONMENTAL SUBSTRATE (MEMORY FIELD)          │
               │   Shared 4D Pheromone Matrix: M_i = (V_i, L_i, S_i, C_i)  │
               └───────────────▲────────────────────────┬───────────────┘
                               │                        │
                    Pheromone Reinforcement     Environmental Sensing
                   (Post-Execution Telemetry)    (Softmax Path Sampling)
                               │                        │
              ┌────────────────┴────────────────────────▼────────────────┐
              │                   SWARM AGENT MESH                       │
              │  • Router Agents: Sample candidate paths via Softmax     │
              │  • Worker Agents: Process requests & stream responses   │
              │  • Evaporation Worker: Continuous decay ($\\gamma$) of state │
              └──────────────────────────────────────────────────────────┘
```

## 🏗️ Complete End-to-End Architecture

```
                               ┌─────────────────────────────────────────┐
                               │         EXTERNAL CLIENT REQUEST         │
                               └────────────────────┬────────────────────┘
                                                    │
                                                    v
                               ┌─────────────────────────────────────────┐
                               │   HTTP Header: Authorization / Bearer   │
                               └────────────────────┬────────────────────┘
                                                    │
                                                    v
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ INGRESS GOVERNANCE & MULTI-TENANCY (Phase 11)                                          │
│                                                                                        │
│  1. AUTHENTICATION (`api/auth.py`)                                                     │
│     └── SHA-256 API Key verification (Redis lookup with static-map fallback)          │
│                                                                                        │
│  2. DISTRIBUTED RATE LIMITING (`api/rate_limiter.py`)                                  │
│     └── Atomic Redis Lua Token Bucket (RPM & TPM enforcement)                          │
│                                                                                        │
│  3. TENANT GOVERNANCE & CONTEXT (`api/governance.py`)                                  │
│     └── Inject `TenantContext` & apply tier matrix weight adjustments ($\\Delta w$)      │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ (Authenticated & Quota Approved)
                                            v
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ STIGMERGIC MULTI-AGENT SWARM MESH (Phases 1–10)                                        │
│                                                                                        │
│  1. ROUTER AGENTS (`core/router_agent.py`)                                             │
│     ├── Inspect 4D Pheromone Field: S_i = w_V*V_i - w_L*L_i + w_S*S_i - w_C*C_i         │
│     └── Sample Path via Boltzmann/Softmax: P(i) = exp(S_i / \\tau) / \\sum exp(S_j / \\tau) │
│                                                                                        │
│  2. WORKER AGENTS & STREAMING (`api/streaming.py`)                                     │
│     └── Dispatch request to target inference backend (vLLM, Ollama, TensorRT-LLM)     │
│                                                                                        │
│  3. ENVIRONMENTAL REINFORCEMENT & DECAY (`core/memory_field.py`)                      │
│     ├── Post-execution pheromone deposit: M_i(t+\\Delta t) = (1-\\gamma)M_i(t) + \\gamma R │
│     └── Continuous evaporation worker prevents stale path lock-in                       │
└───────────────────────────┬───────────────────────────────────────────┬────────────────┘
                            │                                           │
                            v                                           v
┌──────────────────────────────────────────┐    ┌──────────────────────────────────────────┐
│ STATE CHECKPOINTING & HYDRATION          │    │ GITOPS & CLUSTER HARDENING               │
│ (Phase 12)                               │    │ (Phase 13)                               │
│                                          │    │                                          │
│ • Async Redis AOF & Disk Snapshots       │    │ • ArgoCD Application & ApplicationSet    │
│ • Boot-time Warm-Start Hydration         │    │ • HPA Autoscaling (2 → 10 Replicas)      │
│ • Admin CLI (`cli/checkpoint_ctl.py`)    │    │ • E2E Chaos & Load Validation Suite      │
└──────────────────────────────────────────┘    └──────────────────────────────────────────┘
```

## Implementation Map (Phases 1-13)

| Phase(s) | Module | Primary Capabilities |
|----------|--------|----------------------|
| 1-4 | `core/router_agent.py`, `core/memory_field.py` | Core 4D Pheromone Matrix engine (V, L, S, C), softmax candidate selection, dynamic feedback reinforcement, and decay algorithms. |
| 5-7 | `api/server.py`, `api/streaming.py` | FastAPI core routes (`/v1/chat/completions`, `/v1/completions`, `/v1/models`), async SSE streaming engine, and worker registry. |
| 8-10 | `core/worker_registry.py`, `api/metrics.py` | Worker health-check polling, fallback circuit breakers, and Prometheus metrics instrumentation. |
| 11 | `api/auth.py`, `api/rate_limiter.py`, `api/governance.py` | SHA-256 API Key auth, atomic Redis Lua sliding-window rate limiting (RPM/TPM), and tier-based governance policy overrides (free, pro, enterprise). |
| 12 | `core/checkpointing.py`, `cli/checkpoint_ctl.py` | Periodic matrix snapshotting to Redis/Disk, warm-start state hydration on router startup, and admin CLI management tool. |
| 13 | `deploy/argocd/`, `deploy/helm/`, `scripts/e2e_chaos_hpa_validation.py` | ArgoCD GitOps manifests (Application & ApplicationSet), production Helm profile (`values-prod.yaml`), HPA/KEDA autoscaling, PodDisruptionBudget, and hermetic chaos validation suite. |

## Mathematical Formulation (4D Stigmergic Routing)

The routing engine models candidate LLM endpoints as nodes in a 4D stigmergic memory field. For each candidate node $i$, the engine evaluates four dimensional parameters:

- **Velocity ($V_i$)**: Measured output token generation rate (tokens/sec).
- **Latency ($L_i$)**: Time-to-First-Token (TTFT) in seconds.
- **Stability ($S_i$)**: Exponential moving average of successful non-5xx response ratios.
- **Cost ($C_i$)**: Normalized cost weight per 1M processed tokens.

### Composite Score Computation

Given a set of base matrix weights $w = (w_V, w_L, w_S, w_C)$ modified by the active tenant governance tier bias $\Delta w_{\text{tier}}$, the score $S_i$ for node $i$ is:

$$S_{i} = (w_V + \Delta w_V) V_i - (w_L + \Delta w_L) L_i + (w_S + \Delta w_S) S_i - (w_C + \Delta w_C) C_i$$

> **Weight mapping** in `config.yaml`: `weights.alpha = w_V` (success/velocity), `weights.beta = w_L` (latency), `weights.gamma = w_S` (saturation/stability), `weights.delta = w_C` (capability-fit/cost).

### Boltzmann Softmax Selection

Candidate nodes are sampled stochastically using a Boltzmann distribution governed by temperature parameter $\tau$:

$$P(\text{Node}_i) = \frac{\exp\left(\frac{S_i}{\tau}\right)}{\sum_{j=1}^{N} \exp\left(\frac{S_j}{\tau}\right)}$$

### Pheromone Decay & Reinforcement

Traces continuously decay toward baseline over time $\Delta t$ at evaporation rate $\gamma$, reinforced by observed performance $R_{\text{obs}}$:

$$\mathbf{M}_i(t + \Delta t) = (1 - \gamma)^{\Delta t} \mathbf{M}_i(t) + \gamma R_{\text{obs}}$$
## Configuration & Feature Flags

The router loads `config.yaml` at startup. The `security` and `checkpoints` sections are materialized at runtime and toggled via environment variables for 12-factor/container deployments.

**config.yaml**

```yaml
storage_backend: 'in_memory'      # in_memory | redis
decay_rate: 0.05
decay_interval_sec: 0.5
saturation_scale: 0.1
decay_success: true
decay_capability: true

weights:
  alpha: 1.0    # velocity (success) weight
  beta: 2.0     # latency weight
  gamma: 1.5    # saturation/stability weight
  delta: 1.5    # capability-fit weight
temperature: 0.5

redis:
  host: 'localhost'
  port: 6379
  db: 0
  password: ''

server:
  host: '0.0.0.0'
  port: 8000
  workers:
    - node_id: 'slm-fast'
      type: 'mock'
      base_delay_sec: 0.03
      load_factor: 0.3
      capability_tags: ['slm', 'fast', 'low-latency']
    - node_id: 'llm-reasoner'
      type: 'mock'
      base_delay_sec: 0.15
      load_factor: 0.5
      capability_tags: ['llm', 'reasoning']
```

**Environment Variable Overrides**

| Variable | Scope | Effect |
|----------|-------|--------|
| `STIGMERGIC_STORAGE_BACKEND` | memory field | `in_memory` or `redis` |
| `STIGMERGIC_REDIS_HOST` | redis | Redis host override |
| `STIGMERGIC_SECURITY_ENABLED` | ingress | Enable/disable Phase 11 auth & rate limiting |
| `STIGMERGIC_DEFAULT_TENANT_KEYS` | auth | JSON array of tenant key descriptors |
| `STIGMERGIC_CHECKPOINTS_ENABLED` | router | Enable/disable Phase 12 checkpointing |
| `STIGMERGIC_CHECKPOINT_INTERVAL` | router | Snapshot interval (s) |
| `STIGMERGIC_CHECKPOINT_STORAGE_PATH` | router | Local disk checkpoint directory |

## Ingress Governance & Multi-Tenancy

All requests pass through the Phase 11 FastAPI middleware before reaching the 4D routing engine. Bearer tokens are SHA-256 hashed and resolved to a `TenantContext`; an atomic Redis Lua token bucket enforces per-tenant RPM/TPM quotas, with tier-specific matrix weight overrides steering traffic toward the right backends.

**Tenant Tiers & Matrix Weight Overrides**

| Tier | RPM Limit | TPM Limit | Weight Delta ($\Delta w$) | Behavior |
|------|-----------|-----------|---------------------------|----------|
| `free` | 60 | 10,000 | $w_C \times 2.0$, $w_S \times 0.5$ | Prioritizes cost-efficient backends; restricts burst concurrency. |
| `pro` | 600 | 100,000 | Baseline $(1.0, 1.2, 2.0, 0.8)$ | Balanced 4D distribution across latency, stability, and speed. |
| `enterprise` | 6,000 | 1,000,000 | $w_L \times 1.5$, $w_V \times 1.5$ | Minimizes TTFT and maximizes streaming throughput. |
## Installation & Deployment

### 1. Local Development Setup

```bash
# Clone repository
git clone https://github.com/aragit/stigmergic-mesh-router.git
cd stigmergic-mesh-router

# Create virtual environment & install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run unit and integration test suite
pytest -v
```

### 2. Docker Compose Quickstart

```bash
# Spin up Redis, Prometheus, Grafana, and 3 Router instances (mock workers)
docker-compose up --build -d

# Check cluster status
curl http://localhost:8000/v1/models
```

### 3. Production Helm & GitOps Deployment

```bash
# Lint chart using production configuration
helm lint deploy/helm/stigmergic-mesh-router \
  -f deploy/helm/stigmergic-mesh-router/values-prod.yaml

# Install chart
helm install stigmergic-mesh-router deploy/helm/stigmergic-mesh-router \
  --namespace stigmergic-mesh \
  --create-namespace \
  -f deploy/helm/stigmergic-mesh-router/values-prod.yaml
```

```bash
# Apply ArgoCD Application manifest (automated sync, prune, self-heal)
kubectl apply -f deploy/argocd/application.yaml -n argocd

# (Optional) Apply Multi-Environment ApplicationSet
kubectl apply -f deploy/argocd/applicationset.yaml -n argocd
```

## API Reference

### Authentication Header

All protected endpoints require a Bearer token whose SHA-256 hash matches a registered tenant key.

```http
Authorization: Bearer <YOUR_API_KEY>
```

### POST /v1/chat/completions

Executes 4D path selection and forwards the request to the optimal model backend.

**Request**

```http
POST /v1/chat/completions HTTP/1.1
Host: router.stigmergic.internal
Authorization: Bearer sk-tenant-pro-key-99812
Content-Type: application/json

{
  "model": "auto",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain stigmergic coordination in multi-agent networks."}
  ],
  "temperature": 0.7,
  "stream": true
}
```

**Response Headers (Rate Limit Context)**

```http
HTTP/1.1 200 OK
Content-Type: text/event-stream
X-RateLimit-Limit-RPM: 600
X-RateLimit-Remaining-RPM: 599
X-RateLimit-Reset-RPM: 42
X-Tenant-Tier: pro
X-Selected-Node: worker-vllm-us-east-03
```

### GET /v1/models

Returns the list of active backend models registered in the memory field.

**Request**

```http
GET /v1/models HTTP/1.1
Authorization: Bearer sk-tenant-pro-key-99812
```

**Response Body**

```json
{
  "object": "list",
  "data": [
    {
      "id": "auto",
      "object": "model",
      "created": 1723111200,
      "owned_by": "stigmergic-router",
      "permissions": {"tier": "pro", "routing_algorithm": "4d-softmax"}
    },
    {
      "id": "llama-3-70b-instruct",
      "object": "model",
      "created": 1723111200,
      "owned_by": "worker-vllm-us-east-03"
    }
  ]
}
```
## Admin CLI Usage (cli/checkpoint_ctl.py)

The `checkpoint_ctl.py` CLI provides state inspection, manual exports, and force-import for administrative maintenance.

```bash
# 1. Export current live matrix state from Redis to a JSON file
python cli/checkpoint_ctl.py export  --output ./snapshots/manual_backup.json --redis-url redis://localhost:6379/0

# 2. Inspect serialized state file properties
python cli/checkpoint_ctl.py inspect  --input ./snapshots/manual_backup.json

# 3. Force-import state snapshot into local Redis store
python cli/checkpoint_ctl.py import  --input ./snapshots/manual_backup.json --redis-url redis://localhost:6379/0
```

## Verification & Chaos Validation

### Running Full Test Suite

```bash
pytest -v
```

### Running E2E Chaos & HPA Validation Script

The synthetic chaos runner validates cluster scaling, pod termination recovery, and rate-limit enforcement.

```bash
python scripts/e2e_chaos_hpa_validation.py \
  --namespace stigmergic-mesh \
  --target-host http://localhost:8000 \
  --concurrent-tenants 10 \
  --duration 120
```

The runner executes three phases:

- **Phase A - High-Concurrency Load Injection:** Spikes concurrent tenant traffic to trigger HPA pod scaling (2 -> 10 replicas).
- **Phase B - Fault Injection & Pod Resiliency:** Terminates active router pods mid-stream and verifies zero-downtime rollover with warm-start checkpoint hydration (hydration < 200ms).
- **Phase C - Rate-Limit & Metric Assertion:** Confirms `429` responses under tenant quota breaches and verifies metric propagation to Prometheus (`stigmergic_checkpoint_restore_status == 1`).

### Running Live Simulation

```bash
python run_simulation.py --requests 30
```

## Repository Structure

```
stigmergic-mesh-router/
├── config.yaml                 # Decay rates, weights, temperature, worker map
├── docker-compose.yml          # API server + mock workers + observability
├── prometheus.yml              # Prometheus scrape config
├── run_simulation.py           # Quickstart 30-request mock execution loop
├── core/
│   ├── memory_field.py        # Shared pheromone substrate (4D V/L/S/C)
│   ├── router_agent.py        # Stigmergic routing + trace updates
│   ├── worker_node.py         # Hardware-agnostic worker abstraction
│   ├── worker_registry.py     # Worker registration & health polling
│   ├── checkpointing.py       # Matrix snapshot / hydration (Phase 12)
│   └── decay_engine.py        # Background evaporation engine
├── api/
│   ├── __init__.py
│   ├── server.py              # FastAPI ingress (OpenAI-compatible endpoints)
│   ├── auth.py                # SHA-256 API key auth (Phase 11)
│   ├── rate_limiter.py        # Redis Lua token bucket (Phase 11)
│   ├── governance.py          # Tenant context + weight overrides (Phase 11)
│   ├── streaming.py           # Async SSE dispatch engine
│   └── metrics.py             # Prometheus instrumentation
├── cli/checkpoint_ctl.py      # Checkpoint admin CLI (Phase 12)
├── deploy/argocd/             # GitOps Application + ApplicationSet (Phase 13)
├── deploy/helm/stigmergic-mesh-router/   # Helm chart + values-prod (Phase 13)
├── scripts/                   # k3d, helm/HPA validation, E2E chaos runner
├── tests/                     # Unit + E2E chaos test suite
└── benchmarks/                # Failover & capability-routing benchmarks
```
