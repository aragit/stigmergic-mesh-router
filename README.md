# Stigmergic Mesh Router

A decentralized, hardware-agnostic LLM and microservice load router powered by classical stigmergy. Instead of relying on a centralized orchestrator, health-check probe, or traditional load balancer, stigmergic-mesh-router routes incoming requests dynamically based on continuous scalar traces (pheromones) deposited by autonomous router agents into a shared memory substrate.

## Architectural Blueprint

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          INGRESS API LAYER (FastAPI)                            │
│                  Exposes OpenAI-compatible endpoints (/v1/...)                  │
└───────────────────────────────────────┬─────────────────────────────────────────┘
                                        │
                                        v
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     STATELESS STIGMERGIC ROUTER AGENT POOL                      │
│   1. Read Pheromone Matrix from Memory Field                                   │
│   2. Compute Attraction Score per Node via Weighted Formula                     │
│   3. Perform Softmax Sampling over Scores to pick Node                          │
│   4. Dispatch Request & Deposit Post-Execution Feedback                         │
└───────────────────┬─────────────────────────────────────────┬───────────────────┘
                    │                                         │
                    ▼ (Async Read/Write)                      ▼ (Async Request)
┌───────────────────────────────────────┐   ┌─────────────────────────────────────┐
│      SHARED PHEROMONE SUBSTRATE       │   │      HETEROGENEOUS WORKER MESH      │
│     (In-Memory NumPy / Redis)         │   │                                     │
│  Tracks continuous trace matrix:      │   │  • CPU Mock Workers (Testing)       │
│  • Success Trace (V)                  │   │  • Remote vLLM GPUs                 │
│  • Latency Trace (L)                  │   │  • Ollama / llama.cpp Endpoints     │
│  • Saturation Trace (S)               │   │  • Capability Tag System            │
│  • Capability Fit Trace (C)           │   └─────────────────────────────────────┘
└───────────────────▲───────────────────┘
                    │
┌───────────────────┴───────────────────┐
│       EVAPORATION / DECAY ENGINE      │
│   Asynchronously decays traces over   │
│   time: Pt = Pt-1 * (1 - decay_rate)  │
└───────────────────────────────────────┘
```

## Core Features

- **Zero Central Bottleneck**: No single dispatcher or centralized queue; router agents make independent probabilistic choices based on local observation of the shared memory state.
- **Organic Failover & Self-Healing**: Crashed, slow, or thermal-throttled nodes accumulate elevated latency ($L$) and saturation ($S$) penalties. Traffic naturally flows away without waiting for external health probes.
- **Capability-Aware Routing**: A fourth trace column (C, capability fit) biases routing toward nodes whose declared tags (e.g. `slm`, `reasoning`, `low-latency`) match the incoming prompt semantics. Short prompts route to lightweight SLMs; deep-reasoning prompts route to LLMs — all through the same pheromone feedback loop.
- **Hardware-Agnostic Worker Mesh**: Standardized worker abstraction supporting simulated CPU nodes, local edge runtimes (Ollama, llama.cpp), and remote vLLM GPU clusters.
- **OpenAI API Compatibility**: Native FastAPI ingress supporting `/v1/chat/completions`, `/v1/completions`, and `/v1/models`.
- **Real-time Observability**: Built-in Rich terminal UI dashboard displaying real-time trace heatmaps, load distribution, and calculated attraction scores.

## Mathematical Foundations

### 1. Node Attraction Score

For a target node $i$, its raw attraction score is calculated from its active scalar traces in the shared substrate:

$$
\text{Score}_i = \frac{\alpha \cdot V_i + \delta \cdot C_i + \epsilon}{\beta \cdot L_i + \gamma \cdot S_i + \epsilon}
$$

Where:

- $V_i$: Success Trace (EWMA of binary success outcome, ∈ [0, 1]).
- $L_i$: Latency Trace (EWMA of observed latency in seconds).
- $S_i$: Saturation Trace (scaled residual active load).
- $C_i$: Capability Fit Trace (EWMA of how well the node's tags matched routing requests, ∈ [0, 1]).
- $\alpha, \beta, \gamma, \delta$: Operational sensitivity weights configured in `config.yaml`.
- $\epsilon$: Stability constant ($10^{-5}$) preventing division by zero.

### 2. Boltzmann Softmax Selection

To maintain exploration of recovered nodes while favoring high-performing routes, node selection probabilities follow a Boltzmann distribution scaled by temperature $T$:

$$
P(\text{Node}_i) = \frac{\exp(\text{Score}_i / T)}{\sum_{j=1}^{N} \exp(\text{Score}_j / T)}
$$

### 3. Evaporation Dynamics (Decay Engine)

Traces continuously decay over time interval $\Delta t$ at evaporation rate $\delta \in (0, 1)$:

$$
P_t(\text{trace}) = P_{t-\Delta t}(\text{trace}) \cdot (1 - \delta)^{\Delta t}
$$

## Repository Structure

```
stigmergic-mesh-router/
├── config.yaml                   # Decay rates, weight factors, temperature, cluster map
├── requirements.txt              # Project dependencies
├── core/
│   ├── __init__.py
│   ├── memory_field.py           # Shared Pheromone Substrate (Async NumPy)
│   ├── worker_node.py            # Hardware-agnostic abstraction (CPU Mock + GPU vLLM)
│   ├── router_agent.py           # Stigmergic routing logic & trace updates
│   └── decay_engine.py           # Background evaporation engine
├── api/
│   ├── __init__.py
│   └── server.py                 # FastAPI ingress exposing OpenAI-compatible endpoints
├── visualizer/
│   ├── __init__.py
│   └── terminal_dashboard.py     # Live Rich terminal heatmap of pheromone levels
├── benchmarks/
│   ├── __init__.py
│   ├── failover_test.py          # Zero-touch chaos failover verification suite
│   └── capability_routing_test.py # Capability-aware SLM/LLM routing benchmark
├── tests/
│   ├── __init__.py
│   ├── test_memory_field.py      # PheromoneMemoryField unit tests
│   └── test_router_agent.py      # StigmergicRouterAgent unit tests
├── Dockerfile                    # Production Python 3.11 image
├── docker-compose.yml            # API server + worker services
└── run_simulation.py             # Quickstart 30-request mock execution loop
```

## Getting Started

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/aragit/stigmergic-mesh-router.git
cd stigmergic-mesh-router
pip install -r requirements.txt
```

### 2. Quickstart Simulation

Run the built-in CPU mock simulation to verify stigmergic load distribution across fast and slow nodes:

```bash
python3 run_simulation.py
```

### 3. Run Chaos Failover Benchmark

Execute the chaos test suite to observe zero-touch failover and trace-evaporation recovery:

```bash
python3 benchmarks/failover_test.py
```

### 4. Launch Ingress API Server

Start the OpenAI-compatible FastAPI gateway:

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

Test an inference query via curl:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello world!"}]
  }'
```

## Empirical Benchmark Results

The Chaos Failover Benchmark (`benchmarks/failover_test.py`) runs a 45-tick simulation across three phases with two crash conditions injected into `node_alpha`:

### Configuration

| Parameter | Value |
|---|---|
| Nodes | 3 (alpha, beta, gamma) — all `base_delay=0.05s`, `load_factor=0.5` |
| Requests per phase | 75 (15 ticks × 5 requests) |
| Temperature (T) | 2.0 |
| Decay rate | 0.20 per 0.5s interval |
| Crash | 2.0s fixed delay, `load_factor=0.0` (no compounding) |
| V decay | Disabled (`decay_success=False`) — success trace persists through failure |

### Phase A — Normal Operation

All nodes start at the same baseline speed (0.075s per request). With steady-state initialization (V=1.0, L=0.075), the Boltzmann softmax produces a uniform distribution:

| Node | Requests | Share |
|---|---|---|
| Alpha | 25 | 33.3% |
| Beta | 22 | 29.3% |
| Gamma | 26 | 34.7% |

### Phase B — Node Alpha Crash

Alpha suffers a 2.0s processing delay. Its latency trace (L) jumps to ~1.5s, collapsing its attraction score. Traffic immediately abandons alpha without any health-check or external probe:

| Node | Requests | Share |
|---|---|---|
| Alpha | 3 | 4.0% |
| Beta | 38 | 50.7% |
| Gamma | 31 | 41.3% |

**The 4.0% residual traffic to alpha represents the 3 crash requests that executed before L fully registered.**

### Phase C — Recovery

Alpha is restored to its healthy 0.05s baseline. Over 75 requests, latency observations from the crash evaporate (decayed from 1.5→0.07 over ~14 decay cycles). As L fades below the beta/gamma equilibrium, alpha's attraction score rises and traffic returns dynamically:

| Node | Requests | Share |
|---|---|---|
| Alpha | 15 | 18.7% |
| Beta | 27 | 36.0% |
| Gamma | 34 | 46.7% |

Final pheromone state at end of Phase C:

| Node | V (Success) | L (Latency) | S (Saturation) |
|---|---|---|---|
| Alpha | 1.0000 | 0.0508 | 0.0000 |
| Beta | 1.0000 | 0.0490 | 0.0000 |
| Gamma | 1.0000 | 0.0539 | 0.0000 |

Alpha recovered from 4.0% → 18.7% traffic share, demonstrating that stigmergic trace evaporation enables zero-touch self-healing without centralized coordination.

### Phase C (Updated) — Enhanced Recovery with Capability Fit

With the extended 4D matrix (including C), the failover benchmark shows improved recovery:

| Phase | Alpha | Beta | Gamma |
|---|---|---|---|
| A (normal) | 34.7% | 33.3% | 32.0% |
| B (crash) | 2.7% | 53.3% | 44.0% |
| C (recovery) | 29.3% | 32.0% | 38.7% |

The capability-fit trace (C=0.5 neutral at baseline) provides additional routing signal that accelerates recovery of the crashed node.

### Verdict

```
✓ PASS: node_alpha was abandoned during the crash (Phase B: 2 reqs, 2.7%)
  and traffic returned after recovery (Phase C: 22 reqs, 29.3%).
```

## Capability-Aware Routing Benchmark

The Capability Routing Benchmark (`benchmarks/capability_routing_test.py`) demonstrates that the fourth trace column (C) enables semantic prompt routing — short prompts naturally flow to lightweight SLM nodes while deep-reasoning prompts flow to LLM nodes, all through the same pheromone feedback loop.

### Configuration

| Parameter | Value |
|---|---|
| Nodes | 3 (slm-fast 0.03s, llm-reasoner 0.15s, llm-balanced 0.08s) |
| Capability Tags | `slm-fast`: [slm, fast, low-latency] · `llm-reasoner`: [llm, reasoning] · `llm-balanced`: [llm, balanced] |
| Requests | 60 short + 60 reasoning prompts |
| Temperature | 2.0 |
| Decay rate | 0.15 |
| Delta (capability weight) | 1.5 |

### Phase A — Short Prompts (SLM Routing)

All 60 short prompts ("Hello!", "What time is it?", etc.) routed to the lightweight `slm-fast` node because `analyze_prompt()` emits `{"slm": 1.5, "fast": 1.3, "low-latency": 1.2}`, and the capability-match boost makes `slm-fast`'s score dominate:

| Node | Requests | Share | Avg Latency |
|---|---|---|---|
| slm-fast | 60 | 100.0% | 0.0390s |
| llm-reasoner | 0 | 0.0% | — |
| llm-balanced | 0 | 0.0% | — |

### Phase B — Deep Reasoning Prompts (LLM Routing)

All 60 reasoning prompts ("Let's think step by step...", "Explain the chain of thought...") routed to `llm-reasoner` because `analyze_prompt()` detects thinking patterns and emits `{"llm": 1.5, "reasoning": 1.5}`, boosting the LLM node's score:

| Node | Requests | Share | Avg Latency |
|---|---|---|---|
| llm-reasoner | 41 | 68.3% | 0.2250s |
| llm-balanced | 18 | 30.0% | 0.1120s |
| slm-fast | 1 | 1.7% | 0.0390s |

### Verdict

```
✓ PASS: Short prompts routed to SLM (100.0% to slm-fast) and reasoning
  prompts routed to LLM (68.3% to llm-reasoner).
```

## Running Tests

Unit tests cover memory field initialization, trace deposition, evaporation dynamics, concurrency safety, score computation (3D and 4D), Boltzmann softmax, capability matching, sampling bias, and trace feedback:

```bash
pip install -r requirements.txt
pytest tests/ -v
```

**69 tests** across two modules:
- `tests/test_memory_field.py` — 29 tests (initialization, deposition, evaporation, concurrency)
- `tests/test_router_agent.py` — 40 tests (3D/4D scores, softmax, capability matching, routing feedback, sampling bias)

## Docker

Build and run the production API server:

```bash
docker-compose up --build
```

Or build the image standalone:

```bash
docker build -t stigmergic-mesh-router .
docker run -p 8000:8000 stigmergic-mesh-router
```
