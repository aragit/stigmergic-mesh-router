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
│  • Saturation Trace (S)               │   └─────────────────────────────────────┘
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
- **Hardware-Agnostic Worker Mesh**: Standardized worker abstraction supporting simulated CPU nodes, local edge runtimes (Ollama, llama.cpp), and remote vLLM GPU clusters.
- **OpenAI API Compatibility**: Native FastAPI ingress supporting `/v1/chat/completions`, `/v1/completions`, and `/v1/models`.
- **Real-time Observability**: Built-in Rich terminal UI dashboard displaying real-time trace heatmaps, load distribution, and calculated attraction scores.

## Mathematical Foundations

### 1. Node Attraction Score

For a target node $i$, its raw attraction score is calculated from its active scalar traces in the shared substrate:

$$
\text{Score}_i = \frac{\alpha \cdot V_i + \epsilon}{\beta \cdot L_i + \gamma \cdot S_i + \epsilon}
$$

Where:

- $V_i$: Success Trace (accumulates based on throughput in tokens/second).
- $L_i$: Latency Trace (accumulates based on execution delay and errors).
- $S_i$: Saturation Trace (accumulates with active concurrency load).
- $\alpha, \beta, \gamma$: Operational sensitivity weights configured in `config.yaml`.
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
│   └── failover_test.py          # Zero-touch chaos failover verification suite
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
