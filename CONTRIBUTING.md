# Contributing to Stigmergic Mesh Router

Thank you for your interest in contributing to **Stigmergic Mesh Router**! We welcome contributions to core multi-agent swarm algorithms, ingress governance, state persistence, checkpointing, and GitOps manifests.

---

## 🛠️ Development Setup

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/aragit/stigmergic-mesh-router.git
cd stigmergic-mesh-router
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Code Quality & Formatting

We strictly enforce linting and type-checking. Before submitting a PR, ensure all checks pass:

```bash
# Linting and import sorting
ruff check .

# Type safety
mypy .
```

## 🧪 Testing Guidelines

Run the full hermetic test suite (unit, integration, and chaos validation):

```bash
pytest -v --cov=.
```

- **Unit & Integration Tests**: place new test suites under `tests/`.
- **Determinism Requirement**: tests must be hermetic and avoid reliance on live external backends unless mocked via test fixtures.

## 🔀 Pull Request Process

- **Branch Naming**: use descriptive names (`feature/swarm-heuristic`, `fix/rate-limiter-lua`, `docs/readme-update`).
- **Commit Messages**: follow Conventional Commits (`feat:`, `fix:`, `chore:`, `docs:`, `test:`).
- **Automated Verification**: ensure all tests and static analysis pass locally before pushing.
- **License Compliance**: all submitted contributions are licensed under the Apache-2.0 License.
