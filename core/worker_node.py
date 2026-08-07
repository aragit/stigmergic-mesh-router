"""Worker node abstractions: abstract base, CPU mock, GPU/vLLM client, and Ollama client."""

import asyncio
import random
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import httpx


class BaseWorkerNode(ABC):
    """Abstract base class for all worker nodes in the mesh.

    Subclasses must implement :meth:`execute_inference`.  When declared
    via configuration, each worker can expose a list of ``capability_tags``
    (e.g. ``["slm", "fast", "low-latency"]``) that the router uses to
    compute a capability-fit trace (C) and bias routing decisions.
    """

    def __init__(
        self,
        node_id: str,
        capability_tags: Optional[List[str]] = None,
    ) -> None:
        self.node_id: str = node_id
        self.capability_tags: List[str] = capability_tags or []

    @abstractmethod
    async def execute_inference(
        self,
        prompt: str,
        max_tokens: int = 128,
    ) -> Dict[str, Any]:
        """Execute a single inference request.

        Parameters
        ----------
        prompt
            The input prompt / text to process.
        max_tokens
            Maximum number of tokens to generate.

        Returns
        -------
        Dict[str, Any]
            Result dictionary containing at least ``latency_sec``,
            ``tokens``, ``success``, and ``active_load`` (residual load
            after the request completes).
        """
        ...


class CPUMockWorkerNode(BaseWorkerNode):
    """Simulated CPU-bound worker whose delay scales with concurrency.

    The simulated latency follows::

        delay = base_delay * (1 + start_load * load_factor)

    where ``start_load`` is the number of concurrent requests on this
    node at the moment the request began.  This models real-world
    degradation under load.
    """

    def __init__(
        self,
        node_id: str,
        base_delay_sec: float = 0.05,
        load_factor: float = 0.5,
        success_rate: float = 1.0,
        capability_tags: Optional[List[str]] = None,
    ) -> None:
        super().__init__(node_id, capability_tags)
        self.base_delay_sec: float = base_delay_sec
        self.load_factor: float = load_factor
        self.success_rate: float = success_rate
        self._active_load: int = 0
        self._lock = asyncio.Lock()
        self.requests_served: int = 0

    async def execute_inference(
        self,
        prompt: str,
        max_tokens: int = 128,
    ) -> Dict[str, Any]:
        async with self._lock:
            self._active_load += 1
            start_load = self._active_load

        result: Dict[str, Any] = {}
        try:
            delay = self.base_delay_sec * (1.0 + start_load * self.load_factor)
            await asyncio.sleep(delay)

            tokens = random.randint(max(10, max_tokens // 4), max_tokens)
            success = random.random() < self.success_rate

            self.requests_served += 1

            result = {
                "node_id": self.node_id,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "tokens": tokens,
                "latency_sec": delay,
                "success": success,
            }
        finally:
            async with self._lock:
                self._active_load -= 1
                result["active_load"] = max(0, self._active_load)

        return result


class GPUvLLMWorkerNode(BaseWorkerNode):
    """Asynchronous client for remote vLLM ``/v1/completions`` endpoints.

    Wraps ``httpx.AsyncClient`` to issue non-blocking HTTP requests to a
    remote LLM inference server.  Supports optional bearer-token auth.
    """

    def __init__(
        self,
        node_id: str,
        base_url: str,
        api_key: str = "",
        model: str = "default",
        timeout_sec: float = 30.0,
        capability_tags: Optional[List[str]] = None,
    ) -> None:
        super().__init__(node_id, capability_tags)
        self.base_url: str = base_url.rstrip("/")
        self.api_key: str = api_key
        self.model: str = model
        self.timeout: httpx.Timeout = httpx.Timeout(timeout_sec)
        self._active_load: int = 0
        self._lock = asyncio.Lock()
        self.requests_served: int = 0

    async def execute_inference(
        self,
        prompt: str,
        max_tokens: int = 128,
    ) -> Dict[str, Any]:
        async with self._lock:
            self._active_load += 1

        start = time.monotonic()
        result: Dict[str, Any] = {}
        try:
            headers: Dict[str, str] = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            payload: Dict[str, Any] = {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": max_tokens,
            }

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/v1/completions",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            elapsed = time.monotonic() - start
            self.requests_served += 1

            choices = data.get("choices", [])
            text = choices[0].get("text", "") if choices else ""
            usage = data.get("usage", {})
            tokens = usage.get("completion_tokens", len(text.split()))

            result = {
                "node_id": self.node_id,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "tokens": tokens,
                "latency_sec": elapsed,
                "success": True,
                "text": text,
            }
        except Exception as exc:
            elapsed = time.monotonic() - start
            result = {
                "node_id": self.node_id,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "tokens": 0,
                "latency_sec": elapsed,
                "success": False,
                "error": str(exc),
            }
        finally:
            async with self._lock:
                self._active_load -= 1
                result["active_load"] = max(0, self._active_load)

        return result


class OllamaWorkerNode(BaseWorkerNode):
    """Asynchronous client for Ollama ``/api/generate`` and ``/api/chat`` endpoints.

    Communicates with a local or remote Ollama server using the native
    Ollama REST API.  Each worker can declare capability tags (e.g.
    ``["slm", "fast"]``) so the stigmergic router can prefer it for
    requests that match its declared profile.
    """

    def __init__(
        self,
        node_id: str,
        base_url: str,
        model: str = "phi3",
        timeout_sec: float = 60.0,
        capability_tags: Optional[List[str]] = None,
    ) -> None:
        super().__init__(node_id, capability_tags)
        self.base_url: str = base_url.rstrip("/")
        self.model: str = model
        self.timeout: httpx.Timeout = httpx.Timeout(timeout_sec)
        self._active_load: int = 0
        self._lock = asyncio.Lock()
        self.requests_served: int = 0

    async def execute_inference(
        self,
        prompt: str,
        max_tokens: int = 128,
    ) -> Dict[str, Any]:
        async with self._lock:
            self._active_load += 1

        start = time.monotonic()
        result: Dict[str, Any] = {}
        try:
            payload: Dict[str, Any] = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                },
            }

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            elapsed = time.monotonic() - start
            self.requests_served += 1

            text = data.get("response", "")
            done = data.get("done", True)
            tokens = data.get("eval_count", len(text.split()))

            result = {
                "node_id": self.node_id,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "tokens": tokens,
                "latency_sec": elapsed,
                "success": done,
                "text": text,
            }
        except Exception as exc:
            elapsed = time.monotonic() - start
            result = {
                "node_id": self.node_id,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "tokens": 0,
                "latency_sec": elapsed,
                "success": False,
                "error": str(exc),
            }
        finally:
            async with self._lock:
                self._active_load -= 1
                result["active_load"] = max(0, self._active_load)

        return result
