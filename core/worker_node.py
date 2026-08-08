"""Worker node abstractions: abstract base, CPU mock, GPU/vLLM client, and Ollama client.

Supports both synchronous inference (``execute_inference``) and streaming
inference (``execute_streaming``) for SSE-compatible responses.  Streaming
workers can capture Time-To-First-Token (TTFT) and Inter-Token Latency (ITL)
metrics for real-time trace deposits.
"""

import asyncio
import json
import random
import time
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional

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

    async def execute_streaming(
        self,
        prompt: str,
        max_tokens: int = 128,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute streaming inference, yielding chunk metadata.

        The default implementation delegates to :meth:`execute_inference`
        and yields a single chunk.  Subclasses that support native token
        streaming should override this method.

        Each yielded dictionary contains:

        ``chunk_text``
            The text content of this token/chunk.
        ``latency_sec``
            Cumulative latency at the time this chunk was emitted.
        ``is_first_token``
            True for the first chunk (used for TTFT measurement).
        ``token_count``
            Number of tokens decoded in this chunk.
        ``total_tokens``
            Cumulative token count at this point.
        ``success``
            Whether the request succeeded.
        ``active_load``
            Current active load on the worker.
        ``node_id``
            The worker that produced this chunk.
        """
        start = time.monotonic()
        result = await self.execute_inference(prompt, max_tokens)
        elapsed = time.monotonic() - start
        yield {
            "chunk_text": result.get("text", ""),
            "latency_sec": elapsed,
            "is_first_token": True,
            "token_count": result.get("tokens", 0),
            "total_tokens": result.get("tokens", 0),
            "success": result.get("success", True),
            "active_load": result.get("active_load", 0),
            "node_id": result.get("node_id", self.node_id),
        }


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

    async def execute_streaming(
        self,
        prompt: str,
        max_tokens: int = 128,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Simulate streaming token delivery with TTFT and ITL metrics.

        Yields synthetic chunks with realistic timing to simulate
        token-by-token streaming.  The first chunk includes TTFT data;
        subsequent chunks include ITL (inter-token latency).
        """
        async with self._lock:
            self._active_load += 1
            start_load = self._active_load
            self.requests_served += 1

        request_start = time.monotonic()
        try:
            delay = self.base_delay_sec * (1.0 + start_load * self.load_factor)
            total_tokens = random.randint(max(10, max_tokens // 4), max_tokens)
            success = random.random() < self.success_rate

            # Simulate TTFT (first token delay)
            ttft = delay * 0.3
            await asyncio.sleep(ttft)

            tokens_sent = 0
            yield {
                "chunk_text": "Simulating token stream...\n",
                "latency_sec": ttft,
                "is_first_token": True,
                "token_count": 1,
                "total_tokens": 1,
                "success": success,
                "active_load": max(0, self._active_load - 1),
                "node_id": self.node_id,
            }
            tokens_sent = 1

            # Stream remaining tokens with ITL
            remaining_delay = delay - ttft
            for i in range(total_tokens - 1):
                chunk_start = time.monotonic()
                await asyncio.sleep(remaining_delay / max(1, total_tokens - 1))
                time.monotonic() - chunk_start
                tokens_sent += 1
                yield {
                    "chunk_text": f"Token {tokens_sent}\n",
                    "latency_sec": time.monotonic() - request_start,
                    "is_first_token": False,
                    "token_count": 1,
                    "total_tokens": tokens_sent,
                    "success": success,
                    "active_load": max(0, self._active_load - 1),
                    "node_id": self.node_id,
                }

            yield {
                "chunk_text": "",
                "latency_sec": time.monotonic() - request_start,
                "is_first_token": False,
                "token_count": 0,
                "total_tokens": total_tokens,
                "success": success,
                "active_load": max(0, self._active_load - 1),
                "node_id": self.node_id,
                "done": True,
            }

        finally:
            async with self._lock:
                self._active_load -= 1


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

    async def execute_streaming(
        self,
        prompt: str,
        max_tokens: int = 128,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream from vLLM's native SSE endpoint.

        Connects to the vLLM ``/v1/completions`` endpoint with
        ``stream=True`` and yields each SSE chunk.  Captures TTFT on
        the first chunk and token velocity on completion.
        """
        async with self._lock:
            self._active_load += 1
            self.requests_served += 1

        start = time.monotonic()
        ttft_sent = False
        total_tokens = 0

        try:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            payload = {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "stream": True,
            }

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/v1/completions",
                    headers=headers,
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        if line.strip() == "data: [DONE]":
                            continue
                        try:
                            data = json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue
                        choices = data.get("choices", [])
                        if not choices:
                            continue
                        text = choices[0].get("text", "")

                        elapsed = time.monotonic() - start
                        is_first = not ttft_sent
                        if is_first:
                            ttft_sent = True
                        total_tokens += 1

                        yield {
                            "chunk_text": text,
                            "latency_sec": elapsed,
                            "is_first_token": is_first,
                            "token_count": 1,
                            "total_tokens": total_tokens,
                            "success": True,
                            "active_load": max(0, self._active_load - 1),
                            "node_id": self.node_id,
                        }

                    yield {
                        "chunk_text": "",
                        "latency_sec": time.monotonic() - start,
                        "is_first_token": False,
                        "token_count": 0,
                        "total_tokens": total_tokens,
                        "success": True,
                        "active_load": max(0, self._active_load - 1),
                        "node_id": self.node_id,
                        "done": True,
                    }

        except Exception as exc:
            yield {
                "chunk_text": "",
                "latency_sec": time.monotonic() - start,
                "is_first_token": not ttft_sent,
                "token_count": 0,
                "total_tokens": total_tokens,
                "success": False,
                "active_load": max(0, self._active_load - 1),
                "node_id": self.node_id,
                "error": str(exc),
                "done": True,
            }
        finally:
            async with self._lock:
                self._active_load -= 1


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

    async def execute_streaming(
        self,
        prompt: str,
        max_tokens: int = 128,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream from Ollama's ``/api/generate`` endpoint.

        Ollama supports streaming via ``"stream": True``.  Each SSE
        chunk is parsed and yielded with timing metadata.
        """
        async with self._lock:
            self._active_load += 1
            self.requests_served += 1

        start = time.monotonic()
        ttft_sent = False
        total_tokens = 0

        try:
            payload: Dict[str, Any] = {
                "model": self.model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "num_predict": max_tokens,
                },
            }

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/generate",
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        text = data.get("response", "")
                        done = data.get("done", False)

                        elapsed = time.monotonic() - start
                        is_first = not ttft_sent
                        if is_first:
                            ttft_sent = True
                        if text:
                            total_tokens += len(text.split())

                        if done:
                            yield {
                                "chunk_text": "",
                                "latency_sec": elapsed,
                                "is_first_token": is_first,
                                "token_count": 0,
                                "total_tokens": total_tokens,
                                "success": True,
                                "active_load": max(0, self._active_load - 1),
                                "node_id": self.node_id,
                                "done": True,
                            }
                        else:
                            yield {
                                "chunk_text": text,
                                "latency_sec": elapsed,
                                "is_first_token": is_first,
                                "token_count": len(text.split()) if text else 0,
                                "total_tokens": total_tokens,
                                "success": True,
                                "active_load": max(0, self._active_load - 1),
                                "node_id": self.node_id,
                            }

        except Exception as exc:
            yield {
                "chunk_text": "",
                "latency_sec": time.monotonic() - start,
                "is_first_token": not ttft_sent,
                "token_count": 0,
                "total_tokens": total_tokens,
                "success": False,
                "active_load": max(0, self._active_load - 1),
                "node_id": self.node_id,
                "error": str(exc),
                "done": True,
            }
        finally:
            async with self._lock:
                self._active_load -= 1
