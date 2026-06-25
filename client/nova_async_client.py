"""
NovaAsyncOpenAIClient - OpenAI-Compatible Async Client with AWS SigV4 Support

Intended for use by training and evaluation scripts running on HyperPod pods
within the BYOO (Bring Your Own Orchestrator) reward environment. This file is
NOT consumed by the Step Functions pipeline or Lambda handlers — it is a utility
for custom reward functions that need to call the Nova inference endpoint during
multi-turn RL training.

This client provides a drop-in replacement for OpenAI's AsyncOpenAI client,
adding support for the BYOO environment's async task polling pattern and
AWS SigV4 authentication for Lambda Function URLs.

Features:
    - OpenAI-compatible chat.completions.create() interface
    - Automatic async task polling (POST to create, GET to poll)
    - AWS SigV4 request signing for Lambda Function URLs
    - Connection pooling and resource management
    - Rollout result reporting for training feedback

Usage:
    >>> from client.nova_async_client import NovaAsyncOpenAIClient
    >>>
    >>> async with NovaAsyncOpenAIClient(
    ...     base_url="https://your-function-url.lambda-url.us-east-1.on.aws",
    ...     aws_region="us-east-1",
    ...     aws_service="lambda",
    ... ) as client:
    ...     response = await client.chat.completions.create(
    ...         model="nova",
    ...         messages=[{"role": "user", "content": "Hello"}],
    ...     )
    ...     print(response.choices[0].message.content)
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, Generator, List, Literal, TypedDict

import httpx
from openai import AsyncOpenAI
from openai.resources.chat import AsyncChat as OpenAIAsyncChat
from openai.resources.chat.completions import (
    AsyncCompletions as OpenAIAsyncChatCompletions,
)
from openai.resources.completions import (
    AsyncCompletions as OpenAIAsyncCompletions,
)

try:
    import boto3
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest
except ImportError:
    boto3 = None
    SigV4Auth = None
    AWSRequest = None

__all__ = ["NovaAsyncOpenAIClient", "SigV4HTTPXAuth", "RolloutMetric", "RolloutResponse"]

logger = logging.getLogger(__name__)



class RolloutMetric(TypedDict):
    """
    Individual metric/reward score in a rollout response.

    Attributes:
        name: Name of the component score (e.g., "accuracy", "tool_selection")
        value: Numeric value of the score
        type: Score category - "Reward" for training signal, "Metric" for tracking
    """
    name: str
    value: float
    type: Literal["Reward", "Metric"]


class RolloutResponse(TypedDict):
    """
    Rollout response payload for reporting results back to the server.

    Attributes:
        id: Sample identifier (must match the input sample ID)
        aggregate_reward_score: Computed aggregate reward/metric score
        metrics_list: List of individual component scores
    """
    id: str
    aggregate_reward_score: float
    metrics_list: List[RolloutMetric]


class SigV4HTTPXAuth(httpx.Auth):
    """
    AWS SigV4 authentication handler for httpx.

    Automatically signs HTTP requests using AWS Signature Version 4.
    Credentials are refreshed on each request, supporting:
    - IAM roles (EC2, ECS, Lambda, SageMaker)
    - AWS CLI profiles
    - Environment variables
    - Temporary credentials (STS AssumeRole)

    Args:
        service: AWS service name for signing (e.g., "lambda", "execute-api")
        region: AWS region (e.g., "us-east-1")
        profile: Optional AWS CLI profile name
    """

    def __init__(
        self,
        service: str,
        region: str,
        profile: str | None = None,
    ):
        if boto3 is None:
            raise RuntimeError(
                "boto3 is required for SigV4 signing. "
                "Install it with: pip install boto3"
            )

        self.service = service
        self.region = region
        self._session = boto3.Session(
            profile_name=profile,
            region_name=region,
        )

    def auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        """
        Sign each outgoing request with SigV4 credentials.

        Fetches current AWS credentials (auto-refreshed by boto3),
        converts the httpx request to botocore format, applies the
        SigV4 signature, and updates request headers.
        """
        credentials = self._session.get_credentials()
        if credentials is None:
            raise RuntimeError(
                "Unable to locate AWS credentials. "
                "Configure credentials via AWS CLI, environment variables, or IAM role."
            )

        frozen = credentials.get_frozen_credentials()

        headers = dict(request.headers)
        # Remove 'connection' header — not used in SigV4 calculation
        # and causes signature mismatch if included
        headers.pop("connection", None)
        headers.pop("Connection", None)

        # Ensure Host header is present (required for SigV4)
        has_host = any(k.lower() == "host" for k in headers.keys())
        if not has_host:
            if request.url.port and request.url.port not in (80, 443):
                headers["Host"] = f"{request.url.host}:{request.url.port}"
            else:
                headers["Host"] = request.url.host

        aws_request = AWSRequest(
            method=request.method,
            url=str(request.url),
            data=request.content,
            headers=headers,
        )

        SigV4Auth(frozen, self.service, self.region).add_auth(aws_request)
        request.headers.update(dict(aws_request.headers))

        yield request



class NovaAsyncOpenAIClient(AsyncOpenAI):
    """
    OpenAI-compatible async client with task polling and SigV4 auth.

    Provides a familiar interface for interacting with the BYOO environment
    deployed by the nova-forge. Supports the async create-then-poll
    pattern used by the Lambda proxy backend.

    Features:
        - Drop-in replacement for AsyncOpenAI
        - Automatic task polling for long-running inference
        - AWS SigV4 authentication for Lambda Function URLs
        - Connection pooling and keepalive
        - Rollout response reporting for training feedback

    Example:
        >>> async with NovaAsyncOpenAIClient(
        ...     base_url="https://your-function-url.lambda-url.us-east-1.on.aws",
        ...     aws_region="us-east-1",
        ...     aws_service="lambda",
        ... ) as client:
        ...     response = await client.chat.completions.create(
        ...         model="nova",
        ...         messages=[{"role": "user", "content": "Hello"}],
        ...     )
        ...     print(response.choices[0].message.content)
    """

    def __init__(
        self,
        *,
        base_url: str,
        timeout_s: float = 60.0,
        poll_interval_s: float = 0.8,
        max_wait_s: float | None = None,
        aws_region: str | None = None,
        aws_profile: str | None = None,
        aws_service: str = "execute-api",
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        sample_id: Any | None = None,
        request_id: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Nova async client.

        Args:
            base_url: Base URL of the BYOO backend (Lambda Function URL)
            timeout_s: HTTP request timeout in seconds
            poll_interval_s: Interval between polling attempts in seconds
            max_wait_s: Maximum total wait for task completion (defaults to timeout_s)
            aws_region: AWS region for SigV4 signing (enables signing if set)
            aws_profile: Optional AWS CLI profile name
            aws_service: AWS service name for signing ("lambda" or "execute-api")
            max_connections: Maximum concurrent HTTP connections
            max_keepalive_connections: Maximum keepalive connections in pool
            sample_id: Optional sample identifier to include in requests
            request_id: Optional request identifier to include in requests
            **kwargs: Additional arguments passed to AsyncOpenAI
        """
        super().__init__(api_key="no_key", base_url=base_url)  # nosec B105 # noqa: S105 - placeholder, not a real secret

        raw_base = base_url.rstrip("/")
        self._api_base_url = raw_base
        # Remove /v1 suffix for polling endpoint
        self._polling_base_url = (
            raw_base[: -len("/v1")] if raw_base.endswith("/v1") else raw_base
        )

        self._timeout_s = timeout_s
        self._poll_interval_s = poll_interval_s
        self._max_wait_s = timeout_s if max_wait_s is None else max_wait_s
        self._aws_region = aws_region
        self._aws_profile = aws_profile
        self._aws_service = aws_service

        self._sign_requests = aws_region is not None

        if self._sign_requests and boto3 is None:
            raise RuntimeError(
                "boto3 is required for SigV4 signing. "
                "Install it with: pip install boto3"
            )

        auth = None
        if self._sign_requests:
            auth = SigV4HTTPXAuth(
                service=self._aws_service,
                region=self._aws_region,
                profile=self._aws_profile,
            )

        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=5.0,
                read=timeout_s,
                write=10.0,
                pool=None,
            ),
            auth=auth,
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive_connections,
            ),
        )

        self._chat_resource = _NovaAsyncChat(self)
        self._chat_completions = _NovaChatCompletions(self)
        self._text_completions = _NovaCompletions(self)

        self._sample_id = sample_id
        self._request_id = request_id

    @property
    def chat(self) -> OpenAIAsyncChat:
        """OpenAI-compatible chat interface."""
        return self._chat_resource

    @property
    def completions(self) -> OpenAIAsyncCompletions:
        """OpenAI-compatible text completions interface."""
        return self._text_completions

    def _build_headers(self) -> Dict[str, str]:
        """Build default HTTP headers for requests."""
        return {"Content-Type": "application/json"}

    def _make_url(self, path: str) -> str:
        """Construct full URL from path."""
        if not path.startswith("/"):
            path = f"/{path}"
        return f"{self._polling_base_url}{path}"

    async def report_rollout(
        self,
        rollout_response: RolloutResponse | Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Report rollout results back to the server.

        Sends computed metrics and rewards for a rollout sample to the
        backend for tracking and training signal computation.

        Args:
            rollout_response: Payload containing:
                - id: Sample identifier
                - aggregate_reward_score: Overall reward score
                - metrics_list: List of individual component scores

        Returns:
            Server response as dictionary

        Raises:
            ValueError: If rollout_response is missing required fields
            RuntimeError: If the request fails
        """
        if not isinstance(rollout_response, dict):
            raise ValueError("rollout_response must be a dictionary")

        required_fields = {"id", "aggregate_reward_score", "metrics_list"}
        missing_fields = required_fields - set(rollout_response.keys())
        if missing_fields:
            raise ValueError(
                f"rollout_response missing required fields: {missing_fields}"
            )

        metrics_list = rollout_response.get("metrics_list", [])
        if not isinstance(metrics_list, list):
            raise ValueError("metrics_list must be a list")

        for i, metric in enumerate(metrics_list):
            if not isinstance(metric, dict):
                raise ValueError(f"metrics_list[{i}] must be a dictionary")

            metric_required = {"name", "value", "type"}
            metric_missing = metric_required - set(metric.keys())
            if metric_missing:
                raise ValueError(
                    f"metrics_list[{i}] missing required fields: {metric_missing}"
                )

            if metric.get("type") not in {"Reward", "Metric"}:
                raise ValueError(
                    f"metrics_list[{i}].type must be 'Reward' or 'Metric', "
                    f"got: {metric.get('type')}"
                )

        logger.info(f"Reporting rollout response for sample: {rollout_response.get('id')}")

        headers = self._build_headers()
        response = await self._request(
            method="POST",
            path="/v1/rollouts/report",
            json_payload=rollout_response,
            headers=headers,
        )

        data = _parse_json_response(response)
        logger.debug(f"Rollout report response: {data}")
        return data


    async def _create_and_poll(
        self,
        *,
        path: str,
        payload: Dict[str, Any],
        expect: Literal["chat", "completion"],
    ) -> Dict[str, Any]:
        """
        Submit a task and poll until completion.

        The BYOO backend uses an async pattern: POST creates a task and
        returns a step_id, then GET polls until the result is ready.
        If the backend returns a completion directly, polling is skipped.

        Args:
            path: API endpoint path
            payload: Request payload
            expect: Expected response type ("chat" or "completion")

        Returns:
            Final completion result

        Raises:
            RuntimeError: If task fails or returns unexpected format
            TimeoutError: If polling exceeds max_wait_s
        """
        headers = self._build_headers()

        # Attach sample/request IDs if configured
        if self._sample_id is not None:
            payload = {**payload, "sample_id": self._sample_id}
        if self._request_id is not None:
            payload = {**payload, "request_id": self._request_id}

        logger.debug(f"Creating task at {path}")
        response = await self._request(
            method="POST",
            path=path,
            json_payload=payload,
            headers=headers,
        )
        data = _parse_json_response(response)

        # Check if response is already a completion (synchronous mode)
        if _is_expected_payload(data, expect):
            logger.debug("Received immediate completion response")
            return data

        # Extract step_id for polling
        step_id = data.get("step_id")
        if not step_id:
            raise RuntimeError(
                f"Expected step_id or completion object from {self._make_url(path)}, "
                f"received: {data}"
            )

        logger.info(f"Polling task {step_id}")
        poll_path = f"{path.rstrip('/')}/{step_id}"
        return await self._poll_for_task(
            headers=headers,
            path=poll_path,
            expect=expect,
        )

    async def _poll_for_task(
        self,
        *,
        headers: Dict[str, str],
        path: str,
        expect: Literal["chat", "completion"],
    ) -> Dict[str, Any]:
        """
        Poll for task completion until success, failure, or timeout.

        Args:
            headers: HTTP headers for polling requests
            path: Polling endpoint path (includes step_id)
            expect: Expected response type

        Returns:
            Final completion result

        Raises:
            RuntimeError: If task fails or returns malformed data
            TimeoutError: If polling timeout is exceeded
        """
        deadline = (
            time.monotonic() + self._max_wait_s
            if self._max_wait_s is not None
            else None
        )

        max_polls = (
            int(self._max_wait_s / self._poll_interval_s) + 10
            if self._max_wait_s
            else 1000
        )
        poll_count = 0

        while poll_count < max_polls:
            poll_count += 1

            response = await self._request(
                method="GET",
                path=path,
                headers=headers,
            )

            if response.status_code == httpx.codes.ACCEPTED:
                data: Dict[str, Any] = {"status": "pending"}
            else:
                data = _parse_json_response(response)

            # Task finished inline — return directly
            if _is_expected_payload(data, expect):
                logger.info(f"Task completed after {poll_count} polls")
                return data

            status = data.get("status")

            if status == "completed":
                result = data.get("result")
                if not isinstance(result, dict):
                    raise RuntimeError(
                        f"Completed task missing result payload: {data}"
                    )
                if _is_expected_payload(result, expect):
                    logger.info(f"Task completed successfully after {poll_count} polls")
                    return result
                raise RuntimeError(
                    f"Completed task returned malformed result: {result}"
                )

            if status == "failed":
                error_msg = data.get("error") or "Task failed"
                logger.error(f"Task failed: {error_msg}")
                raise RuntimeError(error_msg)

            if status in {"pending", "processing", None}:
                if deadline is not None and time.monotonic() > deadline:
                    raise TimeoutError(
                        f"Polling timed out after {poll_count} attempts "
                        f"({self._max_wait_s}s)"
                    )
                await asyncio.sleep(self._poll_interval_s)
                continue

            raise RuntimeError(f"Unexpected status while polling task: {data}")

        raise TimeoutError(
            f"Exceeded maximum polling attempts ({max_polls})"
        )

    async def _request(
        self,
        *,
        method: str,
        path: str,
        json_payload: Dict[str, Any] | None = None,
        headers: Dict[str, str] | None = None,
    ) -> httpx.Response:
        """
        Execute HTTP request using the shared connection pool.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: API endpoint path
            json_payload: Optional JSON request body
            headers: Optional HTTP headers

        Returns:
            HTTP response

        Raises:
            RuntimeError: If request fails
        """
        url = self._make_url(path)
        headers = headers or self._build_headers()

        try:
            return await self._http_client.request(
                method=method,
                url=url,
                json=json_payload,
                headers=headers,
            )
        except httpx.HTTPError as exc:
            logger.error(f"Request to {url} failed: {exc}")
            raise RuntimeError(f"Request to {url} failed: {exc}") from exc

    async def close(self):
        """Close HTTP client and release connection pool resources."""
        logger.debug("Closing HTTP client")
        await self._http_client.aclose()

    async def __aenter__(self):
        """Context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit — ensures connection cleanup."""
        await self.close()



class _NovaAsyncChat(OpenAIAsyncChat):
    """Internal wrapper that routes chat.completions to the polling handler."""

    def __init__(self, parent: NovaAsyncOpenAIClient) -> None:
        self._parent = parent

    @property
    def completions(self) -> OpenAIAsyncChatCompletions:
        return self._parent._chat_completions


class _NovaChatCompletions(OpenAIAsyncChatCompletions):
    """Internal handler for chat completion requests with async polling."""

    def __init__(self, parent: NovaAsyncOpenAIClient) -> None:
        self._parent = parent

    async def create(self, *args: Any, **kwargs: Any):
        """
        Create a chat completion (OpenAI-compatible).

        Streaming is not supported.
        """
        if kwargs.get("stream", False):
            raise ValueError("stream=True not supported by NovaAsyncOpenAIClient")

        payload = dict(kwargs)
        model = payload.pop("model", None)
        messages = payload.pop("messages", None)

        if model is None:
            raise ValueError("model is required")
        if messages is None:
            raise ValueError("messages is required")

        tools = payload.pop("tools", None)
        body: Dict[str, Any] = {"model": model, "messages": messages, **payload}
        if tools is not None:
            body["tools"] = tools

        data = await self._parent._create_and_poll(
            path="/v1/chat/completions",
            payload=body,
            expect="chat",
        )
        _normalize_completion_payload(data, "chat")
        from openai.types.chat.chat_completion import ChatCompletion
        return ChatCompletion.model_validate(data)


class _NovaCompletions(OpenAIAsyncCompletions):
    """Internal handler for text completion requests with async polling."""

    def __init__(self, parent: NovaAsyncOpenAIClient) -> None:
        self._parent = parent

    async def create(self, *args: Any, **kwargs: Any):
        """
        Create a text completion (OpenAI-compatible).

        Streaming is not supported.
        """
        if kwargs.get("stream", False):
            raise ValueError("stream=True not supported by NovaAsyncOpenAIClient")

        payload = dict(kwargs)
        model = payload.pop("model", None)
        prompt = payload.pop("prompt", None)

        if model is None:
            raise ValueError("model is required")
        if prompt is None:
            raise ValueError("prompt is required")

        body: Dict[str, Any] = {"model": model, "prompt": prompt, **payload}

        data = await self._parent._create_and_poll(
            path="/v1/completions",
            payload=body,
            expect="completion",
        )
        from openai.types.completion import Completion
        return Completion.model_validate(data)


def _normalize_completion_payload(
    data: Dict[str, Any],
    kind: Literal["chat", "completion"],
) -> Dict[str, Any]:
    """
    Add missing required fields so the response validates as an OpenAI object.

    Mutates *data* in-place and returns it for convenience.
    """
    if "id" not in data:
        data["id"] = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    if "created" not in data:
        data["created"] = int(time.time())
    if "object" not in data:
        data["object"] = "chat.completion" if kind == "chat" else "text_completion"
    return data


def _parse_json_response(response: httpx.Response) -> Dict[str, Any]:
    """
    Validate HTTP status, parse JSON body, and confirm it's a dict.

    Raises RuntimeError with error details on failure.
    """
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        try:
            error_data = response.json()
            error_detail = error_data.get("error", {})
            if isinstance(error_detail, dict):
                error_msg = error_detail.get("message", response.text)
            else:
                error_msg = str(error_detail) or response.text
        except Exception:
            error_msg = response.text
        raise RuntimeError(
            f"Request failed with status {response.status_code}: {error_msg}"
        ) from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Expected JSON response, received invalid payload: {response.text}"
        ) from exc

    if not isinstance(data, dict):
        raise RuntimeError(f"Expected JSON object, received: {type(data).__name__}")
    return data


def _is_expected_payload(
    data: Dict[str, Any],
    kind: Literal["chat", "completion"],
) -> bool:
    """
    Return True if *data* looks like a finished OpenAI completion of the given kind.

    Checks the ``object`` field first; falls back to structural inspection
    when the field is absent.
    """
    obj = data.get("object")

    if obj is not None:
        if kind == "chat":
            return obj in {"chat.completion", "chat.completion.chunk"}
        return obj in {"text_completion", "completion"}

    # Fallback: inspect choices structure
    choices = data.get("choices", [])
    if not choices:
        return False
    first_choice = choices[0]
    if kind == "chat":
        return "message" in first_choice or "delta" in first_choice
    return "text" in first_choice
