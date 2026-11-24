"""Structured JSON logs and OpenTelemetry tracing.

Traces go to any OTLP endpoint (Jaeger, Grafana Tempo, an OTel collector) and, when Langfuse keys
are set, to Langfuse's OTLP ingestion endpoint as well. Both are optional: without them spans are
still created, so log lines keep their trace ids.
"""

from __future__ import annotations

import base64
import contextvars
import json
import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from app.core.config import Settings

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
principal_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("principal", default=None)

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with request, principal and trace ids attached."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if rid := request_id_var.get():
            payload["request_id"] = rid
        if who := principal_var.get():
            payload["principal"] = who
        ctx = trace.get_current_span().get_span_context()
        if ctx.is_valid:
            payload["trace_id"] = f"{ctx.trace_id:032x}"
            payload["span_id"] = f"{ctx.span_id:016x}"
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for noisy in ("httpx", "httpcore", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


_configured = False


def configure_tracing(settings: Settings) -> None:
    """Install a tracer provider once. Safe to call from the API process and from Celery workers."""
    global _configured
    if _configured:
        return
    provider = TracerProvider(
        resource=Resource.create({"service.name": settings.app_name, "deployment.environment": settings.environment})
    )
    exporters = []
    if settings.otel_exporter_otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        exporters.append(OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint.rstrip("/") + "/v1/traces"))
    if settings.langfuse_public_key and settings.langfuse_secret_key and settings.langfuse_host:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        token = base64.b64encode(
            f"{settings.langfuse_public_key}:{settings.langfuse_secret_key.get_secret_value()}".encode()
        ).decode()
        exporters.append(
            OTLPSpanExporter(
                endpoint=settings.langfuse_host.rstrip("/") + "/api/public/otel/v1/traces",
                headers={"Authorization": f"Basic {token}"},
            )
        )
    for exporter in exporters:
        provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _configured = True


tracer = trace.get_tracer("rag")


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[trace.Span]:
    """A traced block. Attributes that are not plain values are stored as JSON strings."""
    with tracer.start_as_current_span(name) as current:
        for key, value in attributes.items():
            plain = isinstance(value, (str, int, float, bool))
            current.set_attribute(key, value if plain else json.dumps(value, default=str))
        yield current


def record_llm_usage(current: trace.Span, model: str, usage: Any, latency_s: float) -> None:
    """GenAI semantic-convention attributes, which Langfuse reads for cost and token charts."""
    current.set_attribute("gen_ai.request.model", model)
    current.set_attribute("gen_ai.latency_s", round(latency_s, 3))
    if usage is not None:
        current.set_attribute("gen_ai.usage.input_tokens", getattr(usage, "prompt_tokens", 0) or 0)
        current.set_attribute("gen_ai.usage.output_tokens", getattr(usage, "completion_tokens", 0) or 0)
