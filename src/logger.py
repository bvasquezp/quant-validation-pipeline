"""
Logging estructurado con structlog.

JSON por defecto -- no print(). Necesario para depurar
por que un trial se descarto o por que el gate rechazo una version.
"""

from __future__ import annotations

import logging
import sys

import structlog


_configured = False


def setup_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Configura structlog con salida JSON estructurada.

    Llamar una vez al inicio del pipeline.
    """
    global _configured
    if _configured:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Procesadores comunes
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if fmt == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(numeric_level)

    _configured = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Obtiene un logger estructurado con el nombre dado."""
    setup_logging()
    return structlog.get_logger(name)
