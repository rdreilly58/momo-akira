"""CLI entry point for momo-akira."""

from __future__ import annotations

import click
import uvicorn


@click.command()
@click.option("--config", "-c", default="config.yaml", show_default=True, help="Path to config.yaml")
@click.option("--host", default=None, help="Override server host")
@click.option("--port", "-p", default=None, type=int, help="Override server port")
@click.option("--reload", is_flag=True, default=False, help="Enable auto-reload (dev)")
@click.option("--log-level", default="info", show_default=True)
def main(
    config: str,
    host: str | None,
    port: int | None,
    reload: bool,
    log_level: str,
) -> None:
    """Start the momo-akira PyramidSD inference server."""
    from momo_akira.config import Config
    from momo_akira.server import create_app

    cfg = Config.from_yaml(config)
    app = create_app(cfg)

    uvicorn.run(
        app,
        host=host or cfg.server.host,
        port=port or cfg.server.port,
        reload=reload,
        log_level=log_level,
    )
