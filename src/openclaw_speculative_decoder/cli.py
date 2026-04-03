"""CLI entry point for the proxy server."""

from __future__ import annotations

import click
import uvicorn


@click.command()
@click.option("--config", "-c", default=None, help="Path to config.yaml")
@click.option("--host", default=None, help="Override server host")
@click.option("--port", "-p", default=None, type=int, help="Override server port")
@click.option("--log-level", default=None, help="Override log level")
@click.option("--reload", is_flag=True, default=False, help="Enable auto-reload (dev mode)")
def main(
    config: str | None,
    host: str | None,
    port: int | None,
    log_level: str | None,
    reload: bool,
) -> None:
    """Start the openclaw speculative decoder proxy."""
    import os

    if config:
        os.environ["OPENCLAW_CONFIG"] = config

    # Load config to get defaults (then override with CLI args)
    from .config import load_config
    cfg = load_config(config)

    server_host = host or cfg.server.host
    server_port = port or cfg.server.port
    server_log_level = log_level or cfg.server.log_level

    click.echo(f"Starting openclaw-speculative-decoder on {server_host}:{server_port}")
    click.echo(f"Cascade variant: {cfg.cascade.variant}")
    click.echo(f"  Draft:     {cfg.models.draft.model_id} ({cfg.models.draft.provider})")
    click.echo(f"  Qualifier: {cfg.models.qualifier.model_id} ({cfg.models.qualifier.provider})")
    click.echo(f"  Target:    {cfg.models.target.model_id} ({cfg.models.target.provider})")
    click.echo(f"  tau_Q={cfg.cascade.tau_q}, tau_T={cfg.cascade.tau_t}")

    uvicorn.run(
        "openclaw_speculative_decoder.server:app",
        host=server_host,
        port=server_port,
        log_level=server_log_level,
        reload=reload,
    )


if __name__ == "__main__":
    main()
