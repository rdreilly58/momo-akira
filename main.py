"""Main entrypoint for running the momo-akira server."""
import uvicorn

from momo_akira.config import Config
from momo_akira.server import create_app

cfg = Config.from_yaml("config.yaml")
app = create_app(cfg)

if __name__ == "__main__":
    uvicorn.run(app, host=cfg.server.host, port=cfg.server.port)
