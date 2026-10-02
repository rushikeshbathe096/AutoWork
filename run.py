"""Start everything: the simulated company world (:8001) and the AutoWork control UI (:8000).
Both bind to 127.0.0.1 only."""
import logging
import threading

import uvicorn

from agent.config import WORKSPACE, reset_workspace


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if not WORKSPACE.exists():
        reset_workspace()
    import simworld.app as world_app  # imported first: it publishes SIMWORLD_ADMIN_TOKEN for this process
    from server.app import CONTROL_TOKEN, app

    world = uvicorn.Server(uvicorn.Config(world_app.app, host="127.0.0.1", port=8001, log_level="warning"))
    threading.Thread(target=world.run, daemon=True).start()
    print("Simulated company world: http://localhost:8001   (mail, Acme portal, ERP)")
    print("AutoWork UI:             http://localhost:8000")
    print(f"Control-plane token (only needed for scripted API calls, header X-AutoWork-Token): {CONTROL_TOKEN}")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
