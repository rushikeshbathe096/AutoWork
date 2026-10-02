"""Start everything: the simulated company world (:8001) and the AutoWork control UI (:8000)."""
import logging
import threading

import uvicorn

from agent.config import WORKSPACE, reset_workspace


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if not WORKSPACE.exists():
        reset_workspace()
    world = uvicorn.Server(uvicorn.Config("simworld.app:app", host="127.0.0.1", port=8001, log_level="warning"))
    threading.Thread(target=world.run, daemon=True).start()
    print("Simulated company world: http://localhost:8001   (mail, Acme portal, ERP)")
    print("AutoWork UI:             http://localhost:8000")
    uvicorn.run("server.app:app", host="127.0.0.1", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
