"""Control-plane web app: start runs, stream events (SSE), answer approval/clarification requests."""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from agent.config import PLAYBOOK_PATH, RUNS_DIR, WORKSPACE, WORLD_URL, Settings, admin_headers, reset_workspace
from agent.core import Agent
from agent.human import WebHuman
from agent.llm import LLMClient, LLMError
from agent.memory import Playbook
from agent.vault import Vault

app = FastAPI(title="AutoWork")
STATIC = Path(__file__).parent / "static"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/files", StaticFiles(directory=RUNS_DIR), name="files")

# ---------------------------------------------------------------- control-plane protection
# Threat: any website open in the user's browser can send requests to localhost:8000 (CSRF), and a
# DNS-rebinding page can even read responses. Defences:
#   1. Host header must be localhost/127.0.0.1           -> defeats DNS rebinding
#   2. mutating /api calls need a per-process secret token -> defeats CSRF (other sites can't read it)
#   3. if a browser sends an Origin, it must be ours       -> second CSRF layer
# No CORS middleware is installed, so browsers never grant other origins read access.
PORT = int(os.environ.get("AUTOWORK_PORT", "8000"))
CONTROL_TOKEN = os.environ.get("AUTOWORK_CONTROL_TOKEN") or secrets.token_urlsafe(24)
ALLOWED_ORIGINS = {f"http://localhost:{PORT}", f"http://127.0.0.1:{PORT}"}
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"])


@app.middleware("http")
async def require_token_for_mutations(request: Request, call_next):
    if request.method not in ("GET", "HEAD", "OPTIONS") and request.url.path.startswith("/api/"):
        origin = request.headers.get("origin")
        if origin is not None and origin not in ALLOWED_ORIGINS:
            return JSONResponse({"detail": "cross-origin request rejected"}, status_code=403)
        if not secrets.compare_digest(request.headers.get("x-autowork-token", ""), CONTROL_TOKEN):
            return JSONResponse({"detail": "missing or invalid X-AutoWork-Token"}, status_code=403)
    return await call_next(request)


class Run:
    def __init__(self):
        self.events: list[dict] = []
        self.id: str | None = None
        self.done = False
        self.human = WebHuman(self.emit)
        self.lock = threading.Lock()

    def emit(self, kind: str, data: dict):
        # Persistence + redaction happen in Agent.emit; this is only the in-memory feed for SSE.
        with self.lock:
            self.events.append({"seq": len(self.events), "t": round(time.time(), 2), "type": kind, "data": data})


RUNS: dict[str, Run] = {}


class Faults(BaseModel):
    acme_login_flaky: bool = False
    erp_submit_timeout: bool = False
    erp_session_expiry: int = Field(0, ge=0, le=100)


class StartReq(BaseModel):
    task: str = Field(min_length=1, max_length=4000)
    mode: Literal["autonomous", "balanced", "supervised"] = "balanced"
    max_steps: int = Field(40, ge=1, le=100)
    use_playbook: bool = True
    reset_world: bool = False
    faults: Faults | None = None


class AnswerReq(BaseModel):
    qid: str = Field(max_length=64)
    approved: bool | None = None
    comment: str = ""
    answer: str = ""


@app.get("/", response_class=HTMLResponse)
def index():
    """The token is embedded in the page; other origins cannot read this response (no CORS)."""
    html = (STATIC / "index.html").read_text().replace("__AUTOWORK_TOKEN__", CONTROL_TOKEN)
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


@app.post("/api/runs")
def start_run(req: StartReq):
    if any(not r.done for r in RUNS.values()):
        raise HTTPException(409, "A run is already in progress")
    faults = req.faults.model_dump() if req.faults else None
    if req.reset_world or faults:
        httpx.post(f"{WORLD_URL}/admin/reset", json=faults, headers=admin_headers(), timeout=10).raise_for_status()
        reset_workspace()
    if not WORKSPACE.exists():
        reset_workspace()
    settings = Settings.from_env()
    try:
        llm = LLMClient(settings)
    except LLMError as e:
        raise HTTPException(400, str(e)) from e
    run = Run()
    agent = Agent(
        llm,
        run.human,
        WORKSPACE,
        RUNS_DIR,
        Playbook(PLAYBOOK_PATH) if req.use_playbook else None,
        emit=run.emit,
        mode=req.mode,
        max_steps=req.max_steps,
        vault=Vault.load(),
        max_tokens_total=settings.max_tokens_total,
        max_active_seconds=settings.max_active_seconds,
    )
    (RUNS_DIR / agent.run_id).mkdir(parents=True, exist_ok=True)
    run.id = agent.run_id
    RUNS[run.id] = run

    def work():
        try:
            agent.run(req.task)
        finally:
            run.done = True

    threading.Thread(target=work, daemon=True, name=f"run-{run.id}").start()
    return {"run_id": run.id}


@app.get("/api/runs")
def list_runs():
    out = []
    for d in sorted(RUNS_DIR.iterdir(), reverse=True)[:30]:
        rep = d / "report.json"
        if rep.exists():
            r = json.loads(rep.read_text())
            out.append({"run_id": d.name, "task": r["task"], "status": r["status"], "steps": r["steps"]})
        elif d.name in RUNS:
            out.append({"run_id": d.name, "task": "", "status": "running", "steps": 0})
    return out


RUN_ID = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{4}$")


@app.get("/api/runs/{rid}/events")
async def events(rid: str):
    if not RUN_ID.match(rid):
        raise HTTPException(404)
    run = RUNS.get(rid)
    if run is None:
        f = RUNS_DIR / rid / "events.jsonl"
        if not f.exists():
            raise HTTPException(404)
        past = [json.loads(line) for line in f.read_text().splitlines()]

        async def replay():
            for e in past:
                yield f"data: {json.dumps(e, default=str)}\n\n"

        return StreamingResponse(replay(), media_type="text/event-stream")

    async def stream():
        i = 0
        while True:
            while i < len(run.events):
                yield f"data: {json.dumps(run.events[i], default=str)}\n\n"
                i += 1
            if run.done and i >= len(run.events):
                break
            await asyncio.sleep(0.25)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/runs/{rid}/answer")
def answer(rid: str, req: AnswerReq):
    run = RUNS.get(rid)
    if not run:
        raise HTTPException(404)
    payload = (
        {"approved": bool(req.approved), "comment": req.comment} if req.approved is not None else {"answer": req.answer}
    )
    if not run.human.respond(req.qid, payload):
        raise HTTPException(404, "No such pending question")
    return {"ok": True}


@app.get("/api/playbook")
def playbook():
    return Playbook(PLAYBOOK_PATH).load()


@app.delete("/api/playbook")
def clear_playbook():
    PLAYBOOK_PATH.unlink(missing_ok=True)
    return {"ok": True}


@app.post("/api/world/reset")
def world_reset(faults: Faults | None = None):
    r = httpx.post(
        f"{WORLD_URL}/admin/reset", json=faults.model_dump() if faults else None, headers=admin_headers(), timeout=10
    )
    reset_workspace()
    return r.json()
