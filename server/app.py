"""Control-plane web app: start runs, stream events (SSE), answer approval/clarification requests."""
from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.config import PLAYBOOK_PATH, RUNS_DIR, WORKSPACE, WORLD_URL, reset_workspace
from agent.core import Agent
from agent.human import WebHuman
from agent.llm import LLMClient, LLMError
from agent.memory import Playbook

app = FastAPI(title="AutoWork")
STATIC = Path(__file__).parent / "static"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/files", StaticFiles(directory=RUNS_DIR), name="files")


class Run:
    def __init__(self):
        self.events: list[dict] = []
        self.id: str | None = None
        self.done = False
        self.human = WebHuman(self.emit)
        self.lock = threading.Lock()

    def emit(self, kind: str, data: dict):
        with self.lock:
            self.events.append({"seq": len(self.events), "t": round(time.time(), 2), "type": kind, "data": data})
        if self.id:
            with open(RUNS_DIR / self.id / "events.jsonl", "a") as f:
                f.write(json.dumps(self.events[-1], default=str) + "\n")


RUNS: dict[str, Run] = {}


class StartReq(BaseModel):
    task: str
    mode: str = "balanced"
    max_steps: int = 40
    use_playbook: bool = True
    reset_world: bool = False
    faults: dict = {}


class AnswerReq(BaseModel):
    qid: str
    approved: bool | None = None
    comment: str = ""
    answer: str = ""


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.post("/api/runs")
def start_run(req: StartReq):
    if any(not r.done for r in RUNS.values()):
        raise HTTPException(409, "A run is already in progress")
    if req.reset_world or req.faults:
        httpx.post(f"{WORLD_URL}/admin/reset", json=req.faults or None, timeout=10)
        reset_workspace()
    if not WORKSPACE.exists():
        reset_workspace()
    try:
        llm = LLMClient()
    except LLMError as e:
        raise HTTPException(400, str(e)) from e
    run = Run()
    agent = Agent(llm, run.human, WORKSPACE, RUNS_DIR, Playbook(PLAYBOOK_PATH) if req.use_playbook else None,
                  emit=run.emit, mode=req.mode, max_steps=req.max_steps)
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


@app.get("/api/runs/{rid}/events")
async def events(rid: str):
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
    payload = {"approved": bool(req.approved), "comment": req.comment} if req.approved is not None else \
        {"answer": req.answer}
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
def world_reset(faults: dict | None = None):
    r = httpx.post(f"{WORLD_URL}/admin/reset", json=faults, timeout=10)
    reset_workspace()
    return r.json()
