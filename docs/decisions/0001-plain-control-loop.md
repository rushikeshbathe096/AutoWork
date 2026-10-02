# ADR 0001: A plain control loop instead of an agent framework

**Status:** accepted

## Context
The assignment is graded on reliability, verification and on being able to explain every mechanism. Frameworks (LangChain/LangGraph, browser-use, AutoGen) provide a loop, but the interesting parts (gating, approval binding, stuck detection, context compression, verification) would sit behind their abstractions or have to be fought around.

## Decision
A hand-written loop in `agent/core.py`: plan → (LLM picks ONE tool → policy gate → execute → observe)* → finish → verify → learn. Collaborators are typed against small Protocols (`agent/interfaces.py`).

## Alternatives
- **LangGraph**: good state-machine model, but adds a dependency and a vocabulary for what is under 500 lines here (`core.py`).
- **browser-use**: strong browser agent, but its own loop and prompts make it hard to insert network-level gates and a separate read-only verifier.

## Consequences
+ Every reliability mechanism is visible, in one place, and unit-tested with a scripted LLM.
+ No framework lock-in; the only runtime deps are FastAPI, Playwright and the OpenAI SDK.
− We maintain the loop ourselves (retries, message formatting, parallel tool-call handling).
− No built-in persistence or checkpointing (see README "What I'd build next").
