# ADR 0002: Text DOM snapshots with numbered elements, not screenshots + coordinates

**Status:** accepted

## Context
The agent must perceive web pages and act on them precisely. Options: screenshots to a vision model with click coordinates, raw HTML, or a distilled text snapshot.

## Decision
`agent/browser.py` injects a script that tags every visible interactive element with `data-aw-id=N` and returns: URL, HTTP status, alert texts, a numbered element list (`[8] input "Amount (numbers only…)" value=""`) and trimmed page text. The model acts by number (`browser_click(8)`). Screenshots are still captured every step, but for humans and evidence, not for the model.

## Alternatives
- **Vision + coordinates**: works on any UI (canvas, PDFs), but is slower, costs more per step, needs a multimodal model, and misclicks are hard to detect.
- **Raw HTML**: exact, but far more tokens per page (markup, scripts, styles) and full of noise.

## Consequences
+ Cheap enough for fast open models on Groq; clicks are exact; failures ("element [8] does not exist") are explicit.
+ `browser_fill` can read values back from the DOM to verify the agent's own input.
− Blind to canvas-rendered UIs, image-only documents and CAPTCHAs (documented limitation; vision fallback is a "next" item).
− Element ids are only valid for the current page; the agent must use the latest observation.
