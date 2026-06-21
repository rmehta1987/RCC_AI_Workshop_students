"""LLM backends for the Day-1 Module-3b CXR agent.

Two interchangeable backends behind one `decide(question, transcript, tools)` API:

* MockLLM         — deterministic, offline, NO key. Routes the 3-part CXR question
                    to the right tools in a valid order and composes an auditable
                    answer (incl. cross-tool conflict detection). This is what makes
                    Mastery Gate 3b pass on a compute node with no internet/keys.
* OpenAICompatLLM — talks to any OpenAI-compatible endpoint: OpenAI, a local
                    **vLLM** server (the cluster's `marshal_env_torch260_vllm084.sif`
                    serving e.g. Qwen), or Ollama. Set OPENAI_BASE_URL + OPENAI_API_KEY.

`get_llm()` auto-selects: OpenAI-compatible if a key/endpoint is configured, else Mock.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List


# --------------------------------------------------------------------------- #
class MockLLM:
    """A rule-based stand-in for an LLM controller. Deterministic by design."""

    name = "mock"

    def decide(self, question: str, transcript: List[dict], tools: Dict[str, str]) -> dict:
        q = (question or "").lower()
        plan: List[str] = []
        if any(k in q for k in ("pneumonia", "classif", "diagnos", "is there",
                                "detect", "abnormal", "finding", "opacity")):
            plan.append("classify_cxr")
        if any(k in q for k in ("where", "region", "location", "lobe", "localize",
                                "localise", "which part", "side")):
            plan.append("gradcam_region")
        if any(k in q for k in ("risk", "chart", "clinical", "high-risk", "high risk",
                                "patient", "history", "tabular", "comorbid")):
            plan.append("tabular_risk")
        if not plan:
            plan = ["classify_cxr"]
        seen = set()
        plan = [t for t in plan if not (t in seen or seen.add(t))]

        called = [step["call"].get("tool")
                  for step in transcript if isinstance(step.get("call"), dict)
                  and "tool" in step["call"]]
        for t in plan:
            if t not in called and t in tools:
                return {"tool": t, "args": {}}
        return {"answer": self._compose(question, transcript)}

    @staticmethod
    def _compose(question: str, transcript: List[dict]) -> str:
        res = {step["call"]["tool"]: step.get("result", {})
               for step in transcript if isinstance(step.get("call"), dict)
               and "tool" in step["call"]}
        parts: List[str] = []
        p = res.get("classify_cxr", {}).get("pneumonia_prob")
        if p is not None:
            parts.append(f"Pneumonia probability {p:.2f} "
                         f"({'likely present' if p >= 0.5 else 'not clearly present'}).")
        hot = res.get("gradcam_region", {}).get("hotspot")
        if hot is not None:
            parts.append(f"Grad-CAM localizes activation to: {hot}.")
        cr = res.get("tabular_risk", {}).get("clinical_risk")
        if cr is not None:
            parts.append(f"Tabular clinical risk {cr:.2f}.")
        # The safeguard the stretch task probes: never silently average a conflict.
        if (p is not None and hot not in (None, "none", "None")
                and p < 0.5):
            parts.append("CONFLICT: classifier is negative but Grad-CAM shows a focal "
                         "hotspot - escalate for human review rather than averaging.")
        return " ".join(parts) if parts else "Insufficient tool evidence to answer."


# --------------------------------------------------------------------------- #
_SYSTEM = """You are a careful radiology triage controller that orchestrates tools.
RULES:
- Answer ONLY after calling EVERY tool relevant to the question. A multi-part question
  ("is there pneumonia, WHERE, and is the patient high-risk given their chart?") requires
  MULTIPLE tool calls - do not stop early.
- If the question mentions the patient's chart, history, or risk, you MUST call
  tabular_risk: it RETURNS the chart-based clinical risk. Never say the chart data is
  missing - call the tool to get it.
- Only cite findings a tool actually returned. If two tools disagree, say so explicitly
  rather than averaging.
- Each turn reply with EXACTLY ONE JSON object and nothing else:
    {"tool": "<tool_name>", "args": {}}        to call a tool, or
    {"answer": "<concise grounded answer>"}    once you have called all relevant tools.
Available tools:
%s
"""


class OpenAICompatLLM:
    """OpenAI-compatible chat backend (OpenAI / vLLM / Ollama)."""

    name = "openai-compat"

    def __init__(self, model: str = None, base_url: str = None,
                 api_key: str = None, temperature: float = 0.0, max_retries: int = 2):
        from openai import OpenAI  # lazy; only needed when actually used
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        self.model = model or os.environ.get("AIMED_LLM_MODEL") or "gpt-4o-mini"
        self.temperature = temperature
        self.max_retries = max_retries
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self._fallback = MockLLM()

    def decide(self, question: str, transcript: List[dict], tools: Dict[str, str]) -> dict:
        tool_desc = "\n".join(f"- {n}: {d}" for n, d in tools.items())
        messages = [
            {"role": "system", "content": _SYSTEM % tool_desc},
            {"role": "user", "content": f"Question: {question}\n"
                                        f"Evidence so far (JSON): {json.dumps(transcript)[:4000]}\n"
                                        f"Reply with exactly one JSON object and nothing else. /no_think"},
        ]
        for _ in range(self.max_retries + 1):
            try:
                r = self.client.chat.completions.create(
                    model=self.model, messages=messages,
                    temperature=self.temperature, max_tokens=300)
                txt = r.choices[0].message.content
                obj = _extract_json(txt)
                if obj and ("tool" in obj or "answer" in obj):
                    return obj
            except Exception:
                break
        # Robust fallback so a flaky endpoint never breaks a class.
        return self._fallback.decide(question, transcript, tools)


def _extract_json(text: str):
    """Robustly pull the {tool|answer} JSON out of an LLM reply, tolerating
    reasoning models (Qwen3 / R1 <think> blocks) and ```json fences."""
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)  # drop reasoning
    text = re.sub(r"```(?:json)?|```", "", text)                      # drop code fences
    spans = []
    if "{" in text and "}" in text:
        spans.append(text[text.find("{"): text.rfind("}") + 1])      # full first..last span (handles nesting)
    spans += re.findall(r"\{.*?\}", text, re.DOTALL)                  # then minimal candidates
    for s in spans:
        try:
            obj = json.loads(s)
            if isinstance(obj, dict) and ("tool" in obj or "answer" in obj):
                return obj
        except Exception:
            continue
    return None


# --------------------------------------------------------------------------- #
def get_llm(backend: str = "auto", model: str = None):
    """Pick a backend. 'auto' uses an OpenAI-compatible endpoint when one is
    configured (OPENAI_API_KEY or OPENAI_BASE_URL), else the offline MockLLM."""
    if backend == "mock":
        return MockLLM()
    have_endpoint = bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_BASE_URL"))
    if backend == "openai" or (backend == "auto" and have_endpoint):
        try:
            return OpenAICompatLLM(model=model)
        except Exception as e:
            print(f"[llm] OpenAI-compatible backend unavailable ({e}); using MockLLM.")
            return MockLLM()
    return MockLLM()
