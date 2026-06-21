#!/usr/bin/env python
"""Minimal CXR Agent (Day-1 Module 3b) — build the agent from scratch.

Demystify "agent": wrap the artifacts students already built (Module-1 tabular
model, Module-2 CNN + Grad-CAM) as callable tools, and write a tiny ReAct loop
where an LLM decides which tool to call. The transcript is the artifact — the
agentic equivalent of Grad-CAM.

The LLM backend is pluggable (common/llm_backends):
  * default  -> MockLLM (deterministic, offline, NO key) — makes Gate 3b pass
  * vLLM/OpenAI/Ollama -> set OPENAI_BASE_URL (+ OPENAI_API_KEY). e.g. point it
    at the cluster's vLLM `.sif` serving Qwen:  OPENAI_BASE_URL=http://NODE:8000/v1

  python minimal_cxr_agent.py --smoke                  # stub tools, mock LLM, <1s
  python minimal_cxr_agent.py --real-tools             # train the actual CNN+tabular
  python minimal_cxr_agent.py --broken-classifier      # stretch: inject a wrong tool
  OPENAI_BASE_URL=http://gpuNNNN:8000/v1 python minimal_cxr_agent.py --backend openai
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from common import aimed_config as cfg            # noqa: E402
from common import llm_backends                   # noqa: E402

QUESTION = ("Is there pneumonia, and if so where, and is this patient high-risk "
            "given their chart?")

TOOL_DESC = {
    "classify_cxr": "Return pneumonia probability for the chest X-ray.",
    "gradcam_region": "Return the coarse image region the model focused on.",
    "tabular_risk": "Return clinical risk from the patient's tabular chart.",
}


# --------------------------------------------------------------------------- #
# Tool builders
# --------------------------------------------------------------------------- #
def build_stub_tools(broken=False):
    """Cheap deterministic tools (no torch) — orchestration lesson in <1s."""
    z = np.load(cfg.CXR_SMOKE)
    imgs = z["images"].astype("float32") / 255.0
    labels = z["labels"]
    img = imgs[int(np.argmax(labels))]  # a positive example
    feats = np.array([62, 1, 31.0, 95, 180, 110, 45, 4, 4.9, 88], dtype=float)  # diabetes-like row

    def classify(image):
        if broken:
            return {"pneumonia_prob": 0.0}  # always "normal" (the injected fault)
        return {"pneumonia_prob": float(np.clip(image.mean() * 3.0, 0, 1))}

    def gradcam(image):
        a = np.asarray(image)[0] if np.asarray(image).ndim == 3 else np.asarray(image)
        r, c = np.unravel_index(int(np.argmax(a)), a.shape)
        side = "right" if c < a.shape[1] / 2 else "left"
        vert = "upper" if r < a.shape[0] / 3 else ("mid" if r < 2 * a.shape[0] / 3 else "lower")
        return {"hotspot": f"{side} {vert} lobe"}

    def tabular(features):
        x = np.asarray(features, dtype=float)
        risk = 1 / (1 + np.exp(-((x[2] - 28) * 0.1 + (x[0] - 50) * 0.02)))  # bmi+age driven
        return {"clinical_risk": float(risk)}

    tools = {
        "classify_cxr": {"fn": classify, "modality": "image", "desc": TOOL_DESC["classify_cxr"]},
        "gradcam_region": {"fn": gradcam, "modality": "image", "desc": TOOL_DESC["gradcam_region"]},
        "tabular_risk": {"fn": tabular, "modality": "tabular", "desc": TOOL_DESC["tabular_risk"]},
    }
    return tools, img, feats


def build_real_tools(broken=False, device="auto"):
    """The authentic tools: the Module-2 CNN + the Module-1 tabular model."""
    import radiograph_reader as RR
    import clinical_risk_console as CC

    Xtr, ytr = RR.load_pneumonia("train", smoke=True)
    cxr_model = RR.train(RR.build_model(pretrained=False), Xtr, ytr, epochs=1, device=device)
    img = Xtr[int(np.argmax(ytr))]

    df, X, y, groups, feats_names = CC.load_data()
    Xc_tr, Xc_te, yc_tr, yc_te, _, _ = CC.split(X, y, groups)
    tab_model = CC.train_models(Xc_tr, yc_tr)["histgb"]
    feats = X[0]

    def classify(image):
        if broken:
            return {"pneumonia_prob": 0.0}
        return RR.classify_cxr_tool(cxr_model, image)

    tools = {
        "classify_cxr": {"fn": classify, "modality": "image", "desc": TOOL_DESC["classify_cxr"]},
        "gradcam_region": {"fn": lambda im: RR.gradcam_region_tool(cxr_model, im),
                           "modality": "image", "desc": TOOL_DESC["gradcam_region"]},
        "tabular_risk": {"fn": lambda f: {"clinical_risk":
                         float(tab_model.predict_proba(np.asarray(f)[None])[0, 1])},
                         "modality": "tabular", "desc": TOOL_DESC["tabular_risk"]},
    }
    return tools, img, feats


# --------------------------------------------------------------------------- #
# The ReAct loop
# --------------------------------------------------------------------------- #
def run_agent(question, image, features, tools, llm, max_steps=6):
    transcript = []
    tool_specs = {n: t["desc"] for n, t in tools.items()}
    for _ in range(max_steps):
        step = llm.decide(question, transcript, tool_specs)
        if "answer" in step:
            return step["answer"], transcript
        name = step.get("tool")
        if name not in tools:
            transcript.append({"call": step, "result": {"error": f"unknown tool {name}"}})
            continue
        arg = image if tools[name]["modality"] == "image" else features
        try:
            result = tools[name]["fn"](arg)
        except Exception as e:  # a failing tool must not crash the agent
            result = {"error": str(e)}
        transcript.append({"call": step, "result": result})
    return "max steps reached", transcript


def tools_called(transcript):
    return [s["call"]["tool"] for s in transcript
            if isinstance(s.get("call"), dict) and "tool" in s["call"]]


def gate3b_check(transcript, required=("classify_cxr", "tabular_risk")):
    """Gate 3b: required tools were called, and classify precedes gradcam."""
    called = tools_called(transcript)
    have_required = all(t in called for t in required)
    order_ok = True
    if "classify_cxr" in called and "gradcam_region" in called:
        order_ok = called.index("classify_cxr") < called.index("gradcam_region")
    return {"called": called, "have_required": have_required, "order_ok": order_ok,
            "passed": bool(have_required and order_ok)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="auto", choices=["auto", "mock", "openai"])
    ap.add_argument("--real-tools", action="store_true",
                    help="train the actual CNN + tabular model (slower)")
    ap.add_argument("--broken-classifier", action="store_true",
                    help="stretch: inject a classifier that always says 'normal'")
    ap.add_argument("--question", default=QUESTION)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg.setup_caches()
    llm = llm_backends.get_llm(backend=args.backend)
    print(f"[agent] LLM backend = {llm.name}")

    if args.real_tools and not args.smoke:
        tools, img, feats = build_real_tools(broken=args.broken_classifier)
    else:
        tools, img, feats = build_stub_tools(broken=args.broken_classifier)

    answer, transcript = run_agent(args.question, img, feats, tools, llm)
    print(f"\n[agent] QUESTION: {args.question}")
    print("[agent] TRANSCRIPT (the auditable trace):")
    for i, step in enumerate(transcript):
        print(f"  step {i}: {json.dumps(step)}")
    print(f"\n[agent] ANSWER: {answer}")

    g = gate3b_check(transcript)
    print(f"\n[agent] Gate-3b: called={g['called']} "
          f"required={g['have_required']} order_ok={g['order_ok']} -> "
          f"{'PASS' if g['passed'] else 'RETRY'}")
    if args.broken_classifier:
        print("[agent] (broken-classifier stretch: note the CONFLICT flag in the answer)")


if __name__ == "__main__":
    main()
