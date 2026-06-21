#!/usr/bin/env python
"""MedRAX launcher + trajectory-dissection fallback (Day-1 Module 3b, Part 2).

Mirror the day's pattern: build small (minimal_cxr_agent), then inspect the real
thing (MedRAX). MedRAX is an OPTIONAL tier (--with-medrax). This launcher:

  * if MedRAX is installed AND an LLM endpoint is configured -> initialize the
    agent with a SELECTIVE subset of lightweight tools and print how to run a
    ChestAgentBench case;
  * otherwise -> run the **trajectory-dissection fallback** the design allows:
    analyse an agent trajectory for the graded failure modes (hallucinated tool
    call, ignored cross-tool conflict) and name the safeguard. This is the
    no-GPU / no-key path and it still exercises Gate-3b's concept skill.

  python medrax_launcher.py                 # dissection fallback (always works)
  python medrax_launcher.py --launch        # try the real MedRAX agent
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from common import aimed_config as cfg            # noqa: E402

# Lightweight tools that map onto what students built (classifier + segmentation).
SELECTED_TOOLS = ["ImageVisualizerTool", "ChestXRayClassifierTool", "ChestXRaySegmentationTool"]

# A worked example trajectory with TWO planted failures (for the dissection drill).
EXAMPLE_TRAJECTORY = {
    "question": "Does this CXR show pneumonia, where, and is the patient high-risk?",
    "steps": [
        {"call": {"tool": "ChestXRayClassifierTool"}, "result": {"pneumonia_prob": 0.18}},
        # NOTE: the narrative below will CLAIM a segmentation call that never happened.
        {"call": {"tool": "ImageVisualizerTool"}, "result": {"ok": True}},
    ],
    "final_report": ("Segmentation confirms a dense right-lower-lobe consolidation "
                     "consistent with pneumonia; recommend antibiotics."),
}


def dissect(traj: dict) -> dict:
    """Flag the graded failure modes in an agent trajectory."""
    called = [s["call"]["tool"] for s in traj["steps"] if "tool" in s.get("call", {})]
    report = traj.get("final_report", "").lower()
    findings = []

    # 1) Hallucinated tool call: report cites a tool that was never invoked.
    if "segmentation" in report and not any("Segmentation" in c for c in called):
        findings.append({
            "failure": "hallucinated_tool_call",
            "evidence": "final report cites segmentation, but no segmentation tool was called",
            "safeguard": "verify every cited finding against the transcript (tool-call provenance check)",
        })

    # 2) Ignored cross-tool conflict: classifier says low prob but report says pneumonia.
    probs = [s["result"].get("pneumonia_prob") for s in traj["steps"]
             if isinstance(s.get("result"), dict) and "pneumonia_prob" in s["result"]]
    if probs and min(probs) < 0.5 and "pneumonia" in report and "consistent" in report:
        findings.append({
            "failure": "ignored_cross_tool_conflict",
            "evidence": f"classifier prob {min(probs):.2f} (<0.5) contradicts the 'pneumonia' conclusion",
            "safeguard": "explicit conflict resolution: surface disagreement, do not silently override/average",
        })
    return {"tools_called": called, "findings": findings, "n_failures": len(findings)}


def try_launch():
    base = os.environ.get("OPENAI_BASE_URL")
    key = os.environ.get("OPENAI_API_KEY")
    medrax_dir = cfg.PROJECT_DIR / "repos" / "MedRAX"
    print("[medrax] selective tools:", SELECTED_TOOLS)
    if not medrax_dir.exists():
        print(f"[medrax] MedRAX not cloned at {medrax_dir}.")
        print("[medrax] enable the tier:  bash data/scripts/download_datasets.sh --with-medrax")
        return False
    if not (base or key):
        print("[medrax] no LLM endpoint (OPENAI_BASE_URL/OPENAI_API_KEY). Point it at the")
        print("[medrax] cluster vLLM .sif or a hosted model. Falling back to dissection.")
        return False
    try:
        sys.path.insert(0, str(medrax_dir))
        from medrax.agent import initialize_agent  # type: ignore
        agent, tools = initialize_agent(
            str(medrax_dir / "medrax" / "docs" / "system_prompts.txt"),
            tools_to_use=SELECTED_TOOLS, model_dir=cfg.HF_HOME)
        print(f"[medrax] agent initialized with {len(tools)} tools. Run a ChestAgentBench case "
              f"by feeding an image path to agent.workflow(...).")
        return True
    except Exception as e:
        print(f"[medrax] launch failed ({e}); using dissection fallback.")
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--launch", action="store_true", help="attempt the real MedRAX agent")
    args = ap.parse_args()
    cfg.setup_caches()

    launched = try_launch() if args.launch else False
    if not launched:
        print("\n[medrax] === trajectory-dissection fallback ===")
        res = dissect(EXAMPLE_TRAJECTORY)
        print(f"[medrax] tools actually called: {res['tools_called']}")
        for f in res["findings"]:
            print(f"  ⚠️ {f['failure']}: {f['evidence']}")
            print(f"     safeguard -> {f['safeguard']}")
        assert res["n_failures"] == 2, "dissection should find both planted failures"
        print(f"[medrax] dissection OK — {res['n_failures']} failure modes identified.")


if __name__ == "__main__":
    main()
