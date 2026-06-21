#!/usr/bin/env python
"""End-to-end validation (Goal 11). Run on a COMPUTE node via slurm/validate_*.sbatch.

Proves the whole stack works where students actually run it, then writes a
VALIDATION_REPORT.md (pass/fail per component, versions, GPU, timings). Each check
is isolated — one failure never aborts the rest — and degraded paths (CPU,
precomputed table, mock LLM) are exercised explicitly so Goal 13 is covered too.
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
import traceback
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from common import aimed_config as cfg            # noqa: E402

RESULTS = []


def check(name):
    """Decorator: run a check, capture PASS/FAIL/SKIP + timing + detail."""
    def deco(fn):
        def wrapped():
            t0 = time.time()
            try:
                detail = fn()
                status = "SKIP" if (isinstance(detail, str) and detail.startswith("SKIP")) else "PASS"
            except AssertionError as e:
                status, detail = "FAIL", f"assertion: {e}"
            except Exception as e:
                status, detail = "FAIL", f"{type(e).__name__}: {e}"
                if "-v" in sys.argv:
                    traceback.print_exc()
            dt = time.time() - t0
            RESULTS.append((name, status, str(detail), dt))
            print(f"[{status:4s}] {name:34s} {dt:6.1f}s  {detail}")
            return status
        return wrapped
    return deco


DEVICE = "auto"


# --------------------------------------------------------------------------- #
@check("imports (core stack)")
def c_imports():
    import torch, transformers, sklearn, numpy, pandas  # noqa
    import medmnist, gradio  # noqa
    import pytorch_grad_cam  # noqa
    import Bio, pyfaidx  # noqa
    return (f"torch {torch.__version__} | transformers {transformers.__version__} "
            f"| sklearn {sklearn.__version__}")


@check("gpu visibility")
def c_gpu():
    import torch
    if torch.cuda.is_available():
        return f"cuda: {torch.cuda.get_device_name(0)} (cuda build {torch.version.cuda})"
    return "SKIP: no GPU visible — validating CPU fallback path"


@check("data integrity (bundle)")
def c_data():
    import pandas as pd, gzip
    df = pd.read_csv(cfg.BRCA1_CSV)
    assert len(df) == 3893, f"expected 3893 BRCA1 rows, got {len(df)}"
    assert {"pos_hg19", "reference", "alt", "is_lof"} <= set(df.columns)
    with gzip.open(cfg.CHR17_FASTA, "rt") as fh:
        hdr = fh.readline()
    assert "GRCh37" in hdr, f"chr17 not GRCh37: {hdr[:60]}"
    assert cfg.DIABETES_CSV.exists() and cfg.CXR_SMOKE.exists()
    return f"BRCA1 3893 rows; chr17 GRCh37; diabetes+fixtures present"


@check("unit tests (scoring + metrics)")
def c_unit():
    import subprocess
    r = subprocess.run([sys.executable, "-m", "pytest", str(SCRIPTS / "tests"), "-q"],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"pytest failed:\n{r.stdout[-800:]}"
    return r.stdout.strip().splitlines()[-1]


@check("Day1 M1 clinical_risk_console")
def c_clinical():
    import clinical_risk_console as CC
    st = CC.run_report(log=lambda *a: None)
    g = st["gate1"]
    assert g["sensitivity"] >= 0.90 - 1e-9
    return f"thr@90sens={g['threshold_at_90_sensitivity']:.3f} AUPRC={g['auprc']:.3f}"


@check("Day1 M2 radiograph_reader (smoke)")
def c_reader():
    import radiograph_reader as RR
    X, y = RR.load_pneumonia(smoke=True)
    m = RR.train(RR.build_model(pretrained=False), X, y, epochs=1, device="cpu",
                 log=lambda *a: None)
    auroc = RR.evaluate(m, X, y)
    region = RR.locate_peak_region(RR.gradcam(m, X[0]))
    assert "lobe" in region or "zone" in region
    return f"smoke AUROC={auroc:.3f} gradcam_region='{region}'"


@check("Day1 M3a late_fusion")
def c_fusion():
    import late_fusion as LF
    st = LF.run_report(log=lambda *a: None)
    a = st["aurocs"]
    assert all(0.4 < v < 1.0 for v in a.values())
    return (f"img={a['image_only']:.3f} tab={a['tabular_only']:.3f} "
            f"fused={a['fused']:.3f} miss@flag={st['missing']['flag']:.3f}")


@check("Day1 M3b minimal_cxr_agent (mock LLM)")
def c_agent():
    import minimal_cxr_agent as AG
    from common import llm_backends
    tools, img, feats = AG.build_stub_tools()
    ans, tr = AG.run_agent(AG.QUESTION, img, feats, tools, llm_backends.MockLLM())
    g = AG.gate3b_check(tr)
    assert g["passed"], f"gate3b not passed: {g}"
    return f"tools={g['called']} answer[:40]={ans[:40]!r}"


@check("Day1 M3b medrax dissection fallback")
def c_medrax():
    import medrax_launcher as MR
    res = MR.dissect(MR.EXAMPLE_TRAJECTORY)
    assert res["n_failures"] == 2
    return f"failures flagged: {[f['failure'] for f in res['findings']]}"


@check("Day2 M1 HyenaDNA generate + frameshift")
def c_playground():
    from common import models
    import dna_lm_playground as PG
    scorer = models.load_dna_model("hyenadna-small-32k", device=DEVICE)
    fs = PG.frameshift_demo(scorer)
    assert fs["frameshift_lower"], "frameshift did not score lower"
    gen = PG.generate_demo(scorer, "ATGGCG", n=16)
    return f"LL_ref={fs['ll_ref']:.1f} LL_fs={fs['ll_frameshift']:.1f} gen[:12]={gen[:12]!r}"


@check("Day2 M2 vep_scorer (method runs)")
def c_vep_model():
    # Mastery is the METHOD; HyenaDNA-small is ~chance on BRCA1 (a real finding), so we
    # assert the pipeline runs + yields a valid AUROC, not that it is high.
    from vep_scorer import run_vep
    df, auroc, info = run_vep(model_key="hyenadna-small-32k", n=60,
                              window=1024, device=DEVICE, log=lambda *a: None)
    assert 0.0 <= auroc <= 1.0, f"AUROC {auroc} is not a valid value"
    return f"n=60 HyenaDNA AUROC={auroc:.3f} (~chance expected) ({info['source']})"


@check("Day2 Evo 2 headline table (if present)")
def c_vep_evo2():
    from vep_scorer import evo2_auroc
    a = evo2_auroc()
    if a is None:
        return "SKIP: Evo 2 precomputed table not generated (HyenaDNA path is the default)"
    assert a > 0.65, f"Evo 2 table AUROC {a:.3f} below the expected ~0.7"
    return f"Evo 2 BRCA1 AUROC={a:.3f} (scale matters)"


@check("Day2 fallback: precomputed VEP table")
def c_vep_precomputed():
    from vep_scorer import run_vep
    df, auroc, info = run_vep(n=200, use_precomputed=True, log=lambda *a: None)
    assert auroc == auroc  # not NaN
    return f"n={len(df)} AUROC={auroc:.3f} ({info['source']})"


@check("Day2 M3 brca1_triage VUS table")
def c_triage():
    import brca1_triage as BT
    from vep_scorer import run_vep
    df, auroc, info = run_vep(n=300, use_precomputed=True, log=lambda *a: None)
    vus = BT.vus_triage(df, top=5)
    return f"AUROC={auroc:.3f}; VUS table rows={len(vus)}"


@check("Evo 2 tier (optional)")
def c_evo2():
    if cfg.EVO2_TIER != "enabled":
        return "SKIP: Evo 2 tier disabled"
    try:
        import evo2  # noqa
    except Exception as e:
        return f"SKIP: evo2 not built ({type(e).__name__}); precomputed table covers it"
    from common import models
    s = models.load_dna_model("evo2_7b", device="cuda")
    v = s.score_sequences(["ACGTACGTACGT", "ACGTACGAACGT"])
    return f"evo2 scored 2 seqs: {v}"


def run_notebooks(report_dir: Path):
    """Execute each notebook headless and confirm it runs (gates embedded)."""
    nbdir = cfg.PROJECT_DIR / "notebooks"
    nbs = sorted(nbdir.glob("*-solution.ipynb")) or sorted(nbdir.glob("*.ipynb"))
    if not nbs:
        RESULTS.append(("notebooks", "SKIP", "no notebooks found", 0.0))
        print("[SKIP] notebooks: none found")
        return
    import subprocess, tempfile, shutil
    # executed copies go to a private temp dir (NOT world-readable /tmp), cleaned up;
    # avoids leaking any key-bearing intermediate artifacts on shared HPC storage.
    _tmp = tempfile.mkdtemp(prefix="aimed_nbval_")
    try:
        for nb in nbs:
            t0 = time.time()
            r = subprocess.run(
                [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
                 "--execute", "--ExecutePreprocessor.timeout=1200",
                 "--output", f"{nb.stem}_executed.ipynb", "--output-dir", _tmp, str(nb)],
                capture_output=True, text=True)
            status = "PASS" if r.returncode == 0 else "FAIL"
            detail = "executed (gates ran)" if r.returncode == 0 else r.stderr[-300:]
            RESULTS.append((f"notebook {nb.name}", status, detail, time.time() - t0))
            print(f"[{status:4s}] notebook {nb.name:24s} {time.time()-t0:6.1f}s")
    finally:
        shutil.rmtree(_tmp, ignore_errors=True)


def write_report(path: Path, device: str):
    import datetime
    npass = sum(1 for _, s, _, _ in RESULTS if s == "PASS")
    nfail = sum(1 for _, s, _, _ in RESULTS if s == "FAIL")
    nskip = sum(1 for _, s, _, _ in RESULTS if s == "SKIP")
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
    except Exception:
        gpu = "unknown"
    lines = [
        f"# VALIDATION_REPORT — AImed course",
        f"",
        f"- generated: {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"- host: {platform.node()}   device requested: {device}   gpu: {gpu}",
        f"- python: {sys.version.split()[0]}   env: {cfg.ENV_PREFIX}",
        f"- **{npass} PASS / {nfail} FAIL / {nskip} SKIP**",
        f"",
        f"| component | status | seconds | detail |",
        f"|---|---|---|---|",
    ]
    for name, status, detail, dt in RESULTS:
        d = detail.replace("|", "\\|").replace("\n", " ")[:140]
        lines.append(f"| {name} | **{status}** | {dt:.1f} | {d} |")
    lines += ["", f"_Result: {'ALL GREEN' if nfail == 0 else str(nfail)+' FAILURES — see above'}._"]
    path.write_text("\n".join(lines))
    print(f"\nWrote {path}  ({npass} PASS / {nfail} FAIL / {nskip} SKIP)")
    return nfail


def main():
    global DEVICE
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--report", default="VALIDATION_REPORT.md")
    ap.add_argument("--skip-notebooks", action="store_true")
    args = ap.parse_args()
    DEVICE = args.device
    cfg.setup_caches()
    print(f"== AImed validation (device={DEVICE}) ==")

    for fn in [c_imports, c_gpu, c_data, c_unit, c_clinical, c_reader, c_fusion,
               c_agent, c_medrax, c_playground, c_vep_model, c_vep_evo2,
               c_vep_precomputed, c_triage, c_evo2]:
        fn()
    if not args.skip_notebooks:
        run_notebooks(cfg.PROJECT_DIR / "notebooks")

    report = cfg.PROJECT_DIR / args.report
    nfail = write_report(report, DEVICE)
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
