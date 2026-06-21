# AI in Medicine & Foundational Models in Genomics — Student Materials

Public, runnable student materials for the two-day intensive course. Everything here
runs on a **laptop CPU** (no GPU required) or on the UChicago RCC **Midway3** cluster.

> Heavy assets are intentionally **not** in this repo: model weights download once on
> first use, and the genomics frontier results ship as small **precomputed score tables**
> the notebooks read — so you never run a 7B model yourself.

## Quickstart — run it locally (CPU-only laptop)

```bash
# HTTPS (no SSH key needed):
git clone https://github.com/rmehta1987/RCC_AI_Workshop_students.git
# …or SSH (if you have a key on your GitHub account):
git clone git@github.com:rmehta1987/RCC_AI_Workshop_students.git

cd RCC_AI_Workshop_students
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-cpu.txt
bash scripts/run_local.sh          # starts JupyterLab in the background, opens your browser
```

Open `notebooks/day1_m1_clinical_risk.ipynb` and pick the venv's **Python 3** kernel.
First run downloads **HyenaDNA-small** (Day 2); Day-1 radiology uses a bundled smoke
fixture unless you pre-fetch MedMNIST. Full details — including the **Midway3 cluster**
path (Open OnDemand / `sbatch`) — are in **[`docs/STUDENT_README.md`](docs/STUDENT_README.md)**.

## Module map

| Day 1 — AI in Medicine | Day 2 — Genomics |
|---|---|
| `day1_m1_clinical_risk` — tabular risk + calibration | `day2_m1_seqmodel` — DNA LM likelihoods |
| `day1_m2_radiology` — CNN confidence & abstention (OOD) | `day2_m2_vep` — zero-shot variant scoring |
| `day1_m3a_fusion` — late fusion | `day2_m3_brca1_triage` — BRCA1 end-to-end |
| `day1_m3b_agent` — minimal CXR agent | |

Each module ships as a fill-in-the-blank notebook **plus** a `-solution.ipynb` answer key.
Stuck on a TODO? Run the **CHECKPOINT** cell — it reveals the answer so you can keep moving.

## What's here

- `notebooks/` — 14 notebooks (7 student + 7 solutions)
- `scripts/` — the Python the notebooks import (`common/` config + model loaders, the
  per-module modules, `run_local.sh`)
- `data/` — the small, **non-PHI** data the notebooks read (a diabetes table, BRCA1
  variants, both precomputed VEP tables, seqmodeling FASTA, and offline smoke fixtures)
- `docs/` — onboarding (`STUDENT_README.md`), Jupyter launch (`JUPYTER.md`), optional
  keys (`SECRETS.md`)
- `slurm/` — Midway3 job scripts (Jupyter launch, CPU/GPU validation)
- `requirements-cpu.txt` — the CPU dependency set

## Data & licensing

All data is open / non-PHI by design (sklearn diabetes / Efron 2004; BRCA1 SNVs from
Findlay 2018). Sources and provenance: [`data/docs/PROVENANCE.md`](data/docs/PROVENANCE.md).
Materials are provided for educational use.

*No secrets are required — the Day-1 agent falls back to an offline mock LLM, and the
genomic models are public. See [`docs/SECRETS.md`](docs/SECRETS.md).*
