# Student README — get into a working notebook in ~5 minutes

You are in the **AI in Medicine & Genomics** two-day course on UChicago RCC
**Midway3**. Everything is pre-installed and pre-staged. No downloads, no setup.

## The 5 must-do steps (on Midway3)

```bash
# 1. SSH in
ssh <cnetid>@midway3.rcc.uchicago.edu

# 2. Go to the shared course dir
cd /project/workshop-aiml/ai_medicine_genomics

# 3. Activate the shared environment (one line)
source scripts/common/load_profile.sh && aimed_activate
python -c "import torch, transformers; print('env OK')"

# 4. (optional) secrets — ONLY if your team uses an LLM endpoint for the agent
bash scripts/setup_user_secrets.sh        # safe to skip; the agent works offline

# 5. Launch Jupyter and open Day 1
#    Easiest: Open OnDemand  ->  https://midway3.rcc.uchicago.edu  (see docs/JUPYTER.md)
#    Or:      sbatch slurm/launch_jupyter.sbatch   (then read logs/jupyter-<cnetid>-<jobid>.log)
```

> **Work in your own folder — it's writable and kept up to date:**
> `/project/workshop-aiml/ai_medicine_genomics/students/<cnetid>/notebooks/`
> The shared top-level `notebooks/` is the **read-only master** — don't edit it directly. If a notebook
> you've already started gets updated, the new version lands beside yours as `<name>.NEW.ipynb`, so
> **your work is never overwritten**. (Earlier copied notebooks to your home dir? Switch back to your
> `students/<cnetid>/` folder so you keep getting updates.)

In Jupyter, open your `students/<cnetid>/notebooks/day1_m1_clinical_risk.ipynb`, pick the
**`AImed (shared GPU/CPU)`** kernel, and work each module top-to-bottom. (The `sbatch` path drops you at
the read-only course root — just navigate into your `students/<cnetid>/notebooks/` folder.) Stuck on a
step? Run the **CHECKPOINT** cell — it shows the answer so you can keep moving.

**No GPU?** The whole course runs on **CPU** — in OnDemand pick partition `caslake` with `0` GPUs
(account `workshop-aiml`). No API key is needed (the Day-1 agent uses an offline mock LLM); model
weights come from the shared cache.

## Module map

| Day 1 | Day 2 |
|---|---|
| `day1_m1_clinical_risk` — tabular risk + calibration | `day2_m1_seqmodel` — DNA LM likelihoods |
| `day1_m2_radiology` — CNN confidence & abstention (OOD) | `day2_m2_vep` — zero-shot variant scoring |
| `day1_m3a_fusion` — late fusion | `day2_m3_brca1_triage` — BRCA1 end-to-end |
| `day1_m3b_agent` — minimal CXR agent | |

## Run it locally (CPU-only laptop)

Not on Midway3? The whole course runs on a **CPU** — no GPU, no special kernel.

1. **Code + small data** — `git clone` ships the course code **and** the small, non-PHI data the
   notebooks read (the diabetes table, `brca1_variants.csv`, both real precomputed VEP tables —
   HyenaDNA + Evo 2 0.877 — the seqmodeling FASTA, and the smoke fixtures). Only model weights and
   the 22 MB chr17 reference are left out.
   ```bash
   python3.11 -m venv .venv && source .venv/bin/activate
   pip install -r requirements-cpu.txt          # CPU PyTorch + the course stack
   ```
2. **First-run downloads** — **HyenaDNA-small** downloads automatically the first time Day-2 uses it
   (one-time, internet). **MedMNIST does *not* auto-download:** on a fresh clone Day-1 M2 runs on the
   bundled **synthetic smoke fixture** (fine for the Grad-CAM and OOD lessons — the OOD demo also uses
   generated noise + flipped scans). *(Optional)* for the **real** chest X-rays, pre-fetch once from the
   repo root:
   ```bash
   python -c "from medmnist import PneumoniaMNIST; [PneumoniaMNIST(split=s, download=True, size=64, root='caches/medmnist') for s in ('train','val','test')]"
   ```
   (The OOD demo's extra "other-finding CXR" example also wants `ChestMNIST` — append it to that line —
   but it's 383 MB and optional; the OOD lesson works without it.) The 22 MB chr17 reference is **not**
   needed (Day-2 uses the precomputed scoring path); grab it from `aimed_datasets_bundle.tar.gz` only if
   you want to score variants with a model live.
3. **Launch** with the one-command helper. It starts JupyterLab **in the background**, opens your
   browser, and **gives your terminal right back** — there's no server window to keep open or quit:
   ```bash
   bash scripts/run_local.sh          # start (browser opens; prompt returns)
   bash scripts/run_local.sh stop     # shut it down when you're done
   ```
   Open `notebooks/day1_m1_clinical_risk.ipynb` and pick the venv's **Python 3** kernel.
   Everything is offline after that first download. Day 2 never runs Evo 2 locally — it reads the table.

   > **What the helper does** (in case you launch `jupyter lab` yourself): it runs from the repo root
   > with `AIMED_PROFILE=/dev/null`. Your clone includes `CLUSTER_PROFILE.md`, which pins
   > `DATA_DIR`/`HF_HOME` to the HPC's `/project/...` paths; off-cluster that makes every cell fail
   > with "no such file or directory: /project/...". Pointing `AIMED_PROFILE` at an empty file tells
   > the loader to ignore the cluster profile, so all paths derive from your local repo root instead
   > (`./data`, `./caches`).
   >
   > **Self-contained local clones:** the same one override works for everyone — give each person their
   > own clone/copy (it already includes `scripts/` and the committed `data/`), leave `PROJECT_DIR`
   > unset, and each copy auto-resolves to its own `data/` + `caches/` (no shared state). Equivalent:
   > delete `CLUSTER_PROFILE.md` from the copy and no env var is needed at all. *(This is the laptop
   > model — distinct from the cluster's shared `students/<cnetid>/` folders above.)*

## Troubleshooting (one line each)

| Symptom | Fix |
|---|---|
| **`AImed` kernel missing** | `export JUPYTER_PATH=$PROJECT_DIR/env/share/jupyter` then restart Jupyter; confirm `ls $PROJECT_DIR/env/share/jupyter/kernels` shows `aimed`. |
| **`torch.cuda.is_available()` is False** | You're on a CPU node — totally fine. Day 2: use small `--n`/window, or the precomputed table. |
| **GPU partition busy** | HyenaDNA runs on **CPU**; or submit to `caslake`. The whole course works without a GPU. |
| **`module: wrong # args ... SOFTPATH`** | You ran `module purge` — don't. Open a fresh shell; use `module unload <name>` instead. |
| **Quota / disk full** | Don't write to `$HOME`. Write outputs under your own `/scratch/midway3/$USER`. |
| **No OpenAI key for the agent** | Nothing to do — the agent uses the offline MockLLM and the gate still passes. |
| **Day-1 M2 says "using smoke fixture" (local)** | Expected on a fresh clone — MedMNIST isn't auto-fetched; pre-fetch it (Run-it-locally step 2) for the real X-rays. On the cluster M2 uses real data from the shared cache. |
| **HyenaDNA download attempt on a compute node** | You're offline on a compute node (correct). Weights are in the shared cache; `cfg.setup_caches()` (run by the notebook bootstrap) points there. |
| **`run_vep` is slow on CPU** | Use `--use-precomputed`, or a smaller `--n` and `--window 1024`. |

## Etiquette (shared GPUs)

One GPU session per student; `scancel <jobid>` idle sessions. Use the precomputed
BRCA1 table instead of re-scoring 7,786 windows. Right-size `--time`. Check yourself
with `squeue --me`.

*(More detail: `docs/JUPYTER.md`, `docs/SECRETS.md`. Profile / config: `CLUSTER_PROFILE.md`.)*
