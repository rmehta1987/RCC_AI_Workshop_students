# Launching Jupyter on Midway3 (two paths)

Both paths land you in Jupyter on a CPU (`caslake`) node with the **AImed** kernel selected.
The whole course runs on CPU — no GPU needed.

## Path A — Open OnDemand (recommended, no terminal)

1. Browser → **https://midway3.rcc.uchicago.edu** → log in (CNetID + 2FA).
2. **Interactive Apps → Jupyter** (or "Jupyter Notebook/Lab").
3. Fill the form:

   | Field | Value |
   |---|---|
   | Account | `workshop-aiml` |
   | Partition | `caslake` |
   | CPUs / cores | `8` |
   | Memory (GB) | `32` |
   | Walltime (hours) | `4` |
   | Environment / modules | leave default; we select the kernel *inside* Jupyter |

4. **Launch** → **Connect to Jupyter** when it turns green.
5. In Jupyter, open your `students/<cnetid>/notebooks/day1_m1_clinical_risk.ipynb` and pick the
   **`AImed`** kernel (Kernel → Change Kernel). The shared top-level `notebooks/` is the
   read-only master — work in your own writable `students/<cnetid>/` folder.

> If the `AImed` kernel is not listed, run once in a terminal (or a notebook cell):
> `!ls $PROJECT_DIR/env/share/jupyter/kernels` — it should show `aimed`. The
> kernel is registered under `PROJECT_DIR/env`; OnDemand finds it because the
> launch script exports `JUPYTER_PATH` (see Path B) — if using OnDemand's own
> Python, add `JUPYTER_PATH=/project/workshop-aiml/ai_medicine_genomics/env/share/jupyter` in
> the app's "Environment" field.

## Path B — sbatch + SSH tunnel (no OnDemand needed)

```bash
ssh <cnetid>@midway3.rcc.uchicago.edu
cd /project/workshop-aiml/ai_medicine_genomics
sbatch slurm/launch_jupyter.sbatch
squeue --me                     # wait for ST=R, note the node
cat logs/jupyter-<cnetid>-<jobid>.log    # prints the node, port, tunnel cmd, and tokened URL
```

The log prints exactly what to run on your laptop, e.g.:

```bash
ssh -N -L 8412:caslake0123:8412 <cnetid>@midway3.rcc.uchicago.edu
# then open the http://127.0.0.1:8412/lab?token=... URL from the log
```

Pick the **AImed** kernel and open the Day-1 notebook from your `students/<cnetid>/notebooks/` folder
(the `sbatch` path starts you at the read-only course root — navigate into your folder).

## Smoke test (confirms the kernel)

Run in the first notebook cell:

```python
import torch; print("torch", torch.__version__, "(CPU)")
from transformers import AutoModelForCausalLM, AutoTokenizer
name = "LongSafari/hyenadna-small-32k-seqlen-hf"
tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True).eval()
print("HyenaDNA loaded OK")
```

The caslake node should print `HyenaDNA loaded OK` (weights come from the shared
cache, no download).
