# Dataset Bundle — AI in Medicine / Genomics two-day course

This bundle is split into **(A) data already pulled here** (reachable from GitHub or generated offline) and **(B) data you must pull on a Midway3 login node** via `scripts/download_datasets.sh` (Hugging Face / Zenodo / NCBI are not reachable from the build sandbox).

## What's already in this bundle (ready to transfer to shared project space)

### Day 1 — AI in Medicine
- `day1_ai_medicine/tabular_clinical_diabetes.csv` — **real**, 442 patients × 10 features, binary `high_progression` label (above-median 1-year disease progression). Source: scikit-learn bundled Diabetes dataset (Efron et al. 2004). Public domain. **Includes demographic columns (`age`, `sex` as de-identified groups A/B) so the Module-1 subgroup/fairness audit is runnable.** (This replaces the earlier breast-cancer set, which had no demographic columns and could not support the fairness audit.) UCI Heart Disease — which also carries age/sex — is a fine richer alternative pulled on the login node if preferred.

### Day 2 — Genomics
- `day2_genomics/brca1_variants.csv` — **real & cleaned**, 3,893 BRCA1 SNVs with `func_class` (FUNC 2821 / INT 249 / LOF 823), `is_lof` binary label, `clinvar`, `consequence`, `CADD.score`, hg19 coordinates. This is the labelled set the zero-shot VEP AUROC is computed against.
- `day2_genomics/brca1_findlay2018_raw.xlsx` — the full 62-column source (Findlay et al., *Nature* 2018, BRCA1 saturation genome editing).
- `day2_genomics/GRCh37.p13_chr17.fna.gz` — **real**, chromosome 17 reference (GRCh37/hg19) for building ref/alt windows around each variant. Matches the hg19 coordinates in the variant table.
- `day2_genomics/REFERENCE_evo2_brca1_vep.ipynb` — the official Evo 2 BRCA1 zero-shot notebook (reference implementation of the delta-likelihood method).
- `day2_genomics/seqmodeling_extras/` — prokaryotic/eukaryotic material for the sequence-modeling module: ϕX174 G-protein FASTA, E. coli genome (GenBank), exon `samplePositions.tsv`.

Source for all Day-2 files: the public **ArcInstitute/evo2** GitHub repo (Apache-2.0 code; underlying data per original publications).

### Fixtures (synthetic — for OFFLINE smoke-testing the notebooks/scripts only)
- `fixtures/multimodal_smoke.npz` — 600× (512-d image embedding + 8 tabular features + label); label depends on both modalities so fusion can beat either alone.
- `fixtures/cxr_smoke.npz` — 200× tiny synthetic 28×28 grayscale "radiographs". **Replace with real PneumoniaMNIST** for teaching.
- `fixtures/precomputed_vep_scores_SYNTHETIC.csv` — schema-correct precomputed delta-score table so the no-GPU Day-2 fallback path runs. **SYNTHETIC — regenerate with the real model** (`scripts/download_datasets.sh` + a scoring run) before using for anything but a smoke test.

## What you pull on the login node (`scripts/download_datasets.sh`)
- **MedMNIST** (PneumoniaMNIST, ChestMNIST) — Zenodo. License CC-BY-4.0.
- **HyenaDNA** weights (`LongSafari/hyenadna-tiny-1k-seqlen-hf`, `-small-32k-seqlen-hf`) — **the recommended smaller foundation model**: human-reference-genome pretrained, autoregressive, single-nucleotide, 6.6M params (small-32k), runs on a 16 GB GPU or CPU. Preserves the exact delta-log-likelihood VEP method.
- **(optional) Evo 2 7B** (`--with-evo2`) — upgrade tier; needs >32 GB VRAM (A100 ok; 20B/40B need Hopper/FP8 — excluded). Reproduces the >90% BRCA1 result.
- **(optional) Evo 1** (`--with-evo1`) — ⚠️ prokaryotic (OpenGenome) training; good for the prokaryotic sequence-modeling extras, a **poor fit for human BRCA1**. Prefer HyenaDNA or Evo 2 for the variant exercise.
- **(optional) MedRAX** (`--with-medrax`) — agentic CXR repo + tool weights for Module 3b.

## Model choice summary (Day 2)
Teach the **method** with **HyenaDNA-small-32k** (tiny, human-genome, autoregressive, runs anywhere). Treat **Evo 2 7B** as the optional "what scale buys you" upgrade and compare its AUROC to the small model's on the same `brca1_variants.csv`. The pipeline (build window from `GRCh37.p13_chr17.fna.gz` → score ref & alt → `delta = LL(alt) − LL(ref)` → AUROC vs `is_lof`) is identical across all three models.

## Notes
- All datasets are open / non-PHI. No MIMIC or other credentialed data is used; any such extension would require separate PhysioNet credentialing.
- BRCA1 coordinates are **hg19/GRCh37** — use the bundled chr17 (GRCh37), not hg38.
- `CHECKSUMS.sha256` covers the bundled files; `download_datasets.sh` writes a second checksum file for login-node pulls.

## Login-node pulls (2026-06-15)
- MedMNIST PneumoniaMNIST+ChestMNIST (Zenodo, CC-BY-4.0) -> caches/medmnist
- HyenaDNA tiny-1k + small-32k (HF LongSafari, public, Apache-2.0 code) -> caches/hf
- Evo 2 7B weights (HF arcinstitute/evo2_7b, public) -> caches/hf
- torchvision resnet18 IMAGENET1K_V1 (BSD) -> caches/torch
