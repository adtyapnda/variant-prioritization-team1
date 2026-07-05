# Akita — Predicting 3D genome folding from DNA sequence

**Assigned to:** Aditya
**Codebase:** https://github.com/calico/basenji/tree/master/manuscripts/akita
**Utilities:** https://github.com/Fudenberg-Research-Group/akita_utils

## About
Deep CNN that takes ~1 Mb of DNA sequence and predicts a Hi-C contact map
(3D genome folding) for that locus.

## How to run

**Starter notebook:** [`akita_inference_colab.ipynb`](./akita_inference_colab.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adtyapnda/variant-prioritization-team1/blob/main/papers/04-akita/akita_inference_colab.ipynb)

Runs **inference only** with the authors' pre-trained model (no training). It:
1. Installs basenji + `cooltools` + `pysam`
2. Downloads the trained weights (`model_best.h5`) and the hg38 FASTA (~1 GB, one-time)
3. Takes a 1 Mb (2²⁰ bp) hg38 region, one-hot encodes it, and predicts a contact map
4. Plots and saves the map as `akita_pred.png`

CPU runtime is fine for a single prediction — no GPU needed.

> **Dependency note:** basenji pins older library versions, so Colab may throw version
> conflicts. If imports fail right after the install cell: **Runtime > Restart session**,
> then re-run from the *Imports* cell. First run usually needs a little trial and error.

## Output
_TBD — run the notebook, then commit `akita_pred.png` here._

## Status
🟡 Notebook ready — not yet run end-to-end
