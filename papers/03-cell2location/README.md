# **cell2location — Maps fine-grained cell types in spatial transcriptomics**

**Assigned to: Kamakshi**

**Codebase: https://github.com/BayraktarLab/cell2location**

**About -**
Bayesian model that maps cell types onto spatial transcriptomics data by integrating single-cell reference signatures. Estimates which combination of cell types, in what abundance, explains the mRNA counts observed at each spatial location, while modelling technical effects (platform effects, contaminating RNA, unexplained variance).

**How to run -**
Notebook: cell2location_tutorial.ipynb (this folder), run on Google Colab (T4 GPU).

Two-stage pipeline on the public human lymph node Visium dataset (10x Genomics) + a lymphoid organ scRNA-seq reference (~73,260 cells, ~34 annotated cell types):

1. Load and QC both datasets (mitochondrial gene filtering, ENSEMBL ID matching).
2. Stage 1 — estimate reference cell type signatures: NB regression model trained on the scRNA-seq reference (`max_epochs=250`, ~15–20 min on T4).
3. Stage 2 — spatial mapping: cell2location model trained on the Visium data using the stage-1 signatures (`max_epochs=30000`, `batch_size=None`, full-data training — ~1.5–2 hr on T4, the compute-heavy step).
4. Export posterior cell abundance estimates (5% quantile — "at least this amount is present") back onto the spatial AnnData object.
5. Visualize predicted cell-type abundance overlaid on the tissue image (`sc.pl.spatial`).
Needs a GPU-backed environment (Colab free tier works, T4). 

**Output -**

* Per-cell-type spatial abundance grid (8 panels: B_Cycling, B_GC_LZ, T_CD4+_TfH_GC, FDC, B_naive, T_CD4+_naive, B_plasma, Endo) — B_Cycling, B_GC_LZ, T_CD4+_TfH_GC, and FDC all light up in the same spatial locations, consistent with these four types jointly constituting germinal centers.
* 3-color composite overlay (T_CD4+_naive / B_naive / FDC) on the histology image — shows these three cell types occupying largely distinct, non-overlapping territory, recovering the expected T-zone / B-zone / follicle segregation of lymph node anatomy without being given any spatial priors.
* Reference signature QC plot (posterior expected vs. observed expression, reconstruction accuracy) and ELBO loss history for both training stages.

**Issues encountered & fixes -**
* `scvi-colab`'s `install()` step hung repeatedly (10+ min, no progress) — replaced with a direct manual install (`scvi-tools`, `cell2location[tutorials]`, `scanpy` via pip), which completed in under a minute.
* `ImportError: cannot import name 'sentinel' from 'typing_extensions'` — caused by an outdated `typing_extensions` preinstalled in the Colab image conflicting with `anndata`. Fixed with `pip install --upgrade typing_extensions` before the rest of the install, followed by a full runtime restart.
* Reduced Stage 2 `max_epochs` from the tutorial default (30000) to a lower value to manage total runtime — verified via `mod.plot_history()` that the ELBO loss had plateaued before treating the result as converged.

Status
🟡 Stage 1 (reference signatures) and Stage 2 (spatial mapping) both run successfully; core spatial abundance figures generated. Notebook uploaded through the visualization step.
