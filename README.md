# Team 1 — Variant Prioritization

A group project exploring **deep learning methods in genomics**. Each member takes one
research paper, sets up the authors' original code, **runs the model on the example / demo
data, and captures the output** (variant calls, basecalls, cell maps, contact maps, etc.).
Paper #5 (the Eraslan et al. review) is also part of Aditya's work — it surveys the
field rather than shipping one tool, so the exact approach is still being figured out.

---

## 🎯 Goal

For each paper:

1. Get the authors' codebase running (their GitHub repo is linked below).
2. Run the model on the provided demo / example data — **inference only, no training needed**.
3. Save the output into that paper's folder, with a short note on what it shows.

> **Note on compute.** These are heavy deep-learning tools built for **Linux + GPU**, so
> they are hard to run natively on Windows. The practical routes for getting output are:
> - **Google Colab** — free GPU, Linux, ready-made notebooks → best for **Akita** & **cell2location**
> - **Docker Desktop** — official image, runs on CPU with demo data → best for **DeepVariant**
> - **WSL2** — general fallback for the rest

---

## 📄 Paper assignments

| # | Paper | Member | Codebase | How to run |
|---|-------|--------|----------|------------|
| 1 | A universal SNP and small-indel variant caller using deep neural networks (**DeepVariant**) | Bhumika | [google/deepvariant](https://github.com/google/deepvariant) | Docker, CPU is fine |
| 2 | Chiron: translating nanopore raw signal directly into nucleotide sequence using deep learning | Utkarsh | [haotianteng/Chiron](https://github.com/haotianteng/Chiron) | Old TF1 env (tricky) |
| 3 | Cell2location maps fine-grained cell types in spatial transcriptomics | Kamakshi | [BayraktarLab/cell2location](https://github.com/BayraktarLab/cell2location) | Colab (demo data) |
| 4 | Predicting 3D genome folding from DNA sequence with **Akita** | Aditya | [calico/basenji → akita](https://github.com/calico/basenji/tree/master/manuscripts/akita) | Colab, inference |
| 5 | Deep learning: new computational modelling techniques for genomics *(review paper)* | Aditya | — (surveys many tools) | TBD — we'll figure it out |

Extra Akita utilities: [Fudenberg-Research-Group/akita_utils](https://github.com/Fudenberg-Research-Group/akita_utils)

---

## 👥 Team

| Member | GitHub |
|--------|--------|
| Aditya | [@adtyapnda](https://github.com/adtyapnda) |
| Bhumika | [@BhumikaMansinghka](https://github.com/BhumikaMansinghka) |
| Utkarsh | [@utk1017](https://github.com/utk1017) |
| Kamakshi | [kams19-ops](https://github.com/kams19-ops) |

---

## 📁 Repository structure

```
papers/
  01-deepvariant/                 DeepVariant  — Bhumika
  02-chiron/                      Chiron       — Utkarsh
  03-cell2location/               cell2location— Kamakshi
  04-akita/                       Akita        — Aditya
  05-deep-learning-genomics-review/  review    — Aditya (background)
docs/                             shared setup notes
```

*(Folders are added as each member starts their paper.)*

---

## 🚀 Getting started

1. Clone the repo:
   ```bash
   git clone https://github.com/adtyapnda/variant-prioritization-team1.git
   ```
2. Work inside your paper's folder under `papers/`.
3. Commit your setup steps, run scripts / notebooks, and output files.
4. Add a short note in your folder's README describing what the output shows.
