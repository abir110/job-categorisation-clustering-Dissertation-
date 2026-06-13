# Job Categorisation in the Gaming & Tech Industry via Skill-Based Clustering

> MSc Dissertation (Distinction) — Aston University, Data Science

## TL;DR

I analysed **70,000+ job postings** across the gaming and technology sectors to test a simple question: *can you replace messy, inconsistent job titles with a skill-based taxonomy?*

The honest answer is **mostly no — and that's the actual finding.** Job titles in tech are far more fluid and hybrid than traditional taxonomies assume. Only **~6% of postings** could be confidently assigned to a single skill-based category; the remaining **~94% are genuinely hybrid roles** that span multiple skill clusters at once. Rather than force these into artificial categories, this project builds a **confidence-aware classifier** that explicitly flags how "hybrid" a role is — which is arguably more useful for workforce analytics than a forced single label.

---

## The Problem

Job titles are a weak signal. A "Data Analyst" at one company can look identical to a "BI Analyst" at another, while two roles with the *same* title can require completely different skill sets. This makes job titles unreliable for:

- Labour market trend analysis
- Career pathing / skill-gap analysis
- Matching candidates to roles based on actual requirements

## Approach

1. **Cleaning & preprocessing** — standardised skills, salaries, seniority levels, and gaming vs. general-tech classification across 70k postings (64,230 retained after cleaning)
2. **TF-IDF vectorisation** of job skill lists
3. **Clustering** — compared K-Means across k=20–120 and HDBSCAN, selecting the model that balanced cluster count with interpretability
4. **Confidence scoring** — every job gets a cluster membership confidence score (0–1); jobs below 0.5 are explicitly labelled "Uncertain/Hybrid" rather than forced into a cluster
5. **Validation** — Kendall's Tau rank correlation between original categories and new skill-based groups, plus threshold experiments (precision/coverage trade-off at different confidence cut-offs)
6. **Sub-taxonomies** — separate gaming-only and non-gaming-only clustering to test whether the gaming sector has a distinct skill structure

## Key Findings

| Metric | Result |
|---|---|
| Postings analysed (after cleaning) | 64,230 |
| Confidently categorised (conf ≥ 0.5) | ~6.3% |
| Uncertain / Hybrid roles | **93.7%** |
| Distinct skill-based clusters identified | 12 |
| Silhouette score (best K-Means) | 0.07 |

**What this means:** the low silhouette score and high "uncertain" rate aren't a failed model — they're evidence that **the underlying job market doesn't actually have clean boundaries**. The t-SNE visualisation below shows this directly: rather than distinct, separated clusters, most postings sit in one large continuous "skill space," with only a handful of tightly-defined niche clusters (e.g. specific tooling combinations) standing out at the edges.

This has a practical implication: **skill-based job classification works well for niche/specialist roles, but breaks down for generalist and hybrid roles** — which describes most of the modern job market.

## Visuals

- Dataset cleaning pipeline (cleaning_steps.png)
- Original job category distribution (original_categories.png)
- Silhouette score vs. k (silhouette_vs_k.png)
- Cluster confidence distribution (confidence_distribution.png)
- Share of jobs by confidence level (confidence_by_level.png)
- t-SNE visualisation of the full skill space (tsne_clusters.png)
- Confidence threshold vs. cluster purity trade-off (threshold_purity.png)

## Tech Stack

Python · Pandas · NumPy · Scikit-learn (TF-IDF, K-Means, t-SNE) · HDBSCAN · SciPy (Kendall's Tau) · Matplotlib · Seaborn

## Repository Structure

```
├── notebook/
│   └── job_categorisation_clustering.ipynb   # Full analysis pipeline
├── data/
│   ├── cluster_summary.csv                   # Cluster-level summary stats
│   ├── confidence_threshold_experiments.csv  # Purity vs. confidence trade-off
│   ├── kendall_tau_top_categories.csv        # Validation: old vs. new taxonomy correlation
│   └── model_card.txt                        # Final model parameters & metrics
├── figures/                                   # Exported visualisations
└── README.md
```

> Note: the raw scraped job postings dataset (~25MB) is not included in this repo due to size and licensing. The notebook documents the expected schema for reproducibility.

## Author

**Abir Alam Srabon** — MSc Data Science, Aston University
[LinkedIn](https://www.linkedin.com/in/abir-alam-srabon) · [Portfolio](https://abir110.github.io)
