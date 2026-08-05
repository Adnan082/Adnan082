# Adnan Haider Cheema

**Learned surrogates for physical systems — and the instruments that catch them when they're wrong.**

Physics BSc, MSc Applied Data Science. I work on the same problem in four domains: when you replace an expensive physical model with a learned one, you get speed, and you lose the one thing the physical model gave you for free — a principled account of when it's wrong. Most of my repositories are attempts to get that back: calibration diagnostics, conformal intervals, physics kept in the loop as an untrained referee.

Cambridge, UK · [LinkedIn](https://www.linkedin.com/in/adnan-haider-cheema/) · [adnancheema917@gmail.com](mailto:adnancheema917@gmail.com)

---

## Selected work

### NODA — when a neural surrogate makes an ensemble filter confidently wrong

**[`Adnan082/NODA`](https://github.com/Adnan082/NODA)** · JAX · Equinox · jax-cfd · Hydra

Real-time state estimation of a 2D turbulent vorticity field (128×128) from ~200 noisy sensors — about 1.2% coverage — using an ensemble Kalman filter whose forecast model is a Fourier Neural Operator instead of a numerical solver.

**The problem.** An EnKF's only instrument for uncertainty is disagreement between ensemble members. Replace the solver with a neural surrogate and every member passes through the *same weights*, so any error the network makes is shared by all members and produces no disagreement at all. State error still registers correctly. Model error doesn't. The filter reports itself as trustworthy while being badly wrong, with no internal signal that anything has happened.

**What I measured.** Four configurations, everything else held identical. Spread–skill ratio of 1.0 means the ensemble's confidence matches its actual error; below 1.0 means overconfident.

| Config | Forward model | Spread–skill | RMSE |
|---|---|---:|---:|
| A | Numerical solver (control) | 0.998 | 0.116 |
| B | One trained FNO | 0.207 | 1.016 |
| C | One FNO + tuned inflation | 1.251 | 1.005 |
| D | Five independently-trained FNOs | 0.180 | 1.437 |

Two negative results worth more than the positive one. **C** shows that covariance inflation buys a better-looking calibration number without recovering any accuracy — inflation is isotropic, the surrogate's bias is directional. **D** shows that a multi-model ensemble, the standard prescription, does not fix the overconfidence either, and comes out slightly *less* accurate than a single network.

**What worked.** A physics-based referee: the PDE residual of the assimilated state, computed by the governing equations, never touching ensemble spread or network weights. Because it was never trained, it cannot share the surrogate's blind spot. Across an induced Reynolds-number and forcing shift the surrogate never trained on, it detects the regime change at **ROC AUC 0.971**, confirmed on two independently regenerated OOD datasets, with essentially no false alarms beforehand.

```
        ┌──────────── the loop the surrogate lives in ────────────┐
        │                                                          │
   ~200 sensors ──► EnKF update ──► 100 ensemble members           │
   (1.2% of field)       ▲                    │                    │
                         │                    ▼                    │
                         └──── FNO forecast ──┘                    │
                              ⚠ same weights for every member      │
        └──────────────────────────┬───────────────────────────────┘
                                   │  assimilated state
                                   ▼
                        PDE residual  ← never trained, so it cannot
                        (the referee)    share the network's blind spot
```

<details>
<summary><b>Engineering, honest limitations, and what's next</b></summary>

**Why these choices.** JAX + Equinox for `jit`/`vmap` over ensemble members — a 100-member EnKF is embarrassingly parallel and this is where the speedup actually comes from. `jax-cfd` for the ground-truth solver so the control arm is a real numerical baseline rather than a strawman. Hydra configs so every experiment is a command-line override rather than an edited constant. Seeded reproducibility tests and physics-conservation tests in the suite, because a silent determinism break invalidates every calibration number downstream.

**Stated plainly.** The surrogate is roughly 10× less accurate than the real solver at matched ensemble size and 70 assimilation cycles (RMSE 0.999 vs 0.083). Speed is the point, not accuracy — it buys more members or more frequent assimilation. And detecting the drift did **not** produce full recovery once the filter fell back to the numerical solver, because that solver was still configured for the original training regime. Detection is solved here. Knowing what to do about an unknown regime after detecting it is open.

**Next.** Online regime re-identification so the fallback solver reconfigures itself; packaging the calibration diagnostics (spread–skill, rank histograms, residual score) as a standalone harness that works against any learned forecast model.

**Reproduce.** `make data && make bench` regenerates all four experiment figures. CI runs lint and the test suite on every push.
</details>

---

### Solar Wind Dst — physics ODE, learned residual, and an RL agent deciding which to trust

**[`Adnan082/Hybrid_Model_Solar_winds_Prediction`](https://github.com/Adnan082/Hybrid_Model_Solar_winds_Prediction)** · PyTorch · Redis · FastAPI · Dash

Real-time forecasting of the Dst geomagnetic storm index from solar wind measurements, running against live NOAA SWPC feeds. Five agents on a Redis pub/sub bus: a Burton ODE physics solver, a Transformer autoencoder for anomaly detection, a BiLSTM that learns the residual the ODE can't capture, a fusion stage, and an actor–critic agent that learns the blending weights between physics and ML online.

**Why the structure.** The Burton ODE is decades-old, cheap, and degrades gracefully — it is never allowed to leave the system. The BiLSTM is not asked to predict Dst; it's asked to predict what the physics gets wrong, which is a smaller and better-posed target. The RL agent decides, per timestep, how much to trust each.

**Results — per storm class, not aggregate.** Overall RMSE on Dst is dominated by quiet conditions and hides failure during the storms that actually matter, so the ablation is decomposed by severity:

| Storm class | Dst range | Burton RMSE | Corrector RMSE |
|---|---|---:|---:|
| Quiet | > −30 nT | — | 4.91 nT |
| Minor | −30 to −50 nT | — | 7.74 nT |
| Moderate | −50 to −100 nT | — | 11.84 nT |
| Intense | −100 to −200 nT | ~42 nT | 13.71 nT |
| Extreme | < −200 nT | ~42 nT | 6.50 nT |

Burton baselines are measured on the same test rows, not quoted from literature. `—` means the physics baseline isn't competitive enough at that class to be worth reporting.

The Transformer autoencoder separates extreme storms from quiet conditions by a **237× reconstruction-error ratio** (0.47 vs ~0.002), which is what makes it usable as a tiered alert trigger rather than a binary flag. Total model footprint is 790K parameters, and end-to-end model inference measures p50 11.7 ms / p95 18.7 ms / p99 46.4 ms on CPU — the latency budget is what makes a 60-second polling loop against a live satellite feed feasible.

<details>
<summary><b>Caveats I'd want a reviewer to know</b></summary>

The 6.50 nT extreme-class figure and the `val_rmse_nT` in `corrector_config.json` coincide; the config value was computed on a storm-enriched validation set during training. True population RMSE across unsampled data is lower, because quiet conditions dominate. `python validate_storms.py --full-dataset` produces the verified number for your own split.

**The RL blend is not yet independently measured.** The ablation table's blend column is empty. The agent runs and learns weights online, but I haven't yet produced a controlled comparison of blended output against the corrector alone, so I'm not claiming one.

Comparison against published Dst models (Gruet 2018, Siciliano 2021, Shrivastava 2022) is informative but uncontrolled — different datasets, different time windows, different test splits. Treat it as orientation, not a benchmark.

**Next.** Containerise the full stack (currently only Redis runs in Docker); fill in the blend ablation; longer forecast horizons.
</details>

---

### TurbineAgent — fleet prognostics with calibrated intervals, not point estimates

**[`Adnan082/NASA_Turbine_Engine`](https://github.com/Adnan082/NASA_Turbine_Engine)** · PyTorch · FastAPI · Streamlit · MLflow · Docker

Remaining-useful-life prediction across 707 turbofan engines from the NASA C-MAPSS run-to-failure benchmark, using all four fault-mode sub-datasets. Four agents on an asyncio event bus: anomaly detection, RUL regression, SHAP attribution, and a rule-based triage stage that sorts the fleet into five priority tiers.

| Metric | Value |
|---|---|
| RUL MAE | 12.2 cycles (CNN-BiLSTM) |
| RUL RMSE | 17.6 cycles |
| Conformal interval | ±33.3 cycles at 90% coverage |
| Near-failure capture rate | 39.6% |
| Fleet flagged anomalous | 190 / 707 (26.9%) |
| Classified CRITICAL | 10 engines |

**The part that matters.** A maintenance planner cannot act on "MAE 12.2 cycles." They can act on "this engine has 40 cycles left, and I am 90% confident the true value is between 7 and 73." Split conformal prediction, implemented from scratch, gives a distribution-free coverage guarantee without assuming the error is Gaussian — which it isn't, because RUL error grows sharply near end-of-life. Anomaly thresholds are set per operating condition across six KMeans-clustered flight regimes, since a sensor reading that's normal at cruise is not normal at takeoff.

**Where it's weak, plainly:** 39.6% near-failure capture is not good. It's the honest number at a threshold tuned to keep false alarms low enough that fleet-wide triage stays useful, and the trade-off is real, but I'd rather state it than bury it. `docker compose up` runs the full stack.

---

### POLARIS — polarimetric imaging for road-surface state *(in progress)*

**[`Adnan082/POLARIS`](https://github.com/Adnan082/POLARIS---Polarimetric-Observation-for-Learned-All-weather-Road-surface-Identification-of-States)** · PyTorch · AWS

Classifying road surface state — dry, damp, wet, slush, snow — from polarimetric imagery on the [PRISM dataset](https://huggingface.co/datasets/NeurIPS-2026-PRISM/PRISM-Dataset), testing whether Stokes-parameter inputs beat RGB-only baselines under exactly the conditions where RGB fails.

The premise is physical rather than architectural: light reflecting off a wet surface is polarised in a way that carries surface-state information RGB intensity simply doesn't encode. If that holds, the gain should be largest precisely where the RGB baseline is weakest.

Built: Stokes precompute pipeline, session-level splits to prevent temporal leakage, three configured model variants (RGB / polar / fusion), spot-instance training with interruption handling.

**Not yet built: the results.** The comparison isn't finished, so there are no numbers here to report. I'd rather leave this section empty than fill it in optimistically.

---

## Also on this profile

| Repository | What it is | Note |
|---|---|---|
| [`face-detection`](https://github.com/Adnan082/face-detection) | StyleGAN3 vs. real face classification, EfficientNet-B0 fine-tune | 97.8% acc / 0.9970 AUC on a 20k balanced split |
| [`uhi-mitigation-mapper`](https://github.com/Adnan082/uhi-mitigation-mapper) | Tree-planting site prioritisation from Landsat LST, NLCD land cover, and Census vulnerability | Geospatial pipeline, ranked action list for planners |
| [`RAG`](https://github.com/Adnan082/RAG) | Fully local semantic search over 50k arXiv physics abstracts | ChromaDB, no API keys, offline at query time |
| [`churn_prediction`](https://github.com/Adnan082/churn_prediction) | Telecom churn classification with SHAP attribution | Small UCI-family dataset; treat the high AUC with suspicion |
| [`Missile_System`](https://github.com/Adnan082/Missile_System) | Soft actor–critic learning 3D pursuit–evasion control against an evasive target, without a hand-derived guidance law | 5.5M steps, staged curriculum |

---

## How I work

Four projects, one habit: build the learned model, then build the thing that tells you when to stop believing it.

- **Physics stays in the system.** Not as a feature-engineering step that gets discarded, but as a live component — a baseline the network corrects (Dst), an untrained referee that audits the network (NODA), a fallback the system can revert to.
- **Uncertainty is a deliverable, not a diagnostic.** Conformal intervals, spread–skill ratios, rank histograms. A point estimate with no interval is not an answer anyone can act on.
- **Decompose the metric.** Aggregate error hides failure exactly where it's expensive. Per-storm-class RMSE, per-operating-condition thresholds.
- **Negative results are kept.** Config D of NODA didn't work and it's in the README with a chart. That's the finding.
- **Reproducible from a cold clone**, or it doesn't count.

`Python` `JAX` `PyTorch` `NumPy/SciPy` · `FastAPI` `Redis` `Docker` `MLflow` `Hydra` `pytest` `GitHub Actions` · `AWS EC2/S3`

---

## Background

**MSc Applied Data Science**, Anglia Ruskin University, Cambridge — Merit. Major project 79%, highest in cohort.
**BSc Physics**, COMSATS University Islamabad. Quantum mechanics, mathematical modelling, experimental methods.

**Junior Data Scientist (contract)**, Digital Pulse 360 — Nov 2024 to Jan 2026. Client-facing ML delivery: requirements through model deployment, ETL pipelines, rapid prototyping.

Open to research-engineering and applied-ML roles in scientific computing, forecasting, and uncertainty quantification — particularly where a learned model is standing in for something that used to be simulated.
