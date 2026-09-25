# Adnan Haider Cheema

**Forecasts and decisions that say how sure they are, and the checks that catch them when they're wrong.**

I'm a data scientist in Cambridge, UK, with an MSc in Applied Data Science and a BSc in Physics. Physics left me with one habit: a number is only useful if you also know when not to trust it. My repositories build that in:
- data contracts that reject bad input before anything is scored
- calibrated intervals instead of bare point estimates
- drift monitors that work without labels
- write-ups that keep failures as findings

Cambridge, UK · [LinkedIn](https://www.linkedin.com/in/adnan-haider-cheema/) · [adnancheema917@gmail.com](mailto:adnancheema917@gmail.com)

---

## Selected work

### Onboarding fraud triage: approve, review or verify, with a guarantee that is tested

**[`Adnan082/onboarding-fraud-triage`](https://github.com/Adnan082/onboarding-fraud-triage)** · LightGBM · conformal prediction · pandera · SQL (DuckDB) · FastAPI · MLflow · Docker

It scores one million online bank-account applications from the NeurIPS 2022 BAF dataset. Each one goes to one of three outcomes: approve, send for human review, or ask for extra verification. The routing comes with a stated, tested limit on how much fraud gets through.

- **Data first.** Every application is checked against a frozen data contract, so a broken upstream feed is rejected instead of scored. The data is split by month only:
  - train on months 0–4
  - calibrate on month 5, cut three ways
  - test on months 6 and 7, reported separately
- **Decisions, not scores.** Calibrated probabilities feed label-conditional conformal thresholds. The system promises two things: stop at least 55% of fraud, and send no more than 1% of genuine applicants for extra checks.
- **Monitoring without labels.** Three checks watch for drift: PSI computed in SQL, a domain classifier and a conformal-rate test. Their thresholds are set on 200 clean windows.

| Result | Month 6 | Month 7 |
|---|---:|---:|
| ROC-AUC | 0.891 | 0.895 |
| Fraud stopped (promise: at least 55%) | 58.1% | 59.1% |
| Genuine applicants sent to verify (promise: at most 1%) | 1.6% | 1.3% |

**The fraud promise held, and the genuine-applicant promise broke.** Months 6 and 7 drifted away from the calibration month. The monitor flagged that drift without any labels.

The monitor also caught all 3 silent data faults I injected, each within one 4,000-application window, with 0 false alerts on 200 clean windows. The project has 315 tests, 95% coverage on the core modules, and a median scoring time of 6.4 ms.

---

### TurbineAgent: remaining-life forecasts with intervals checked on unseen engines

**[`Adnan082/NASA_Turbine_Engine`](https://github.com/Adnan082/NASA_Turbine_Engine)** · PyTorch · FastAPI · Streamlit · MLflow · Docker · pytest + CI

It predicts remaining useful life (RUL) for the 707 test engines in NASA's C-MAPSS run-to-failure data. One CNN-BiLSTM covers all four sub-datasets, and it is served through FastAPI and a Streamlit dashboard.

A maintenance planner can't act on "RMSE 14 cycles". They can act on "40 cycles left, and the true value is between 12 and 68, with 90% coverage".

| | v1 | v2 |
|---|---:|---:|
| RUL RMSE, 707 test engines | 17.58 | **14.08** |
| RUL MAE | 12.20 | **9.88** |
| Conformal interval | ±33.3, calibrated on test data | **±27.8**, calibrated on held-out training engines |
| Coverage (target 90%) | 92.9% on 565 engines | **92.8% on all 707** |
| Anomaly flags: near failure / healthy | 39.6% / 19.5% | **56.6% / 5.4%** |

**What was wrong with v1.** It was calibrated on part of its own test set, and it trained for 100 epochs with no validation data. v2 splits the data by engine, stops training early, and calibrates on engines it never trained on.

**Where v2 is still weak:**
- FD003 got slightly worse (RMSE 14.52 → 14.97).
- Coverage falls to 82.5% for engines 61–100 cycles from failure, because one pooled quantile can't fit every band.

Every number reproduces from a clean clone, and CI re-runs the tests on every push.

---

### NODA: when a neural surrogate makes an ensemble filter confidently wrong

**[`Adnan082/NODA`](https://github.com/Adnan082/NODA)** · JAX · Equinox · jax-cfd · Hydra

NODA estimates a 2D turbulent vorticity field (128×128) in real time from about 200 noisy sensors, covering roughly 1.2% of the field. It uses an ensemble Kalman filter whose forecast model is a Fourier Neural Operator instead of a numerical solver.

**The problem.** The filter's only measure of uncertainty is how much its ensemble members disagree. When every member passes through the same network weights, any error the network makes is shared by all of them, so it never shows up as disagreement. The filter reports itself as trustworthy while being badly wrong.

| Config | Forward model | Spread–skill (1.0 = honest) | RMSE |
|---|---|---:|---:|
| A | Numerical solver (control) | 0.998 | 0.116 |
| B | One trained FNO | 0.207 | 1.016 |
| C | One FNO + tuned inflation | 1.251 | 1.005 |
| D | Five independently trained FNOs | 0.180 | 1.437 |

**Two standard fixes failed, and both results are kept.**
- Inflation (C) buys a better-looking calibration number without recovering any accuracy.
- A multi-model ensemble (D) stays overconfident, and is slightly less accurate than a single network.

**What worked.** A physics check from outside the model: the PDE residual of the assimilated state. It never touches the network's weights, so it can't share the network's blind spot. It detects an unseen regime shift at **ROC AUC 0.971**.

Detection is solved. Recovering accuracy after the shift is still open.

---

### POLARIS: polarimetric imaging for road-surface state *(in progress)*

**[`Adnan082/POLARIS`](https://github.com/Adnan082/POLARIS---Polarimetric-Observation-for-Learned-All-weather-Road-surface-Identification-of-States)** · PyTorch · AWS

POLARIS tests whether polarisation inputs beat RGB-only baselines at classifying road surfaces as dry, damp, wet, slush or snow, using the [PRISM dataset](https://huggingface.co/datasets/NeurIPS-2026-PRISM/PRISM-Dataset).

**Built so far:**
- the precompute pipeline for the polarisation features (Stokes parameters)
- session-level splits, so no data leaks across time
- three model variants: RGB only, polarisation only, and both combined
- training on AWS spot instances that survives interruptions

**Results:** none yet, so none are reported.

---

## Also on this profile

| Repository | What it is | Note |
|---|---|---|
| [`Hybrid_Model_Solar_winds_Prediction`](https://github.com/Adnan082/Hybrid_Model_Solar_winds_Prediction) | Dst storm-index forecasting: Burton physics ODE + learned residual + RL blending, on live NOAA feeds | **Evaluation under revision.** A re-check found validation leakage, so earlier accuracy numbers are withdrawn. |
| [`RAG`](https://github.com/Adnan082/RAG) | Fully local semantic search over 50k arXiv physics abstracts | ChromaDB, ONNX, no API keys, offline at query time |
| [`uhi-mitigation-mapper`](https://github.com/Adnan082/uhi-mitigation-mapper) | Tree-planting site prioritisation from Landsat surface temperature, NLCD land cover and Census vulnerability | Geospatial pipeline that gives planners a ranked action list |
| [`face-detection`](https://github.com/Adnan082/face-detection) | Classifies StyleGAN3-generated vs real faces with a fine-tuned EfficientNet-B0 | 97.8% accuracy / 0.9970 AUC on a 20k balanced split |
| [`churn_prediction`](https://github.com/Adnan082/churn_prediction) | Telecom churn classification with SHAP attribution | Small public UCI dataset; treat the high AUC with suspicion |

---

## How I work

- **Check the data before the model.** Frozen data contracts, and splits by time or by entity. Never random row splits on data that is ordered in time.
- **Uncertainty is part of the answer.** Calibrated probabilities, conformal intervals, spread–skill ratios. A point estimate with no interval is hard to act on.
- **Break the metric down.** Report it per month, per remaining-life band and per age band. An aggregate score hides failure exactly where it's expensive.
- **Negative results are kept.** The broken fraud guarantee, NODA's config D and TurbineAgent's FD003 regression are all in the READMEs.
- **Reproducible from a clean clone,** checked by CI.
- **AI in the loop, with me in charge.** I build with Claude Code every day. I design the experiments and check what comes back.

`Python` `SQL (DuckDB)` `pandas` `scikit-learn` `LightGBM` `PyTorch` `JAX` · `pandera` `FastAPI` `Docker` `MLflow` `Hydra` `pytest` `GitHub Actions` · `AWS EC2/S3` · `Claude Code`

---

## Background

**MSc Applied Data Science**, Anglia Ruskin University, Cambridge (2024–2025): Merit, with a Distinction (79%) on the major project.
**BSc Physics**, COMSATS University Islamabad (2018–2023).

**Freelance Machine Learning Engineer**, Upwork, Jan 2026 to present.
**Junior Data Scientist** (part-time contract), Digital Pulse 360, Nov 2024 to Jan 2026.
**Data Analyst**, Forward Sports (football manufacturer for Adidas), Jan to Aug 2024.

Looking for data science roles where forecasts, metrics and decisions have to be right, and have to say how sure they are.
