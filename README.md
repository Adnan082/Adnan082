# Adnan Haider Cheema

**Machine Learning Engineer | Reliable Inference & Distributed Training**

I build ML systems with explicit data contracts, calibrated predictions, and reproducible evaluation. My work spans fraud decision services, PyTorch turbine prognostics, JAX neural surrogates, and bandwidth-controlled distributed training. I use FastAPI, Docker, MLflow, and CI to connect model development with serving, monitoring, and measurable failure analysis.

Cambridge, UK · [LinkedIn](https://www.linkedin.com/in/adnan-haider-cheema/) · [Email](mailto:adnancheema917@gmail.com)

## Selected engineering work

| Project | Architecture and evidence | Current boundary |
| --- | --- | --- |
| [Onboarding fraud triage](https://github.com/Adnan082/onboarding-fraud-triage) | LightGBM, FastAPI, data contracts, conformal decisions, label-free monitoring, MLflow and Docker. Reported fraud non-approval: 58.1% / 59.1% on test months 6 / 7. | Synthetic BAF data; the genuine-applicant verification target failed under drift. A portfolio service, not a bank deployment. |
| [Distributed training measurements](https://github.com/Adnan082/diloco-measured) | PyTorch DiLoCo, DDP and LocalSGD on four AWS GPU nodes, with independently verified bandwidth shaping and committed measurement records. | 30.8M parameters. Better utilization did not establish better convergence; hardware reproduction uses experiment-specific scripts. |
| [Turbine remaining-life prediction](https://github.com/Adnan082/NASA_Turbine_Engine) | CNN-BiLSTM, anomaly detection and conformal intervals. The v2 experiment reports RMSE 14.08 cycles and 92.8% marginal coverage on 707 test engines. | v2 is evaluated separately from the default v1 application. Coverage falls to 82.5% in the 61–100-cycle band. |
| [NODA: neural data assimilation](https://github.com/Adnan082/NODA) | JAX/Equinox Fourier Neural Operators inside a localized ensemble Kalman filter, checked by an independent PDE residual. Reported regime-shift detection AUC: 0.971. | Synthetic turbulent-flow experiments. Multi-surrogate ensembles did not fix overconfidence; recovery after an unknown regime shift remains open. |

Each repository contains the experimental setup, result provenance, reproduction instructions, and limitations. Reported results are scoped to those experiments.

## Engineering approach

- Validate inputs before scoring and separate training, calibration, and evaluation data.
- Measure uncertainty and subgroup failure, alongside aggregate predictive performance.
- Keep model logic separate from APIs, orchestration, and reporting.
- Record configurations, seeds, artifact versions, and unsuccessful experiments.
- Use tests and CI to check implementation behavior; distinguish these checks from model-quality validation.

**Core tools:** Python · SQL/DuckDB · PyTorch · JAX/Equinox · scikit-learn · LightGBM · FastAPI · Docker · MLflow · Hydra · pytest · GitHub Actions · AWS EC2/S3

## Supporting work

- [POLARIS](https://github.com/Adnan082/POLARIS---Polarimetric-Observation-for-Learned-All-weather-Road-surface-Identification-of-States): RGB/polarimetric ConvNeXt fusion and Stokes preprocessing. Evaluation results pending.
- [Solar-wind modeling](https://github.com/Adnan082/Hybrid_Model_Solar_winds_Prediction): physics ODE, learned residuals and Redis event processing. Earlier accuracy claims withdrawn pending leakage-free evaluation.
- [Local arXiv search](https://github.com/Adnan082/RAG): local embedding retrieval with ChromaDB and a Streamlit interface.
- [Real/fake face classification](https://github.com/Adnan082/face-detection): EfficientNet transfer learning on a StyleGAN3 dataset.
- [Urban heat mapping](https://github.com/Adnan082/uhi-mitigation-mapper): satellite and census data integration for site prioritization.

## Background

**MSc Applied Data Science**, Anglia Ruskin University, Cambridge (2024–2025): Merit; major project Distinction (79%).  
**BSc Physics**, COMSATS University Islamabad (2018–2023).

Freelance Machine Learning Engineer, Upwork (Jan 2026–present). Previously Junior Data Scientist, Digital Pulse 360 (part-time contract, Nov 2024–Jan 2026), and Data Analyst, Forward Sports (Jan–Aug 2024).

I use AI coding tools, design the experiments, and review the resulting code and evidence.

Seeking Machine Learning / MLOps Engineering roles in reliable inference, scientific ML, and ML systems.
