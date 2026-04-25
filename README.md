<div align="center">

# Adnan Haider Cheema

### Bridging Physics, Simulation & Predictive AI

*Physics-trained data scientist who solves real-world problems by fusing first-principles modelling with deep learning — from forecasting geomagnetic storms using 8.4M NASA observations to predicting turbofan engine failure across 707-unit fleets.*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/adnan-haider-cheema-)
[![Email](https://img.shields.io/badge/Email-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:adnancheema917@gmail.com)
[![GitHub](https://img.shields.io/badge/Portfolio-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Adnan082)

</div>

---

## About

MSc Applied Data Science graduate (Merit; highest cohort project score 79%) with a BSc in Physics and hands-on experience as a Junior Data Scientist. I specialise in end-to-end ML systems — Transformers, LSTMs, reinforcement learning, real-time dashboards, and production data pipelines — and leverage my physics foundation to build simulation frameworks that model complex system dynamics, generate synthetic data, and support decision-making under uncertainty. My work sits at the intersection of physics-informed modelling, predictive analytics, and deployed AI, with domain experience across space weather, aerospace engineering, healthcare, and telecoms.

---

## Technical Toolbox

| Domain | Stack |
|---|---|
| **Languages & Core** | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![SQL](https://img.shields.io/badge/SQL-4479A1?style=flat&logo=postgresql&logoColor=white) ![Git](https://img.shields.io/badge/Git-F05032?style=flat&logo=git&logoColor=white) ![LaTeX](https://img.shields.io/badge/LaTeX-008080?style=flat&logo=latex&logoColor=white) |
| **ML / Deep Learning** | ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white) ![Transformers](https://img.shields.io/badge/Transformers-FFD43B?style=flat) ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![XGBoost](https://img.shields.io/badge/XGBoost-1E90FF?style=flat) ![SHAP](https://img.shields.io/badge/SHAP-6C3483?style=flat) |
| **Simulation & Modelling** | ![Physics-Informed ML](https://img.shields.io/badge/Physics--Informed%20ML-C0392B?style=flat) ![Reinforcement Learning](https://img.shields.io/badge/Reinforcement%20Learning-16A085?style=flat) ![Monte Carlo](https://img.shields.io/badge/Monte%20Carlo-2C3E50?style=flat) ![ODE Solvers](https://img.shields.io/badge/ODE%20Solvers-8E44AD?style=flat) ![Synthetic Data](https://img.shields.io/badge/Synthetic%20Data%20Gen-E67E22?style=flat) |
| **Data & Visualisation** | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat) ![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat) ![Power BI](https://img.shields.io/badge/Power%20BI-F2C811?style=flat&logo=powerbi&logoColor=black) |
| **Deployment & MLOps** | ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white) ![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white) ![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat&logo=mlflow&logoColor=white) ![Redis](https://img.shields.io/badge/Redis-DC382D?style=flat&logo=redis&logoColor=white) ![Prometheus](https://img.shields.io/badge/Prometheus-E6522C?style=flat&logo=prometheus&logoColor=white) |
| **AI & LLM Tooling** | ![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat) ![Claude API](https://img.shields.io/badge/Claude%20API-D4A574?style=flat) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white) ![Dash](https://img.shields.io/badge/Dash-008DE4?style=flat&logo=plotly&logoColor=white) |

---

## Experience

**Junior Data Scientist (Contract)** · Digital Pulse 360 (Remote) · Nov 2024 – Jan 2026
- Delivered multiple end-to-end data science projects for various clients, managing the full lifecycle from requirements gathering to final model deployment.
- Rapidly prototyped custom ML solutions and ETL pipelines, ensuring fast turnaround times for ad-hoc client requests.

---

## High-Impact Project Showcase

### 🌌 Hybrid Multi-Agent Solar Wind Dst Prediction System

> Engineered a real-time 5-agent ML pipeline fusing physics-based ODE modelling, deep learning, and reinforcement learning to forecast geomagnetic storms (Dst index) from 8.4M+ NASA OMNI observations — the only known system combining all three paradigms for Dst prediction.

| Component | Result |
|---|---|
| **Anomaly Detection** | Transformer Autoencoder (239K params) achieving **237× reconstruction error contrast** on extreme storms vs quiet-time baseline |
| **Residual Correction** | BiLSTM corrector (546K params) reaching **6.50 nT RMSE**, beating the Burton ODE physics baseline by **48–73%** across storm classes |
| **RL Optimisation** | Actor-Critic agent with 20K replay buffer learning optimal physics/ML blend weights online |
| **Production Pipeline** | Redis pub/sub, FastAPI (REST + WebSocket), Prometheus monitoring, real-time Dash dashboard with tiered storm alerts (GREEN/YELLOW/RED) |

**Why It Matters:** Geomagnetic storms threaten satellite electronics, power grids, and GPS accuracy. This system provides graduated early warning before Dst depression onset, enabling operators to take protective action.

`Python` · `PyTorch` · `Transformers` · `BiLSTM` · `Reinforcement Learning` · `FastAPI` · `Redis` · `Dash` · `MLflow` · `Docker`

[![Repo](https://img.shields.io/badge/View_Repo-181717?style=flat&logo=github)](https://github.com/Adnan082/Hybrid-Model-Solar-winds-Prediction)

---

### ✈️ TurbineAgent — NASA C-MAPSS Fleet Health Monitor

> Architected a multi-agent AI system for predictive maintenance of 707 turbofan engines, applying physics-informed feature engineering from thermodynamic degradation patterns across all 4 fault-mode sub-datasets in the NASA C-MAPSS run-to-failure benchmark.

| Component | Result |
|---|---|
| **RUL Prediction** | CNN-BiLSTM achieving **MAE 12.2 cycles**, outperforming published LSTM (16.14) and CNN (18.45) baselines |
| **Anomaly Detection** | LSTM Autoencoder with **39.6% near-failure capture rate** using per-operating-condition adaptive thresholds across 6 KMeans-clustered flight regimes |
| **Uncertainty Quantification** | Split conformal prediction producing calibrated **90% coverage intervals** (RUL ± 33.3 cycles), implemented from scratch |
| **Multi-Agent Orchestration** | 4 specialised AI agents via asyncio event bus + LangChain + Claude Haiku, automating fleet-wide triage into 5 priority tiers in **< 60 seconds** |
| **Production Stack** | SHAP explainability, MLflow tracking, FastAPI (4 endpoints), Streamlit dashboard (5 pages), pytest suite, full Docker containerisation |

**Why It Matters:** Enables condition-based maintenance scheduling across an entire fleet, reducing unplanned downtime and extending engine service life through early degradation detection with calibrated confidence bounds.

`Python` · `PyTorch` · `FastAPI` · `Docker` · `MLflow` · `Streamlit` · `Claude API` · `LangChain`

[![Repo](https://img.shields.io/badge/View_Repo-181717?style=flat&logo=github)](https://github.com/Adnan082)

---

### 🔄 Customer Churn Prediction Pipeline — Revenue Retention Engine

> Engineered a production-grade binary classification system for telecom churn prediction, with full model interpretability via SHAP to surface actionable retention levers for business stakeholders.

| Metric | Result |
|---|---|
| **Best Model** | Random Forest — **F1: 0.911 · ROC-AUC: 0.985** |
| **Key Insight** | Customers with complaints churn at **83%** vs 10.1% without |
| **Validation** | GridSearchCV with 5-fold stratified cross-validation |
| **Explainability** | SHAP TreeExplainer & LinearExplainer for feature-level attribution |

**Why It Matters:** Identifies high-risk customers before churn, enabling targeted intervention that directly protects recurring revenue.

`Python` · `Scikit-Learn` · `XGBoost` · `SHAP` · `Pandas` · `Matplotlib`

[![Repo](https://img.shields.io/badge/View_Repo-181717?style=flat&logo=github)](https://github.com/Adnan082/LINK)

---

### 🏥 Type II Diabetes Risk Prediction — Clinical Decision Support

> Developed a supervised ML pipeline for early diabetes risk stratification from clinical health indicators, designed to support screening prioritisation in healthcare settings.

| Metric | Result |
|---|---|
| **Accuracy** | **80%+** using optimised classification pipeline |
| **Preprocessing** | Missing value imputation, normalisation, outlier treatment |
| **Feature Engineering** | Variable transformations and selection based on clinical relevance |

**Why It Matters:** Supports early identification of at-risk patients, enabling preventive interventions that reduce long-term treatment costs and improve outcomes.

`Python` · `Scikit-Learn` · `Pandas` · `Matplotlib`

[![Repo](https://img.shields.io/badge/View_Repo-181717?style=flat&logo=github)](https://github.com/Adnan082/LINK)

---

## The Intersection: Physics × Simulation × AI

```
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│   PHYSICS           │     │   SIMULATION         │     │   ML / AI MODEL     │
│                     │────▶│                      │────▶│                     │
│ Burton ODE for Dst  │     │ Monte Carlo sampling │     │ RL agent learns     │
│ Thermodynamic       │     │ ODE-based baselines  │     │ optimal blend of    │
│ degradation curves  │     │ Synthetic scenarios  │     │ physics + ML        │
│ Conservation laws   │     │ Parameter sweeps     │     │ predictions online  │
└─────────────────────┘     └──────────────────────┘     └─────────────────────┘
```

My physics degree isn't a backstory — it's actively embedded in how I build ML systems. The Solar Wind project uses the Burton ODE (a physics differential equation) as a baseline, then trains a BiLSTM to learn residual corrections the physics can't capture, with an RL agent blending both in real time. The NASA project applies thermodynamic degradation physics to engineer sensor features before the CNN-BiLSTM ever sees the data. This is the pattern: **physics defines the problem structure, simulation generates the scenarios, and ML learns the patterns physics alone can't express.**

---

## Education

| Degree | Institution | Period |
|---|---|---|
| **MSc Applied Data Science (Merit)** | Anglia Ruskin University, Cambridge, UK | Sep 2024 – Oct 2025 |
| **BSc Physics (2:1 equivalent)** | COMSATS University Islamabad | Sep 2018 – Mar 2023 |

**MSc Highlight:** Major project scored 79% — highest in cohort. Modules: Machine Learning, Statistical Methods, Big Data Analytics, Applied AI.
**BSc Foundation:** Quantum mechanics, experimental methods, mathematical modelling, laboratory data analysis.

---

## Publications

📝 **SIAM NEWS Blog** contributor (Dec 2022) — Scientific community engagement and technical communication

---

## Currently Building

- 🌌 Expanding the Solar Wind system with longer forecast horizons and ensemble storm classification
- 🚀 Production ML applications with FastAPI, Docker, and MLflow pipelines
- 🤖 RAG pipelines and LLM-powered applications with LangChain
- 🎲 Physics-informed simulation frameworks for synthetic data generation and system modelling

---

<div align="center">

*Open to Data Science and AI roles — particularly where physics-informed modelling meets production ML systems.*

**UK Graduate Route visa · Eligible for Skilled Worker sponsorship**

[![LinkedIn](https://img.shields.io/badge/Let's_Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/adnan-haider-cheema-)

</div>
