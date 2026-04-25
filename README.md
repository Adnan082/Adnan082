<div align="center">

# Adnan Haider Cheema

### Bridging Predictive AI with Generative Simulation

*I design ML systems that predict — and simulation frameworks that stress-test those predictions before they reach production.*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/adnan-haider-cheema-)
[![Email](https://img.shields.io/badge/Email-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:adnancheema917@gmail.com)
[![Portfolio](https://img.shields.io/badge/Portfolio-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Adnan082)

</div>

---

## About

Data Scientist with a physics foundation and applied ML expertise across financial forecasting, healthcare, and aerospace engineering systems. I build end-to-end pipelines — from raw data through modelling to explainability and deployment — and pair them with simulation frameworks for synthetic data generation, risk quantification, and decision support under uncertainty. Currently completing an MSc in Applied Data Science at Anglia Ruskin University (Cambridge, UK).

---

## Technical Toolbox

| Domain | Stack |
|---|---|
| **Languages & Core** | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![SQL](https://img.shields.io/badge/SQL-4479A1?style=flat&logo=postgresql&logoColor=white) ![Git](https://img.shields.io/badge/Git-F05032?style=flat&logo=git&logoColor=white) ![LaTeX](https://img.shields.io/badge/LaTeX-008080?style=flat&logo=latex&logoColor=white) |
| **ML / Deep Learning** | ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white) ![XGBoost](https://img.shields.io/badge/XGBoost-1E90FF?style=flat) ![SHAP](https://img.shields.io/badge/SHAP-6C3483?style=flat) |
| **Simulation & Modelling** | ![Monte Carlo](https://img.shields.io/badge/Monte%20Carlo-2C3E50?style=flat) ![SimPy](https://img.shields.io/badge/SimPy-27AE60?style=flat) ![Synthetic Data](https://img.shields.io/badge/Synthetic%20Data%20Gen-8E44AD?style=flat) ![Agent Based](https://img.shields.io/badge/Agent--Based%20Modelling-E67E22?style=flat) ![DES](https://img.shields.io/badge/Discrete%20Event%20Sim-3498DB?style=flat) |
| **Data & Visualisation** | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat) ![Power BI](https://img.shields.io/badge/Power%20BI-F2C811?style=flat&logo=powerbi&logoColor=black) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white) |
| **Deployment & Ops** | ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white) ![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white) ![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat) |

---

## High-Impact Project Showcase

### ✈️ NASA C-MAPSS Fleet Health Monitor — Predictive Maintenance at Scale

> Architected a multi-agent AI system for predictive maintenance of 707 turbofan engines using the NASA run-to-failure dataset, combining deep learning forecasting with simulation-driven anomaly detection.

| Metric | Result |
|---|---|
| **RUL Prediction** | CNN-BiLSTM achieving MAE of **12.2 cycles** |
| **Anomaly Detection** | LSTM Autoencoder with per-condition adaptive thresholds |
| **Decision Engine** | Rule-based classifier spanning **5 priority levels** across the full fleet |
| **Interface** | Live Streamlit dashboard with real-time KPI cards and AI chat (Claude Haiku) |

**Business Value:** Enables condition-based maintenance scheduling, reducing unplanned downtime and extending engine service life through early degradation detection.

`Python` · `TensorFlow/Keras` · `LangChain` · `Streamlit` · `asyncio`

[![Repo](https://img.shields.io/badge/View_Repo-181717?style=flat&logo=github)](https://github.com/Adnan082/LINK) [![Live Demo](https://img.shields.io/badge/Live_Demo-FF4B4B?style=flat&logo=streamlit&logoColor=white)](#)

---

### 🔄 Customer Churn Prediction Pipeline — Revenue Retention Engine

> Engineered a production-grade binary classification system for telecom churn prediction, with full model interpretability via SHAP to surface actionable retention levers for business stakeholders.

| Metric | Result |
|---|---|
| **Best Model** | Random Forest — **F1: 0.911 · ROC-AUC: 0.985** |
| **Key Insight** | Customers with complaints churn at **83%** vs 10.1% without |
| **Validation** | GridSearchCV with 5-fold stratified cross-validation |
| **Explainability** | SHAP TreeExplainer & LinearExplainer for feature-level attribution |

**Business Value:** Identifies high-risk customers 30+ days before churn, enabling targeted intervention that directly protects recurring revenue.

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

**Business Value:** Supports early identification of at-risk patients, enabling preventive interventions that reduce long-term treatment costs and improve patient outcomes.

`Python` · `Scikit-Learn` · `Pandas` · `Matplotlib`

[![Repo](https://img.shields.io/badge/View_Repo-181717?style=flat&logo=github)](https://github.com/Adnan082/LINK)

---

## The Intersection: Simulation × AI

```
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│   SIMULATION        │     │   SYNTHETIC DATA     │     │   ML / AI MODEL     │
│                     │────▶│                      │────▶│                     │
│ Monte Carlo         │     │ Edge-case generation │     │ Trained on richer,  │
│ Agent-Based Models  │     │ Class balancing       │     │ stress-tested data  │
│ Discrete-Event Sim  │     │ Scenario modelling   │     │ Validated under     │
│                     │     │                      │     │ extreme conditions  │
└─────────────────────┘     └──────────────────────┘     └─────────────────────┘
```

My approach treats simulations not as standalone exercises, but as **upstream infrastructure for AI systems.** Monte Carlo methods generate synthetic failure scenarios to stress-test predictive maintenance models. Agent-based simulations produce edge-case customer behaviour data that improves churn classifiers on underrepresented segments. The result: models that are validated not just on historical data, but across the full distribution of plausible futures.

---

## Education

| Degree | Institution | Period |
|---|---|---|
| **MSc Applied Data Science** | Anglia Ruskin University, Cambridge, UK | Sep 2024 – Oct 2025 |
| **BSc Physics** | COMSATS University Islamabad | Sep 2018 – Mar 2023 |

**MSc Focus:** Machine learning, statistical modelling, Python programming, data engineering
**BSc Foundation:** Quantum mechanics, experimental methods, mathematical modelling, laboratory data analysis

---

## Publications

📝 **SIAM NEWS Blog** contributor (Dec 2022) — Scientific community engagement and technical communication

---

## Currently Building

- 🔬 MSc dissertation: customer churn prediction with ensemble methods + SHAP interpretability
- 🚀 Deployed ML applications with FastAPI and Docker
- 🤖 RAG pipelines and LLM-powered applications
- 🎲 Monte Carlo simulation frameworks for risk assessment and synthetic data generation

---

<div align="center">

*Open to mid-level Data Science and AI roles — particularly where simulation-driven decision support meets production ML.*

[![LinkedIn](https://img.shields.io/badge/Let's_Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/adnan-haider-cheema-)

</div>
