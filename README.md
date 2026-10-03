# AI-Powered Predictive Maintenance System

End-to-end AI predictive maintenance system for NASA C-MAPSS jet engines. This project combines XGBoost and PyTorch for failure prediction, a Retrieval-Augmented Generation (RAG) pipeline for automated repair diagnostics, and an interactive Streamlit dashboard.

## 🚀 Project Pipeline

* **Data Engineering:** Raw dataset files are processed using Pandas to drop constant sensors, engineer rolling statistical features, and create Remaining Useful Life (RUL) labels, resulting in a clean 24-column dataset.
* **Dual-Model Architecture:** The system trains an XGBoost classifier for failure prediction (0.85 F1 Score) alongside a PyTorch regressor for cycle prediction, utilizing SHAP to explain feature importance.
* **Automated Logging:** Live sensor data runs through the inference models, and the predictions (machine status, confidence, and RUL) are automatically written to an SQLite database.
* **Smart Diagnostics via RAG:** A pipeline uses FAISS and Sentence Transformers to chunk and search PDF repair manuals, auto-generating plain-English maintenance instructions based on the AI's diagnosis.
* **Business Dashboard:** The backend connects to an interactive Streamlit and Plotly dashboard displaying fleet health, engine-level drill-downs, and estimated cost savings.

## 🗂️ Repository Structure

* `src/`: Core Python scripts containing modular logic for data preparation and model pipelines.
* `*.ipynb`: Step-by-step Jupyter notebooks detailing exploratory data analysis, data cleaning, and model evaluation with visual outputs.

## 🛠️ Tech Stack

* **Data Processing & ML:** Python, Pandas, XGBoost, PyTorch, Scikit-Learn, SHAP
* **NLP & Search:** FAISS, Sentence Transformers
* **Web, Visuals & DB:** Streamlit, Plotly, SQLite
*
