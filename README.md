# Predicting Extended ICU Stays using LSTM Networks & Streamlit

An end-to-end deep learning web application designed to predict whether a patient will experience an extended ICU stay (defined as exceeding the 75th percentile) based on vital sign metrics and demographic data.

**GitHub Repository:** [View Source Code](https://github.com/avamungaray/icu_length_of_stay)  
**Live Demo:** [View Streamlit App](https://your-app-name.streamlit.app)

---

## Executive Summary & Project Overview

Managing ICU bed capacity and hospital resources efficiently is a critical operational challenge. Knowing which patients are at high risk for an **extended stay (defined here as the 75th percentile threshold)** allows hospital administrators to optimize bed turnover, manage medical resource allocation, and reduce systemic bottlenecks. 

This project bridges advanced machine learning with real-world clinical utility by combining a custom Long Short-Term Memory (LSTM) binary classification neural network with an interactive web dashboard that outputs the exact probability of an extended stay.

---

## Key Features & Architecture

* **Sequential Deep Learning:** Built using an **LSTM network** to capture temporal patterns and trends in patient vital signs over time to classify risk.
* **Model Interpretability:** Incorporated feature analysis (such as SHAP values) to provide transparency into how vital signs drive specific risk predictions.
* **Tiered Risk Dashboard:** A fully interactive **Streamlit** web application that takes model probability outputs and translates them into actionable risk categories (Optimized, Borderline, High) for rapid clinical interpretation.

---

## The Data & Preprocessing Pipeline

* **Data Description:** Utilizes a clinical dataset consisting of patient demographics and sequential time-series vital sign measurements.
* **Target Definition:** The target variable is binary (1 for extended stay, 0 for standard stay), engineered by calculating the 75th percentile of length of stay across the dataset.
* **Data Hygiene & Engineering:** 
  * Handled missing clinical data points and irregular time intervals through targeted interpolation and forward-filling strategies.
  * Applied robust scaling and tensor reshaping to structure data correctly for sequential LSTM input constraints.
* **Privacy Disclaimer:** Due to patient confidentiality and institutional data constraints, raw datasets and proprietary processed dataframes have been excluded from this public repository.

---

## Modeling & Evaluation

* **Model Selection:** An LSTM architecture was selected to capture temporal dependencies and trends in patient telemetry over time, feeding into a binary classification head.
* **Evaluation Metrics:** Evaluated using classification metrics such as **ROC-AUC, Precision, Recall, and F1-Score** to ensure the model balances false positives and false negatives effectively in a clinical context.

---

## Tech Stack

* **Language:** Python
* **Deep Learning:** TensorFlow / Keras (LSTM architecture)
* **Data Manipulation & Analysis:** Pandas, NumPy, Scikit-Learn
* **Data Visualization:** Matplotlib, Seaborn, Streamlit UI components
* **App Deployment:** Streamlit Community Cloud
* **Version Control:** Git & GitHub

---

## 6. Repository Structure

```text
icu-length-of-stay/
│
├── data/
│   └── processed/                 # Cleaned dataframes (excluded from public repo for privacy)
├── models/
│   ├── lstm_model.h5              # Trained LSTM neural network weights
│   └── preprocessor.pkl           # Fitted scalers / preprocessors for live inference
├── notebooks/
│   └── 01_eda_and_modeling.ipynb  # Jupyter notebooks for EDA and experimentation
├── app.py                         # Main Streamlit application script
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```

---

## Local Installation & Setup Guide

If you want to run this project on your local machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/icu-length-of-stay.git](https://github.com/your-username/icu-length-of-stay.git)
   cd icu-length-of-stay
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

---

## Future Improvements & Next Steps

* **Threshold Sensitivity Analysis:** Experiment with alternative duration thresholds (such as the 80th or 90th percentiles) to evaluate model performance and stability under stricter definitions of an "extended stay."
* **Advanced Architecture Exploration:** Implement attention mechanisms or test alternative sequence models (such as Temporal Convolutional Networks or Transformers) to compare temporal pattern capture against the current LSTM baseline.
* **Temporal Interpretability:** Expand model explainability by integrating temporal SHAP values to visualize precisely when a patient's vital sign trajectory tipped the probability score toward an extended stay.
* **Multimodal Integration:** Explore incorporating unstructured clinical text data (such as physician or nurse shift notes) alongside numerical telemetry to further boost prediction accuracy.

---

## Contact & Acknowledgements

* **Author:** Ava Mungaray - [LinkedIn](https://www.linkedin.com/in/avamungaray/) - mungarayava@gmail.com
* **Credits:** Project utilizes MIMIC-IV publically available dataset. Access to dataset was gained after completing an ethics course. Project was completed as a capstone for Eastern University's Data Science Master's Program.