# Default Prediction of Commercial Real Estate Properties

## Project Overview
This project focuses on the **default prediction of commercial real estate (CRE) properties** using supervised machine learning techniques. The objective is to identify whether a loan associated with a commercial property is likely to **default or remain non-default**, based on a combination of financial ratios, property-level indicators, and macroeconomic variables.

The motivation for this work comes from the increasing importance of **data-driven credit risk assessment** in commercial lending. Compared to traditional rule-based or linear risk models, machine learning methods are capable of capturing complex, nonlinear relationships between explanatory variables and default behavior.

The project workflow follows a structured and reproducible machine learning pipeline, including data preprocessing, exploratory analysis, feature engineering, handling class imbalance, model training, and evaluation. While the study is inspired by existing academic research, the implementation, experimentation, and interpretation are carried out independently with a strong focus on practical learning and portfolio demonstration.

This repository is intended to serve both as an **academic reference** and a **data science portfolio project**, showcasing the application of machine learning techniques to a real-world financial risk problem.

---

## Project Motivation
Commercial real estate loans involve large capital investments and long maturities, making accurate default prediction essential for lenders, investors, and risk managers. Even small improvements in predictive accuracy can lead to better portfolio performance and reduced financial losses.

Traditional credit evaluation approaches often rely on fixed thresholds and limited assumptions, which may not fully reflect the underlying risk dynamics. This project explores how machine learning models can improve default prediction by leveraging historical loan data and multiple financial indicators simultaneously.

**The primary goals of this project are:**
- To understand the key factors influencing default in commercial real estate loans
- To compare the performance of different machine learning algorithms
- To study the impact of class imbalance and sampling techniques on model performance
- To evaluate models using appropriate metrics beyond simple accuracy

---

## Dataset Description
The dataset consists of **4,793 commercial real estate loans** described by **17 explanatory variables** and one binary target variable indicating loan repayment status.

The original data was collected from multiple sources, including:
- National Council of Real Estate Investment Fiduciaries (NCREIF)
- Trepp
- Federal Reserve Bank of St. Louis

For this project, two MATLAB files were provided:
- One containing **defaulted loans**
- One containing **non-defaulted loans**

These datasets were merged into a single dataframe and used for all subsequent analyses.

---

## Project Structure & Workflow
The project follows a systematic, end-to-end machine learning workflow:

1. **Data Loading and Integration**  
   Multiple datasets containing defaulted and non-defaulted loans are merged into a unified analytical dataset.

2. **Exploratory Data Analysis (EDA)**  
   Statistical summaries, visualizations, correlation matrices, and boxplots are used to understand variable distributions and differences between default and non-default loans.

3. **Feature Engineering and Selection**  
   Highly skewed variables are log-transformed, and correlated features are identified and reduced using feature selection techniques such as Recursive Feature Elimination (RFE).

4. **Handling Class Imbalance**  
   Different strategies including class weighting, SMOTE oversampling, and random undersampling are applied and compared.

5. **Model Training and Evaluation**  
   Logistic Regression, Support Vector Machines (SVM), and Random Forest models are trained and evaluated using ROC–AUC and other classification metrics.

---

## Model Evaluation
Because the dataset is imbalanced, model performance is evaluated using metrics that provide a balanced view of both classes:
- ROC–AUC (primary optimization metric)
- Precision, Recall, and F1-score
- Macro-averaged metrics
- Confusion matrices

Multiple train–test splits are used to assess model stability and robustness.

---

## Key Findings
- Financial ratios such as **DSCR, LTV, NOI Ratio, and PV Ratio** show strong predictive power
- Log-transformations help normalize skewed variables and improve model performance
- Models trained without addressing class imbalance perform poorly
- Class weighting and SMOTE generally outperform pure undersampling
- Logistic Regression remains competitive when properly tuned

---



## Conclusion
This project demonstrates that machine learning techniques can significantly enhance default prediction for commercial real estate loans. While no single model dominates in all scenarios, the experiments highlight important trade-offs between interpretability, performance, and data efficiency.

Overall, **Logistic Regression with class weighting or SMOTE** offers a strong balance between predictive power and explainability, making it well-suited for real-world credit risk applications.

---

## Tools and Libraries Used
The project is implemented entirely in Python using commonly adopted data science and machine learning libraries:

- **Python** – Core programming language
- **Pandas, NumPy** – Data manipulation and numerical computation
- **Matplotlib, Seaborn** – Data visualization
- **scikit-learn** – Machine learning models, preprocessing, and evaluation
- **imbalanced-learn (SMOTE)** – Techniques for handling imbalanced datasets
- **Jupyter Notebook / Google Colab** – Interactive development environment

---

## Results Summary

- **Logistic Regression (class-weighted and SMOTE):**  
  ROC–AUC values consistently ranged between **0.65 and 0.71**, showing stable performance across multiple random splits. Both approaches handled class imbalance effectively, with no major difference between them.

- **Logistic Regression (undersampling):**  
  Performance was slightly weaker, with ROC–AUC values mostly between **0.63 and 0.69**, likely due to information loss from removing many non-default observations.

- **Support Vector Machines (SVM):**  
  Class-weighted and SMOTE-based SVM models achieved ROC–AUC scores in the range of **0.67 to 0.71**, occasionally outperforming Logistic Regression but requiring careful hyperparameter tuning.

Overall, addressing class imbalance was essential. **Class weighting and SMOTE outperformed undersampling**, while Logistic Regression showed lower variance and more stable behavior across runs.

------|------------------------|-----------------|----------------|
| Logistic Regression | Class-weighted | ~0.65 – 0.70 | Stable and interpretable, strong baseline |
| Logistic Regression | SMOTE (Train only) | ~0.65 – 0.71 | Slight improvement in recall for defaults |
| Logistic Regression | Undersampling | ~0.63 – 0.69 | Performance drops due to data loss |
| Support Vector Machine | Class-weighted | ~0.67 – 0.70 | Captures nonlinear patterns well |
| Support Vector Machine | SMOTE | ~0.66 – 0.71 | Comparable to weighted SVM |
| Random Forest | Various strategies | Competitive | Robust but less interpretable |


---


### Key Observations
- **Class weighting and SMOTE** consistently outperformed pure undersampling
- Logistic Regression showed **low variance across random splits**, indicating stability
- SVM achieved slightly higher peak AUC but required careful tuning
- Undersampling reduced model performance due to loss of informative observations

