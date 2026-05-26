 Diabetes Risk Prediction
A machine learning project that predicts whether a patient has diabetes using the Pima Indians Diabetes Dataset. The pipeline covers data preprocessing, exploratory data analysis (EDA), model training, hyperparameter tuning, threshold optimization, and a GUI-based prediction tool.

Problem Statement
Predict the likelihood of diabetes in a patient (binary classification):

0 → No Diabetes
1 → Diabetes (High Risk)
Dataset

Source: Pima Indians Diabetes Dataset
Samples: 768 patients
Features: 8 medical attributes
Target: Outcome (0 or 1)

Pipeline Overview
1. Problem Definition
   └─ Binary classification: Diabetic or Not

2. Data Collection
   └─ Load Pima Indians Diabetes CSV

3. Data Preprocessing
   ├─ Replace invalid zeros with NaN (Glucose, BP, SkinThickness, Insulin, BMI)
   └─ Impute missing values using Median Strategy

4. Exploratory Data Analysis (EDA)
   ├─ Outcome count plot
   ├─ Glucose & BMI distributions
   ├─ Correlation heatmap
   ├─ Glucose vs Outcome boxplot
   ├─ t-SNE visualization
   └─ UMAP visualization

5. Model Training
   ├─ Logistic Regression (baseline)
   └─ Random Forest (improved + tuned)

6. Model Evaluation
   ├─ Accuracy, Precision, Recall, F1 Score
   ├─ Confusion Matrix
   ├─ ROC-AUC Curve
   └─ Cross-validation (5-Fold)

 Models Used
Logistic Regression

Used as a baseline model
Trained on standardized features

Random Forest (Improved)

n_estimators=200, max_depth=7
class_weight='balanced' — handles class imbalance
Medical threshold of 0.40 applied (lowers false negatives)

Random Forest (Tuned + Threshold Optimized)

Hyperparameter tuning via GridSearchCV with StratifiedKFold
Threshold sweep from 0.25 to 0.60
Final threshold selected to maintain Recall ≥ 70% (medical priority)



 Note: In medical prediction, Recall is prioritized to minimize missed diabetes cases.

 GUI Prediction Tool
medical_gui_model.py provides an interactive desktop GUI where you can enter patient data and get an instant diabetes risk prediction.
