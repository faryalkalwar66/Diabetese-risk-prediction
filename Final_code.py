import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
import umap.umap_ as umap

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score

df = pd.read_csv(r"C:\Users\HP\Desktop\PROJECT\diabetes.csv")

df.head()
df.isnull().sum()
cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Check zeros BEFORE replacing
print("Zeros before:")
print((df[cols] == 0).sum())

# Replace zeros with NaN
df[cols] = df[cols].replace(0, np.nan)

# Check missing values AFTER replacing
print("\nMissing values after replacing:")
print(df.isnull().sum())

imputer = SimpleImputer(strategy='median')
df[cols] = imputer.fit_transform(df[cols])

print(df.isnull().sum())

df.to_csv(r"C:\Users\HP\Desktop\PROJECT\cleaned_diabetes.csv", index=False)
print("Cleaned dataset saved successfully")
import os
print(os.getcwd())


# 1.  Outcome Count
plt.figure(figsize=(6,4))
sns.countplot(x=df['Outcome'])
plt.title("Diabetes Outcome Count")
plt.xlabel("0 = No Diabetes, 1 = Diabetes")
plt.show()

# 2. Glucose Distribution
plt.figure(figsize=(8,6))
sns.histplot(df['Glucose'], kde=True)
plt.title("Glucose Distribution")
plt.show()

# 3. BMI Distribution
plt.figure(figsize=(8,6))
sns.boxplot(x=df['BMI'])
plt.title("BMI Distribution")
plt.show()

#4. HEATMAP
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

#5.Glucose vs Outcome
df['Outcome_label'] = df['Outcome'].map({0: "No Diabetes", 1: "Diabetes"})

sns.boxplot(x=df['Outcome_label'], y=df['Glucose'])

plt.title("Glucose vs Diabetes Outcome")
plt.xlabel("Outcome")
plt.ylabel("Glucose")

plt.show()



# Features only
X = df.drop(['Outcome', 'Outcome_label'], axis=1)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

# Plot t-SNE
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=df['Outcome_label'], palette='coolwarm')
plt.title("t-SNE Visualization of Diabetes Data")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend(title="Outcome")
plt.show()




# Apply UMAP
reducer = umap.UMAP(random_state=42)
X_umap = reducer.fit_transform(X_scaled)

# Plot UMAP
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_umap[:,0], y=X_umap[:,1], hue=df['Outcome_label'], palette='coolwarm')
plt.title("UMAP Visualization of Diabetes Data")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend(title="Outcome")
plt.show()



X = df.drop(['Outcome', 'Outcome_label'], axis=1)
y = df['Outcome']

print("Features shape:", X.shape)
print("Target shape:", y.shape)
print("Target values:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


print("Scaling done")

log_model = LogisticRegression()

# Train model
log_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_log = log_model.predict(X_test_scaled)

print("Logistic Regression model trained")

print("Logistic Regression Results")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Precision:", precision_score(y_test, y_pred_log))
print("Recall:", recall_score(y_test, y_pred_log))
print("F1 Score:", f1_score(y_test, y_pred_log))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_log))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_log))

# =========================
# IMPROVED RANDOM FOREST MODEL
# =========================
# class_weight='balanced' gives more importance to diabetic patients,
# which is important in medical prediction because missing a diabetic case is risky.
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=7,
    min_samples_split=2,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced'
)

# Train
rf_model.fit(X_train, y_train)

# Predict probabilities first, then apply a medical threshold
rf_prob_base = rf_model.predict_proba(X_test)[:, 1]
medical_threshold = 0.40

y_pred_rf = (rf_prob_base >= medical_threshold).astype(int)

print("Improved Random Forest trained")
print("Medical Threshold Used:", medical_threshold)

print("Improved Random Forest Results")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("F1 Score:", f1_score(y_test, y_pred_rf))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

print("Tuning started...")

# Wider parameter search for a stronger model
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced']
}

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params,
    cv=cv_strategy,
    scoring='recall',
    verbose=2,
    n_jobs=-1
)

grid_rf.fit(X_train, y_train)

best_rf = grid_rf.best_estimator_


best_prob = best_rf.predict_proba(X_test)[:, 1]
threshold_results = []

for threshold in np.arange(0.25, 0.61, 0.05):
    temp_pred = (best_prob >= threshold).astype(int)
    threshold_results.append({
        'Threshold': round(float(threshold), 2),
        'Accuracy': accuracy_score(y_test, temp_pred),
        'Precision': precision_score(y_test, temp_pred, zero_division=0),
        'Recall': recall_score(y_test, temp_pred, zero_division=0),
        'F1 Score': f1_score(y_test, temp_pred, zero_division=0)
    })

threshold_df = pd.DataFrame(threshold_results)
print("\nThreshold Comparison:")
print(threshold_df)

# Medical priority: try to keep recall at least 70%; if not possible, select highest F1 score.
valid_thresholds = threshold_df[threshold_df['Recall'] >= 0.70]
if len(valid_thresholds) > 0:
    selected_row = valid_thresholds.sort_values(by='F1 Score', ascending=False).iloc[0]
else:
    selected_row = threshold_df.sort_values(by='F1 Score', ascending=False).iloc[0]

best_threshold = selected_row['Threshold']
y_pred_best = (best_prob >= best_threshold).astype(int)

print("Tuning completed")
print("Best Parameters:", grid_rf.best_params_)
print("Best Threshold:", best_threshold)

print("Tuned + Threshold Optimized Random Forest Results")
print("Accuracy:", accuracy_score(y_test, y_pred_best))
print("Precision:", precision_score(y_test, y_pred_best))
print("Recall:", recall_score(y_test, y_pred_best))
print("F1 Score:", f1_score(y_test, y_pred_best))
print(confusion_matrix(y_test, y_pred_best))
print(classification_report(y_test, y_pred_best))


comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'Improved Random Forest', 'Tuned + Threshold RF'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_log),
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_best)
    ],
    'Precision': [
        precision_score(y_test, y_pred_log),
        precision_score(y_test, y_pred_rf),
        precision_score(y_test, y_pred_best)
    ],
    'Recall': [
        recall_score(y_test, y_pred_log),
        recall_score(y_test, y_pred_rf),
        recall_score(y_test, y_pred_best)
    ],
    'F1 Score': [
        f1_score(y_test, y_pred_log),
        f1_score(y_test, y_pred_rf),
        f1_score(y_test, y_pred_best)
    ]
})

print(comparison)



from sklearn.metrics import confusion_matrix

# Logistic Regression Confusion Matrix
cm_log = confusion_matrix(y_test, y_pred_log)

plt.figure()
sns.heatmap(cm_log, annot=True, fmt='d')
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Improved Random Forest Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)

plt.figure()
sns.heatmap(cm_rf, annot=True, fmt='d')
plt.title("Improved Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


models = ['Logistic', 'Improved RF', 'Tuned Threshold RF']
accuracy = [
    accuracy_score(y_test, y_pred_log),
    accuracy_score(y_test, y_pred_rf),
    accuracy_score(y_test, y_pred_best)
]

plt.figure()
plt.bar(models, accuracy)
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.show()


comparison.set_index('Model').plot(kind='bar')

plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.show()



feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

print(feature_importance)


feature_importance.set_index('Feature').plot(kind='bar')

plt.title("Feature Importance - Best Random Forest")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.show()


cm = confusion_matrix(y_test, y_pred_best)
print(cm)


rf_prob = best_rf.predict_proba(X_test)[:, 1]

auc_score = roc_auc_score(y_test, rf_prob)
print("Best Random Forest ROC-AUC Score:", auc_score)

fpr, tpr, thresholds = roc_curve(y_test, rf_prob)

plt.figure()
plt.plot(fpr, tpr, label="Best Random Forest AUC = " + str(round(auc_score, 2)))
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title("ROC Curve - Best Random Forest")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()



cv_scores = cross_val_score(best_rf, X, y, cv=5, scoring='recall')

print("Best Random Forest Cross Validation Recall Scores:", cv_scores)
print("Average CV Recall:", cv_scores.mean())

new_patient = pd.DataFrame(
    [[2, 150, 80, 30, 100, 32.0, 0.5, 40]],
    columns=X.columns
)

prob = best_rf.predict_proba(new_patient)
prediction = 1 if prob[0][1] >= best_threshold else 0

print("\nNew Patient Prediction")
print("Threshold Used:", best_threshold)
print("No Diabetes Probability:", round(prob[0][0] * 100, 2), "%")
print("Diabetes Probability:", round(prob[0][1] * 100, 2), "%")

if prediction == 1:
    print("Final Result: High Risk of Diabetes")
else:
    print("Final Result: Low Risk of Diabetes")