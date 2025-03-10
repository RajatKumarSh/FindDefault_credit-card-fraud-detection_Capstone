# Step 5. Model Selection & Training
# We will train multiple machine learning models.
# Compare performance using key metrics like accuracy, precision, recall, and F1-score.
# Select the best model based on overall performance.
# Why Model Selection is Important?
# No single model works best for all problems, so we will test multiple models.
# We will train four models and compare their performance:
# Logistic Regression - Simple and interpretable model.
# Decision Tree - Handles complex patterns but can overfit.
# Random Forest - An ensemble model that improves stability.
# XGBoost - A powerful gradient boosting algorithm.
# How Do We Compare Models?
# Accuracy - Measures overall correctness but is misleading for imbalanced data.
# Precision - Important when false positives (wrong fraud flags) are costly.
# Recall - Critical to catching all fraudulent transactions.
# F1-Score - Balances precision and recall, making it the best metric for fraud detection.
# Selecting the Best Model
# The best model will be the one that maximizes F1-score while maintaining high precision and recall.
# After training all models, we will compare them visually and numerically.
# The highest-performing model will be selected for further tuning and real-world deployment.
# 5.1 Training Logistic Regression
# Why Logistic Regression?
# Logistic Regression serves as a baseline model for comparison.
# It is simple, interpretable, and effective for binary classification.
# Helps us benchmark other models against a standard approach.
# Approach:
# Correct Train-Test Split: Ensured proper data separation before applying SMOTE.
# Class Weight Balancing: Used class_weight="balanced" to account for fraud cases.
# Performance Evaluation: Assessed using Precision, Recall, and F1-score.


from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Step 1: Correct Train-Test Split (Before Applying SMOTE)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 2: Apply SMOTE Only on Training Data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Confirm Shapes
print("Train set shape after SMOTE:", X_train_resampled.shape)
print("Test set shape:", X_test.shape)

# Step 3: Retrain Logistic Regression (Baseline Model)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

log_reg = LogisticRegression(class_weight="balanced", random_state=42)
log_reg.fit(X_train_resampled, y_train_resampled)

# Predicting on test data
y_pred_log = log_reg.predict(X_test)

# Performance Evaluation
print("Logistic Regression Performance:")
print(classification_report(y_test, y_pred_log))

# 5.2 Training Decision Tree Classifier
# Why Decision Trees?
# Decision Trees can model complex patterns and interactions between features.
# However, they may overfit without proper constraints.
# Applying depth control and regularization helps improve generalization.
# Approach:
# Max Depth: max_depth=10 (Limits excessive branching).
# Min Samples per Split: min_samples_split=5 (Prevents deep unnecessary splits).
# Min Samples per Leaf: min_samples_leaf=2 (Ensures a minimum number of samples per de F1-score.

# Import Decision Tree classifier
from sklearn.tree import DecisionTreeClassifier

# Training Decision Tree Model
dt_model = DecisionTreeClassifier(
    max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42
)
dt_model.fit(X_train_resampled, y_train_resampled)

# Predicting on test data
y_pred_dt = dt_model.predict(X_test)

# Performance Evaluation
print("Decision Tree Performance:")
print(classification_report(y_test, y_pred_dt))

# 5.3 Training Random Forest Classifier
# Why Random Forest?
# Random Forest is an ensemble learning method that improves Decision Trees by reducing overfitting.
# It aggregates multiple decision trees to enhance stability and generalization.
# Expected to perform better than a single Decision Tree by averaging multiple models.
# Approach:
# Number of Trees: n_estimators=100 (Builds 100 decision trees).
# Class Weighting: class_weight="balanced" (Handles imbalanced fraud cases).
# Max Depth: max_depth=10 (Prevents over-complexity).
# Min Samples per Split: min_samples_split=5 (Avoids excessive branching).
# Min Samples per Leaf: min_samples_leaf=2 (Ensures meaningfund F1-score.

# Import Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

# Training Random Forest Model
rf_model = RandomForestClassifier(
    n_estimators=100, class_weight="balanced", max_depth=10, 
    min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1
)
rf_model.fit(X_train_resampled, y_train_resampled)

# Predicting on test data
y_pred_rf = rf_model.predict(X_test)

# Performance Evaluation
print("Random Forest Performance:")
print(classification_report(y_test, y_pred_rf))

# 5.4 Training XGBoost Classifier
# Why XGBoost?
# XGBoost is a gradient boosting algorithm that improves weak models in sequential steps.
# It optimizes feature importance and handles imbalanced datasets well.
# Expected to perform better than standalone Decision Trees and Random Forest due to its boosting nature.
# Approach:
# Boosting Rounds: n_estimators=100 (Limits overfitting by capping iterations).
# Max Depth: max_depth=6 (Prevents excessive branching).
# Learning Rate: learning_rate=0.1 (Balances model adaptability).
# Objective Function: eval_metric="logloss" (Optimized for binary classifd F1-score.

# Import XGBoost
import xgboost as xgb

# Training XGBoost Model
xgb_model = xgb.XGBClassifier(
    use_label_encoder=False, eval_metric='logloss', max_depth=6, 
    learning_rate=0.1, n_estimators=100, random_state=42
)
xgb_model.fit(X_train_resampled, y_train_resampled)

# Predicting on test data
y_pred_xgb = xgb_model.predict(X_test)

# Performance Evaluation
print("XGBoost Performance:")
print(classification_report(y_test, y_pred_xgb))


# 5.5 Comparing Model Performance
# Why Compare Models?
# After training multiple models, we need to compare their effectiveness to select the best one.
# We will evaluate models using key metrics:
# Accuracy: Measures overall correctness.
# Precision: Indicates how many detected fraud cases were actually fraud.
# Recall: Measures how well fraud cases were identified.
# F1-Score: Balances precision and recall, making it crucial for fraud detection.
# Goal:
# Identify the best-performing model for fraud detection, considering both precision and recall.

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt

# Store model predictions
models = {
    "Logistic Regression": y_pred_log,
    "Decision Tree": y_pred_dt,
    "Random Forest": y_pred_rf,
    "XGBoost": y_pred_xgb
}

# Compare metrics for all models
comparison_df = pd.DataFrame(columns=["Accuracy", "Precision", "Recall", "F1-Score"])

for model_name, y_pred in models.items():
    comparison_df.loc[model_name] = [
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred)
    ]

# Display model comparison
print("\nModel Performance Comparison:\n")
print(comparison_df)

# Visualizing the comparison
plt.figure(figsize=(12, 6))
comparison_df.plot(kind='bar', figsize=(12, 6), colormap='RdYlBu')  # Using a color map for better visibility
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.legend(loc="lower right")

# Save the plot
plt.savefig(os.path.join(visuals_dir, "Model Performance Comparison.png"), bbox_inches='tight')

plt.show()


# 5.6 ROC Curve Visualization – Model Evaluation
# Why Use the ROC Curve?
# Helps assess the trade-off between True Positive Rate (TPR) and False Positive Rate (FPR).
# Visualizes model performance beyond accuracy and F1-score.
# Metric Used: AUC-ROC (Area Under the Curve)
# Higher AUC → Better ability to distinguish fraud from non-fraud.
# Approach:
# Compute predicted probabilities for each model.
# Calculate ROC Curve and AUC Score for all models.
# Compare Logistic Regression, Decision Tree, Random Forest, and XGBoost.
# Plot all ROC curves to visualize model effectiveness.

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Compute predicted probabilities for each model
y_pred_proba_log = log_reg.predict_proba(X_test)[:, 1]
y_pred_proba_dt = dt_model.predict_proba(X_test)[:, 1]
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

# Compute ROC curves
fpr_log, tpr_log, _ = roc_curve(y_test, y_pred_proba_log)
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_proba_dt)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_proba_xgb)

# Compute AUC scores
auc_log = auc(fpr_log, tpr_log)
auc_dt = auc(fpr_dt, tpr_dt)
auc_rf = auc(fpr_rf, tpr_rf)
auc_xgb = auc(fpr_xgb, tpr_xgb)

# Plot ROC Curves
plt.figure(figsize=(10, 6))
plt.plot(fpr_log, tpr_log, label=f"Logistic Regression (AUC = {auc_log:.4f})")
plt.plot(fpr_dt, tpr_dt, label=f"Decision Tree (AUC = {auc_dt:.4f})")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.4f})")
plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC = {auc_xgb:.4f})")

# Plot random chance line
plt.plot([0, 1], [0, 1], 'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()

# Save the plot
plt.savefig(os.path.join(visuals_dir, "ROC Curve Comparison.png"), bbox_inches='tight')

plt.show()

# 5.7 Probability Calibration for Best Model
# Why Calibrate Model Probabilities?
# Raw model probabilities may be miscalibrated, leading to inaccurate fraud probability scores.
# Calibrating probabilities ensures better decision-making when adjusting the classification threshold.
# Applies sigmoid calibration to the best-performing model (Random Forest).
# Approach:
# Use CalibratedClassifierCV to calibrate probabilities using sigmoid scaling.
# Apply calibration to Random Forest (best model based on F1-score).
# Visualize the distribution of calibrated fraud probabilities.

from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt

# Apply probability calibration to the selected best model (Random Forest)
calibrated_model = CalibratedClassifierCV(rf_model, method="sigmoid", cv="prefit")
calibrated_model.fit(X_train_resampled, y_train_resampled)

# Get calibrated probabilities for test data
y_probs_best = calibrated_model.predict_proba(X_test)[:, 1]

# Plot calibrated probability distribution
plt.figure(figsize=(8, 5))
plt.hist(y_probs_best, bins=20, edgecolor="black", alpha=0.7)
plt.xlabel("Calibrated Fraud Probability")
plt.ylabel("Frequency")
plt.title("Calibrated Distribution of Fraud Probabilities")
plt.grid()

# Save the plot
plt.savefig(os.path.join(visuals_dir, "Calibrated Distribution of Fraud Probabilities.png"), bbox_inches='tight')

plt.show()

