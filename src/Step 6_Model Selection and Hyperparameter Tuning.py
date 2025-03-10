# Step 6. Model Selection & Hyperparameter Tuning
# 6.1 Selecting the Best Model for Tuning
# Why Tune Hyperparameters?
# Fine-tuning helps optimize precision-recall tradeoffs, reducing false positives while maintaining high fraud detection rates.
# The best model from previous evaluations is Random Forest, based on its:
# Best F1-score (58.6%)
# Highest precision (44.3%)
# Highest AUC (0.9813)
# Goal:
# Optimize Random Forest to further improve F1-score without sacrificing recall.
# Tune key hyperparameters to balance precision, recall, and overall stability.
# Approach:
# We will perform RandomizedSearchCV to find optimal values for:

# n_estimators: Number of trees in the forest.
# max_depth: Controls tree depth to prevent overfitting.
# min_samples_split: Minimum samples required to split a node.
# min_samples_leaf: Minimum samples required per leaf.
# max_features: Number of features to consider for each split.


from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import classification_report

# Optimized Hyperparameter Grid (Smaller Search Space)
param_grid = {
    "n_estimators": [50, 100, 150],  # Reduced range
    "max_depth": [5, 10, 15],  # Removed None to prevent deep trees
    "min_samples_split": [2, 5],  # Reduced options
    "min_samples_leaf": [1, 2],  # Reduced options
    "max_features": ["sqrt"]  # Using only sqrt (faster & optimal)
}

# Initialize Random Forest model with multi-core processing
rf_model_tune = RandomForestClassifier(
    class_weight="balanced",
    random_state=42,
    n_jobs=-1  # Uses all CPU cores for speed
)

# Perform RandomizedSearchCV with optimized parameters
random_search = RandomizedSearchCV(
    estimator=rf_model_tune,
    param_distributions=param_grid,
    n_iter=10,  # Reduced number of trials
    cv=3,  # 3-fold cross-validation
    verbose=2,
    random_state=42,
    n_jobs=-1  # Parallel execution
)

# Fit on training data
random_search.fit(X_train_resampled, y_train_resampled)

# Print best parameters
print("Best Hyperparameters for Random Forest:", random_search.best_params_)

# Train the best model using optimal parameters
best_rf_model = random_search.best_estimator_

# Predicting on test data
y_pred_best_rf = best_rf_model.predict(X_test)

# Performance Evaluation
print("Optimized Random Forest Performance:")
print(classification_report(y_test, y_pred_best_rf))


# 6.2 Threshold Tuning for Fraud Detection
# Why Adjust the Classification Threshold?
# By default, models classify fraud if probability > 0.5 (standard threshold).
# This may not be optimal for fraud detection, where higher precision is often needed.
# We need to experiment with different thresholds to balance false positives and false negatives.
# Approach:
# Get probability predictions from the optimized Random Forest model.
# Test different thresholds (0.3, 0.5, 0.7, etc.).
# Measure Precision, Recall, and F1-score at each threshold.
# Select the best threshold based on business needs (maximize fraud detection while minimizing false alarms).

import numpy as np
from sklearn.metrics import precision_recall_curve

# Get fraud probabilities from the optimized Random Forest model
y_probs_rf = best_rf_model.predict_proba(X_test)[:, 1]

# Define threshold values to test
thresholds = np.arange(0.1, 0.9, 0.1)

# Store precision, recall, and F1-score for different thresholds
precision_scores = []
recall_scores = []
f1_scores = []

for threshold in thresholds:
    # Convert probabilities to binary predictions based on threshold
    y_pred_thresh = (y_probs_rf >= threshold).astype(int)
    
    # Calculate precision, recall, and F1-score
    precision, recall, _ = precision_recall_curve(y_test, y_pred_thresh)
    f1_score_value = (2 * precision * recall) / (precision + recall)
    
    precision_scores.append(precision[1])
    recall_scores.append(recall[1])
    f1_scores.append(f1_score_value[1])

# Convert results into a DataFrame for easy visualization
import pandas as pd
import matplotlib.pyplot as plt

threshold_results = pd.DataFrame({
    "Threshold": thresholds,
    "Precision": precision_scores,
    "Recall": recall_scores,
    "F1-Score": f1_scores
})

# Display results
print(threshold_results)

# Plot Precision-Recall vs. Threshold
plt.figure(figsize=(8, 5))
plt.plot(thresholds, precision_scores, label="Precision", marker="o")
plt.plot(thresholds, recall_scores, label="Recall", marker="s")
plt.plot(thresholds, f1_scores, label="F1-Score", marker="d")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision-Recall Tradeoff with Different Thresholds")
plt.legend()
plt.grid()

# Save the plot
plt.savefig(os.path.join(visuals_dir, "Precision-Recall Tradeoff with Different Thresholds.png"), bbox_inches='tight')

plt.show()


# Apply updated threshold to make fraud predictions
final_threshold = 0.7
y_pred_final = (y_probs_best >= final_threshold).astype(int)

# Evaluate final model performance
from sklearn.metrics import classification_report

print("Final Model Performance with Threshold =", final_threshold)
print(classification_report(y_test, y_pred_final))


# 6.3 Selecting the Best Model
# Why Select the Best Model?
# We will dynamically select the one with the highest F1-score.
# This ensures that the best-performing model is used for fraud detection.
# The final model is evaluated using the optimized Threshold = 0.7 from Step 6.2.
# Approach:
# Compute F1-score for all models and auto-select the best-performing model.
# Retrieve best hyperparameters dynamically.
# Apply final fraud detection threshold (0.7) for optimized predictions.

from sklearn.metrics import f1_score

# Store all models and their predictions
model_predictions = {
    "Logistic Regression": y_pred_log,
    "Decision Tree": y_pred_dt,
    "Random Forest": y_pred_rf,
    "XGBoost": y_pred_xgb
}

# Compute F1-scores for each model
f1_scores = {model: f1_score(y_test, y_pred) for model, y_pred in model_predictions.items()}

# Select the model with the highest F1-score
best_model_name = max(f1_scores, key=f1_scores.get)
best_model = {
    "Logistic Regression": log_reg,
    "Decision Tree": dt_model,
    "Random Forest": best_rf_model,  # Using the tuned Random Forest model
    "XGBoost": xgb_model
}[best_model_name]

# Print the best model details
print(f"Best Model Selected: {best_model_name}\n")
print("Best Hyperparameters:")
print(best_model.get_params())

# Final Predictions using the best model
y_probs_best = best_model.predict_proba(X_test)[:, 1]

# Apply final threshold (selected from tuning step)
final_threshold = 0.7
y_pred_final = (y_probs_best >= final_threshold).astype(int)

# Print final model performance
print("\nFinal Decision Threshold:", final_threshold)
print("\nFinal Model Performance:")
print(classification_report(y_test, y_pred_final))


# ## **Step 6 Summary: Best Model Selection & Optimization**  

# ### **Best Model: Random Forest (Auto-Selected)**  
# **Automatically chosen based on the highest F1-score.**  
# **Optimized using Hyperparameter Tuning & Threshold Tuning (`0.7`).**  
# **Achieved the best balance between Precision & Recall.**  

# ---

# ### **Final Hyperparameters for Random Forest (Tuned)**
# | **Hyperparameter**       | **Final Value** |
# |-------------------------|----------------|
# | `n_estimators`         | 150            |
# | `max_depth`           | 15             |
# | `min_samples_split`   | 2              |
# | `min_samples_leaf`    | 1              |
# | `max_features`        | "sqrt"         |
# | `class_weight`        | "balanced"     |

# ---

# ### **Final Model Performance (Threshold = `0.7`)**
# | **Metric**      | **Final Model (`Threshold = 0.7`)** |
# |----------------|--------------------------------|
# | **Precision (Fraud - 1)** | `76%` |
# | **Recall (Fraud - 1)** | `80%` |
# | **F1-Score** | `78%` |
# | **Accuracy** | `100%` |

# ---

# ### **Model Performance Before & After Optimization**
# | **Metric**      | **Before Optimization** | **After Optimization (Final Model)** |
# |----------------|----------------------|----------------------|
# | **Precision (Fraud)** | `44%` | `76%` |
# | **Recall (Fraud)** | `87%` | `80%` |
# | **F1-Score** | `58.6%` | `78%` |
# | **Threshold** | `0.5` | `0.7` |

# ---

# ### **Key Insights**  
# **Precision improved from `44%` to `76%`** → **Fewer false positives, better fraud flagging.**  
# **Balanced Recall (`80%` vs. `87%`)** → **Still catching most fraud cases.**  
# **F1-score (`78%`) shows an optimized trade-off** between detecting fraud & reducing false alarms.  
# **Model is now fine-tuned and optimized for deployment.**  


# Ensure best_rf_model is correctly stored
print("Best Model Type:", type(best_rf_model))
print("Best Model Parameters:", best_rf_model.get_params())