# Step 8: Final Model Evaluation & Deployment Considerations
# 8.1 Precision-Recall & ROC-AUC Analysis
# Why This Matters?
# Fraud detection models must balance precision (avoiding false fraud alerts) and recall (catching as many fraud cases as possible).
# Precision-Recall Curve & ROC Curve help us evaluate trade-offs and assess model effectiveness.
# Precision-Recall Curve Analysis
# Precision-Recall is more informative than ROC-AUC in imbalanced datasets.
# Higher AUC-PR = Better fraud detection performance.
# X-axis: Recall (How many actual fraud cases we catch).
# Y-axis: Precision (How many predicted fraud cases are actually fraud).
# ROC Curve Analysis
# ROC Curve helps measure how well the model separates fraud vs. non-fraud transactions.
# X-axis: False Positive Rate (% of normal transactions incorrectly flagged as fraud).
# Y-axis: True Positive Rate (% of fraud transactions correctly detected).
# Goal: A strong fraud detection model has high recall with low FPR (steep curve towards top-left).

# Precision-Recall Curve Analysis
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

# Compute Precision-Recall Curve
y_probs_best = best_rf_model.predict_proba(X_test)[:, 1]  # Get fraud probabilities
precision, recall, _ = precision_recall_curve(y_test, y_probs_best)
pr_auc = auc(recall, precision)
# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', label=f"Random Forest (AUC = {pr_auc:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for Final Model")
plt.legend(loc="lower left")
plt.grid()

# Save the plot
plt.savefig(os.path.join(visuals_dir, "Precision-Recall Curve for Final Model.png"), bbox_inches='tight')

plt.show()


# ROC Curve Analysis
from sklearn.metrics import roc_curve

# Compute ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_probs_best)
roc_auc = auc(fpr, tpr)

# Plot Full ROC Curve
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='blue', label=f"Random Forest (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color="grey", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.legend(loc="lower right")

# Save the plot
plt.savefig(os.path.join(visuals_dir, "ROC Curve.png"), bbox_inches='tight')

plt.grid()

# Zoomed-in ROC Curve (Low False Positive Region)
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, color='blue', label="Zoomed-in View")
plt.plot([0, 1], [0, 1], color="grey", linestyle="--")
plt.xlim([0, 0.1])  # Focus on low FPR
plt.ylim([0.9, 1])  # Focus on high recall
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (Recall)")
plt.title("Zoomed-in ROC Curve (Low FPR Region)")
plt.legend(loc="lower right")
plt.grid()

plt.tight_layout()

# Save the plot
plt.savefig(os.path.join(visuals_dir, "Zoomed-in ROC Curve (Low FPR Region).png"), bbox_inches='tight')

plt.show()

# 8.2 Optimized Threshold Selection
# Why Optimize the Decision Threshold?

# Default classification uses 0.5 as the threshold.
# Lower threshold = More fraud cases caught (but more false positives).
# Higher threshold = Fewer false alerts (but some fraud may be missed).
# Goal: Find the best balance using Precision, Recall, and F1-score.

from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# Define threshold values from 0 to 1
thresholds = np.linspace(0, 1, 100)
precision_scores, recall_scores, f1_scores = [], [], []

# Evaluate Precision, Recall, and F1-Score at different thresholds
for thresh in thresholds:
    y_pred_thresh = (y_probs_best >= thresh).astype(int)
    precision_scores.append(precision_score(y_test, y_pred_thresh))
    recall_scores.append(recall_score(y_test, y_pred_thresh))
    f1_scores.append(f1_score(y_test, y_pred_thresh))

# Plot Threshold vs. Metrics
plt.figure(figsize=(8, 6))
plt.plot(thresholds, precision_scores, label="Precision", linestyle="--", marker="o")
plt.plot(thresholds, recall_scores, label="Recall", linestyle="-.", marker="s")
plt.plot(thresholds, f1_scores, label="F1-Score", linestyle="-", marker="d")
plt.xlabel("Decision Threshold")
plt.ylabel("Score")
plt.title("Threshold Tuning: Precision-Recall Trade-off")
plt.legend()
plt.grid()

# Save the plot
plt.savefig(os.path.join(visuals_dir, "Threshold Tuning: Precision-Recall Trade-off.png"), bbox_inches='tight')

plt.show()

# Selecting the Optimal Threshold
from sklearn.metrics import roc_curve

# Compute ROC Curve to find the best threshold
fpr, tpr, thresholds = roc_curve(y_test, y_probs_best)

# Select the threshold where TPR - FPR is maximized
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

# Print the selected optimal threshold
print(f"Optimal Threshold Selected: {optimal_threshold:.2f}")

# Apply the optimal threshold to get final fraud predictions
y_pred_optimal = (y_probs_best >= optimal_threshold).astype(int)

# Print classification report
from sklearn.metrics import classification_report
print("\nFinal Model Performance at Optimal Threshold:")
print(classification_report(y_test, y_pred_optimal))

# Step 8.3: Applying the Optimal Threshold & Making Final Predictions
# Why Apply an Optimal Threshold?
# By default, machine learning models classify transactions as fraud or non-fraud using a 0.5 threshold.
# However, in fraud detection, we often need a lower threshold to maximize recall (catching fraud cases).
# Based on Step 8.2, we selected an optimal threshold of 0.19, which balances recall and precision.
# What Happens in This Step?
# Apply the selected threshold (0.19) to classify transactions.
# Print a few sample predictions to verify fraud detection.
# Compare default (0.5) vs. optimized (0.19) threshold to see improvement.

# Applying the Optimal Threshold

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

# Compute fraud probabilities for test transactions
y_probs_final = best_rf_model.predict_proba(X_test)[:, 1]  # Select probability of fraud class

# Apply Default (0.5) vs. Optimized (0.19) Thresholds
y_pred_default = (y_probs_final >= 0.5).astype(int)  # Default threshold
y_pred_optimized = (y_probs_final >= optimal_threshold).astype(int)  # Optimized threshold (0.19)
# Display Classification Report for Optimized Threshold
print("\n**Final Model Performance at Optimized Threshold (0.19):**\n")
print(classification_report(y_test, y_pred_optimized))

# Compare Default (0.5) vs. Optimized (0.19)
df_comparison = pd.DataFrame({
    "Actual Label": y_test,
    "Fraud Probability": y_probs_final,
    "Prediction (Default 0.5)": y_pred_default,
    "Prediction (Optimized 0.19)": y_pred_optimized
})

# Display a few sample predictions to verify
print("\n**Sample Fraud Predictions (Comparison of Default vs. Optimized Threshold):**")
print(df_comparison.sample(10, random_state=42))  # Display 10 random transactions for review

# Step 8.4: Final Deployment Considerations
# Why Deployment Considerations Matter?
# A well-trained fraud detection model must be integrated into real-world systems for real-time fraud prevention.
# The model should be able to process new transactions efficiently and flag potential fraud instantly.
# Key deployment considerations include:
# Model Persistence (Saving & Loading for Future Use).
# Seamless Integration into Financial Systems.
# Real-Time Fraud Prediction & Decision Making.
# Ensuring Explainability & Compliance.
# Step 8.4.1 - Saving & Loading the Final Model
# Why Save the Model?

# Enables real-time fraud detection without retraining.
# Can be deployed as an API or integrated into banking/payment systems.
# Format:

# We use joblib to store the trained model efficiently.
# The model is saved as a .pkl file for later use.

import joblib

# Define the full path for saving the model
model_filename = r"C:\Users\iamra\Desktop\CAPSTONE Project\credit-card-fraud-detection\models\final_fraud_detection_model.pkl"

# Save the model directly to the correct location
joblib.dump(best_rf_model, model_filename)

print(f" Our Model Saved Successfully at: {model_filename}")

# Step 8.4.2 - Loading the Saved Model for Fraud Prediction
# Why Load the Model?

# In production, we load the saved model instead of retraining it.
# Ensures fraud detection can run on new transactions without reprocessing past data.

# Define the full path for loading the model
model_load_path = r"C:\Users\iamra\Desktop\CAPSTONE Project\credit-card-fraud-detection\models\final_fraud_detection_model.pkl"

# Load the saved model
loaded_model = joblib.load(model_load_path)

print("Model Loaded Successfully for Real-Time Prediction!")

# Step 8.5 Real-Time & Batch Fraud Prediction
# Why Real-Time Fraud Prediction?
# Fraud detection models must be capable of identifying fraudulent transactions instantly in real-world banking systems.
# This step focuses on predicting fraud probability for:
# A single new transaction (real-time prediction).
# A batch of transactions (bulk processing).
# How It Works:
# A new transaction arrives.
# The model extracts key transaction features.
# The trained model predicts the fraud probability (0 to 1).
# The decision threshold (from Step 8.3) is applied to classify:

# Fraud Alert! if probability ≥ 0.19
# Safe Transaction if probability < 0.19
# Flag suspicious transactions for further review.

# Expected Output:
# Fraud probability for a sample transaction
# Fraud classification (Fraud / Not Fraud)
# Comparison of predictions using default (0.5) vs. optimized threshold (0.19)
# 8.5.1 Real-Time Fraud Prediction - Run Model on New Transaction

import joblib
import numpy as np

# Define the full path to the saved model
model_load_path = r"C:\Users\iamra\Desktop\CAPSTONE Project\credit-card-fraud-detection\models\final_fraud_detection_model.pkl"

# Load the model from the correct directory
loaded_model = joblib.load(model_load_path)
print("\nModel Loaded Successfully for Real-Time Prediction!")

# Selecting a random transaction from the test dataset
new_transaction = X_test.sample(n=1, random_state=42)

# Predicting fraud probability
fraud_probability = loaded_model.predict_proba(new_transaction)[:, 1][0]

# Print the fraud probability
print(f"\nFraud Probability for New Transaction: {fraud_probability:.4f}")

# Apply the optimal threshold (previously selected as 0.19)
optimal_threshold = 0.19
fraud_label = "Fraud Alert!" if fraud_probability >= optimal_threshold else "Safe Transaction"

# Print classification result
print(f"Classification: {fraud_label}")

# 8.5.2 Batch Fraud Prediction - Running Model on Multiple Transactions

import joblib
import pandas as pd
import numpy as np

# Define the full path to the saved model
model_load_path = r"C:\Users\iamra\Desktop\CAPSTONE Project\credit-card-fraud-detection\models\final_fraud_detection_model.pkl"

# Load the model from the correct directory
loaded_model = joblib.load(model_load_path)
print("\nModel Loaded Successfully for Batch Fraud Prediction!")

# Select a batch of 10 random transactions from the test dataset
batch_transactions = X_test.sample(n=10, random_state=42)

# Predict fraud probabilities for all selected transactions
fraud_probabilities = loaded_model.predict_proba(batch_transactions)[:, 1]

# Define the optimal threshold (previously selected as 0.19)
optimal_threshold = 0.19

# Classify transactions based on the optimized threshold
batch_predictions = (fraud_probabilities >= optimal_threshold).astype(int)

# Create a DataFrame to display results
batch_results = batch_transactions.copy()
batch_results["Fraud Probability"] = fraud_probabilities
batch_results["Prediction"] = batch_predictions
batch_results["Prediction Label"] = batch_results["Prediction"].apply(lambda x: "Fraud Alert!" if x == 1 else "Safe Transaction")

# Display the batch results
print("\n---------Batch Fraud Prediction Results:---------")
print(batch_results[["Fraud Probability", "Prediction Label"]])

# Print summary statistics
print("\n---------Batch Fraud Prediction Summary:---------")
print(f"Total Transactions Evaluated: {len(batch_results)}")
print(f"Fraud Alerts: {batch_results['Prediction'].sum()}")
print(f"Safe Transactions: {len(batch_results) - batch_results['Prediction'].sum()}")


# Step 8 Key Insights from Final Model Evaluation
# Precision-Recall & ROC Analysis:
# The Precision-Recall Curve showed a strong ability to detect fraudulent transactions.
# The optimal threshold (0.19) increased recall to 91%, capturing more fraud cases while controlling false positives.
# The ROC Curve confirmed a high AUC score, demonstrating excellent fraud detection capability.
# Impact of Threshold Optimization:
# Default 0.5 threshold → Missed many fraud cases (low recall).
# Optimized 0.19 threshold → 91% recall, meaning almost all fraud cases were detected.
# Trade-off: Slightly lower precision, but higher recall is preferred in fraud detection.
# Real-Time Fraud Detection Performance:
# The model accurately classified transactions using the optimized threshold.
# Single Transaction Example: Safe Transaction detected with low fraud probability (0.0011).
# Batch Transaction Example: 10 transactions analyzed, all classified as Safe Transactions.
# Business Takeaways:
# Fraud Risk Minimization:

# A lower threshold helps detect more fraud cases, reducing financial losses.
# High recall ensures few fraudulent transactions go undetected.
# Operational Efficiency:

# The batch fraud detection process allows for scalable, real-time fraud prevention.
# Can be integrated into banking APIs or payment gateways for real-time monitoring.
# Next Steps for Deployment:

# Deploy as an API for live fraud detection in banking systems.
# Monitor performance metrics to ensure long-term effectiveness.