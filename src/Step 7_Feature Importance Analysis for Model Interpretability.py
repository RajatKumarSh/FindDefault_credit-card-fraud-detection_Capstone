# Step 7: Feature Importance Analysis for Model Interpretability
# Why Feature Importance Matters?
# In fraud detection, understanding which features impact predictions helps improve model reliability.
# Instead of SHAP, we use Random Forest Feature Importance, which measures how much each feature contributes to decision-making.
# This helps identify key patterns in fraud transactions and possible areas for further analysis. #
# How is Feature Importance Calculated?
# The model assigns an importance score to each feature based on how much it reduces impurity in decision trees.
# Higher values mean the feature plays a significant role in distinguishing fraud from non-fraud transations.
# 7.1 Visualizing Feature Importance

## **Computing Feature Importance for Random Forest Model**

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Get feature importances from the trained Random Forest model
feature_importances = best_rf_model.feature_importances_

# Store as a Pandas Series for easy sorting
feature_importance_series = pd.Series(feature_importances, index=X_test.columns).sort_values(ascending=False)

# **Re-plot Feature Importance Bar Graph**
plt.figure(figsize=(12, 5))
feature_importance_series.plot(kind="bar", color="teal")
plt.xlabel("Features")
plt.ylabel("Feature Importance Score")
plt.title("Random Forest Feature Importance")
plt.xticks(rotation=90)

# Save the plot
plt.savefig(os.path.join(visuals_dir, "Random Forest Feature Importance.png"), bbox_inches='tight')

plt.show()

# Step 7.2: Additional Visualizations & Insights
# 7.2.1 Correlation Heatmap - Top Features
# Why?

# Confirms important relationships between top fraud-related features.
# Helps validate which features impact fraudulent transactions the most.
# What Will We Do?

# Select Top 10 most important features from our Random Forest Feature Importance plot.
# Generate a correlation heatmap to analyze their interdependencies.

# Selecting top 10 features based on importance
top_features = feature_importance_series.head(10).index
corr_matrix = X_test[top_features].corr()

# Plotting heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap - Top 10 Important Features")


# Save the plot
plt.savefig(os.path.join(visuals_dir, "Correlation Heatmap - Top 10 Important Features.png"), bbox_inches='tight')

plt.show()

# 7.2.2 Scatter Plots for Top Feature Interactions
# Why?

# Fraud patterns are often non-linear, so a scatter plot can reveal useful relationships.
# Helps us see if certain patterns consistently indicate fraudulent transactions.
# What Will We Do?

# Select the Top 2 Most Important Features
# Plot their relationship, differentiating fraud and non-fraud cases.

import numpy as np

# Selecting Top 2 features for scatter plot
feature_x = "V14"  # Most important feature
feature_y = "V12"  # Second most important feature

# Create scatter plot
plt.figure(figsize=(8, 5))
plt.scatter(X_test[feature_x], X_test[feature_y], c=y_test, cmap="coolwarm", alpha=0.5)
plt.xlabel(feature_x)
plt.ylabel(feature_y)
plt.title(f"Scatter Plot: {feature_x} vs. {feature_y} (Fraud vs. Non-Fraud)")
plt.colorbar(label="Fraud (1) vs Non-Fraud (0)")

# Save the plot
plt.savefig(os.path.join(visuals_dir, "Scatter Plot.png"), bbox_inches='tight')

plt.show()

# Step 7.3: Feature Distribution - Fraud vs. Non-Fraud

import seaborn as sns
import matplotlib.pyplot as plt

# Select key important features (Top 5 from Feature Importance)
top_features = ["V14", "V10", "V12", "V4", "V17"]

# Set up the figure
plt.figure(figsize=(12, 8))

# Loop through top features and plot their distribution
for i, feature in enumerate(top_features, 1):
    plt.subplot(3, 2, i)
    sns.histplot(data=X_test, x=feature, hue=y_test, bins=50, kde=True, palette=["blue", "red"], alpha=0.6)
    plt.title(f"Feature Distribution: {feature} (Fraud vs. Non-Fraud)")
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.legend(["Non-Fraud (0)", "Fraud (1)"], loc="upper right")

plt.tight_layout()


# Save the plot
plt.savefig(os.path.join(visuals_dir, "Feature Distribution.png"), bbox_inches='tight')

plt.show()


# Key Insights - Feature Importance & Distributions
# Key Takeaways from Feature Importance & Distribution Analysis

# Top Features Impacting Fraud Prediction:

# V14, V10, and V12 have the highest influence on detecting fraudulent transactions.
# V4, V17, and V3 also play a significant role in distinguishing fraud cases.
# Feature Distributions - Fraud vs. Non-Fraud:

# Fraudulent transactions show distinct distribution patterns in key features.
# V14, V10, and V12 tend to have more negative values for fraud cases, indicating potential risk zones.
# V4 and V17 have wider spreads, showing more variability in fraudulent transactions.
# Correlation & Interactions:

# Top features show little correlation with each other, meaning each feature contributes uniquely to fraud detection.
# Scatter plot insights show distinct fraud patterns—fraud cases tend to cluster in certain areas of feature space.
# Business Impact:

# Understanding which features influence fraud detection can help financial institutions improve fraud alert systems.
# High-risk patterns identified in top features can be integrated into rule-based detection systems for early fraud intervention.
# Final Decision: Feature importance analysis confirms that our selected model effectively captures fraud patterns.