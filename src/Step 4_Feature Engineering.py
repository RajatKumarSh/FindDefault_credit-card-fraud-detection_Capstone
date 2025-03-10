# Step 4: Feature Engineering
# 4.1 Splitting Data into Train-Test Sets
# Why? Before training models, we need to divide the dataset into training (80%) and testing (20%) sets.
# Method:

# Use train_test_split() from sklearn.model_selection.
# Stratify by y_resampled to maintain class balance.
# Next Step: Proceed with Skewness Analysis to understand feature distributions.

# Splitting the data into train and test sets (80% train, 20% test)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Print shape to confirm
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# 4.2 Skewness Analysis of Features
# Why? Many machine learning models assume normally distributed features. Checking skewness helps identify highly skewed variables.
# Method:

# Plot KDE (Kernel Density Estimation) graphs for each feature.
# Display skewness values for each feature.
# Decision:

# We will not apply Power Transformations since PCA has already handled most transformations.

import matplotlib.pyplot as plt
import seaborn as sns

# Get feature columns (excluding 'Class' column)
feature_cols = X_train.columns
print("Feature Columns:\n", feature_cols)

# Plotting the distribution of the variables (skewness)
plt.figure(figsize=(17, 28))

for i, col in enumerate(feature_cols, 1):
    plt.subplot(6, 5, i)
    sns.kdeplot(X_train[col], shade=True, color="blue")
    plt.title(f"{col} (Skew: {X_train[col].skew():.2f})")

plt.tight_layout()

# Save the plot
plt.savefig(os.path.join(visuals_dir, "Skeweness Analysis Of Features.png"), bbox_inches='tight')
plt.show()


# 4.3 Feature Selection
# Why? Not all features contribute equally to model performance. Some may be redundant or irrelevant.
# Goal: Select the most important features while removing noisy or highly correlated ones.
# Approach:
# Correlation Analysis – Identify features that are highly correlated (>0.9).
# Feature Importance – Use a machine learning model to rank features.
# Low-Variance Filtering – Remove features that contribute minimal variance.

# Checking highly correlated features
correlation_matrix = X_resampled.corr()

# Finding features with correlation above 0.9 (high correlation threshold)
high_corr_features = [column for column in correlation_matrix.columns if any(correlation_matrix[column] > 0.9)]
print("Highly correlated features (Threshold > 0.9):\n", high_corr_features)

# 4.4 Removing Highly Correlated Features (Updated Approach)
# Why? Removing highly correlated features helps reduce redundancy and prevents overfitting.
# Threshold: We will remove only one feature from each pair with a correlation above 0.9, instead of dropping all highly correlated features.
# Approach:
# Identify feature pairs with correlation > 0.9.
# Keep only one feature from each correlated pair.
# Drop the selected redundant features.
# Reapply SMOTE to rebalance the dataset.

# Reload X_resampled from the original dataset (excluding 'Class')
X_resampled = pd.DataFrame(X, columns=X.columns)

# Compute correlation matrix again to ensure correctness
correlation_matrix = X_resampled.corr()

# Identify pairs of highly correlated features
high_corr_pairs = set()
for col in correlation_matrix.columns:
    for row in correlation_matrix.index:
        if col != row and abs(correlation_matrix.loc[row, col]) > 0.9:
            high_corr_pairs.add((col, row))

# Convert set to a list of features to remove (keeping only one per pair)
features_to_remove = list(set([pair[1] for pair in high_corr_pairs]))

# Drop selected features
X_resampled = X_resampled.drop(columns=features_to_remove, errors='ignore')

print(f"Removed {len(features_to_remove)} highly correlated features. New shape: {X_resampled.shape}")

# Reapply SMOTE on the updated feature set
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_resampled, y)

# Check new shape
print("After reapplying SMOTE:")
print("X_resampled shape:", X_resampled.shape)
print("y_resampled shape:", y_resampled.shape)


# 4.5 Feature Importance Analysis (Optimized for Faster Execution)
# Why? Not all features contribute equally to fraud detection. Removing low-importance features improves efficiency.
# Optimization:

# Reduced n_estimators=50 (fewer trees, faster execution).
# Set max_depth=10 (limits tree depth, prevents unnecessary splits).
# Used n_jobs=-1 (parallel processing for speed).
# Impact: These optimizations maintain feature importance rankings while improving execution time.


from sklearn.ensemble import RandomForestClassifier

# Train a Random Forest with optimizations
rf_model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_resampled, y_resampled)

# Get feature importance scores
feature_importances = pd.Series(rf_model.feature_importances_, index=X_resampled.columns)
feature_importances = feature_importances.sort_values(ascending=False)

# Plot the top 10 most important features
plt.figure(figsize=(10, 5))
sns.barplot(x=feature_importances[:10], y=feature_importances[:10].index, palette="viridis")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Top 10 Most Important Features")

# Save the plot
plt.savefig(os.path.join(visuals_dir, "Top 10 Most Important Features.png"), bbox_inches='tight')

plt.show()


# 4.6 Removing Low-Importance Features
# Why? Features with very low importance contribute little to fraud detection.
# Goal: Remove features with near-zero importance to simplify the model.
# Method:
# Set a threshold (e.g., importance < 0.01).
# Drop those features to improve model efficiency.

# Set a threshold for feature importance (e.g., 0.01)
threshold = 0.01

# Identify low-importance features
low_importance_features = feature_importances[feature_importances < threshold].index.tolist()

# Drop these features from X_resampled
X_resampled = X_resampled.drop(columns=low_importance_features, errors='ignore')

print(f"Removed {len(low_importance_features)} low-importance features. New shape: {X_resampled.shape}")


# Feature Engineering Summary
# After applying feature selection and importance analysis, we finalized the key features for fraud detection.

# Steps Completed:
# Feature Selection: Checked for highly correlated features (None were removed due to PCA).
# Feature Importance Analysis: Identified top features using a Random Forest model.
# Low-Importance Feature Removal: Dropped 17 features with near-zero contribution.
# Final Dataset After Feature Engineering:
# Total Features Before Selection: 30
# Features Removed: 17
# Final Features Retained for Model Training: 13
# The dataset is now optimized for training machine learning models with reduced dimensionality and improved efficiency.