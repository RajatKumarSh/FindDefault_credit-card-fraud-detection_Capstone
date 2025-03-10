# Step 3. Dealing with Imbalanced Data
# 3.1 Handling Class Imbalance (SMOTE)
# Problem: The dataset is highly imbalanced (fraud cases are only 0.172% of total transactions).
# Why Important?

# An imbalanced dataset can cause the model to favor the majority class (non-fraudulent transactions), leading to poor fraud detection.
# Fraud cases are underrepresented, which affects model learning.
# Solution:
# We will use two common techniques to handle imbalance:
# Oversampling (SMOTE) – Increase fraud cases by creating synthetic samples.
# Undersampling – Reduce the number of non-fraud cases.

# Approach:

# First, check the class distribution before and after applying SMOTE.
# Use SMOTE to generate synthetic fraud transactions and balance the dataset.

from imblearn.over_sampling import SMOTE

# Splitting features (X) and target variable (y)
X = df.drop(columns=['Class'])
y = df['Class']

# Applying SMOTE for oversampling the minority class
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Display the new class distribution
print("Class distribution after SMOTE:\n", y_resampled.value_counts())

# 3.2 Saving the Processed Dataset
# Once data preprocessing is completed, it is essential to save the processed dataset for future use.
# This allows us to reuse the cleaned and transformed data without re-running the entire preprocessing step.

# Why Save the Processed Dataset?
# Prevents redundant execution of preprocessing steps.
# Ensures reproducibility for further model training and analysis.
# Makes it easier to debug and track changes in the dataset.
# Saved Dataset Details
# Filename: processed_data.csv
# Location: Stored in the data/ folder.
# Contents: The dataset after handling missing values, scaling, and balancing.
# The saved dataset will be used in subsequent steps, including feature engineering and model training.

# Define the correct path for the processed dataset
processed_data_path = os.path.join(r"C:\Users\iamra\Desktop\CAPSTONE Project\credit-card-fraud-detection\data", "processed_data.csv")

# Save the processed dataset after preprocessing
df.to_csv(processed_data_path, index=False)

print(f"Processed dataset saved successfully at: {processed_data_path}")