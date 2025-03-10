# Importing Necessary Libraries & Loading the Dataset

# Importing necessary libraries
#### Importing Necessary Libraries

# Data Handling
import numpy as np
import pandas as pd

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Preprocessing & Model Selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# OS Interaction
import os

# Warnings (to suppress unnecessary warnings)
import warnings
warnings.filterwarnings("ignore")

# Ensuring XGBoost is installed (required for model training)
!pip install xgboost


Step 1. Exploratory data analysis
1.1 Loading the dataset
file_path = r"C:\Users\iamra\Desktop\CAPSTONE Project\credit-card-fraud-detection\data\creditcard.csv"
df = pd.read_csv(file_path)

# Displaying the first 5 rows
df.head()

# 1.2 Data Exploration & Analysis
# To understand the dataset better, we will:

# Check for missing values and data types.
# Analyze the distribution of fraudulent vs. non-fraudulent transactions.
# Exploring basic statistics and visualize feature distributions.
# Identify potential correlations between features.
# Displaying dataset information (columns, data types, missing values)
df.info()

# Checking for missing values
missing_values = df.isnull().sum()
missing_values[missing_values > 0]  # Display only columns with missing values (if any)

1.3 Fraud vs. Non-Fraud Distribution

# # Counting the number of fraudulent (1) and non-fraudulent (0) transactions

# Defining the absolute path for the visuals directory
visuals_dir = r"C:\Users\iamra\Desktop\CAPSTONE Project\credit-card-fraud-detection\visuals"

# Ensuring the visuals directory exists
os.makedirs(visuals_dir, exist_ok=True)

# Counting the number of fraudulent (1) and non-fraudulent (0) transactions
fraud_counts = df['Class'].value_counts()
fraud_percent = (fraud_counts / df.shape[0]) * 100  # Convert to percentage

# Display class distribution with percentages
print("Class Distribution:\n", fraud_counts)
print(f"\nFraudulent Transactions: {fraud_counts[1]} ({fraud_percent[1]:.5f}%)")
print(f"Non-Fraudulent Transactions: {fraud_counts[0]} ({fraud_percent[0]:.5f}%)")

# Visualizing class imbalance
plt.figure(figsize=(6, 4))
sns.barplot(x=fraud_counts.index, y=fraud_counts.values, palette=['teal', 'red'])
plt.xticks([0, 1], ['Non-Fraud (0)', 'Fraud (1)'])
plt.ylabel("Count")
plt.xlabel("Transaction Type")
plt.title("Fraud vs. Non-Fraud Distribution")

# Save the plot to the defined path
save_path = os.path.join(visuals_dir, "fraud_distribution.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")

# Show the plot
plt.show()

print(f"Fraud distribution plot saved successfully at: {save_path}")

# 1.4 Feature Distributions & Correlations
# Analyze how Time and Amount are distributed.
# Visualizing feature distributions to detect patterns.
# Computing correlations to understand relationships between variables.
# 1.4.1 Understanding Time & Amount Distributions
# Plotting the distribution of 'Time' and 'Amount' features
plt.figure(figsize=(12, 5))

# Time Distribution
plt.subplot(1, 2, 1)
sns.histplot(df['Time'], bins=50, kde=True, color="blue")
plt.title("Transaction Time Distribution")
plt.xlabel("Seconds Since First Transaction")
plt.ylabel("Frequency")

# Amount Distribution (Log Scale for Better Visualization)
plt.subplot(1, 2, 2)
sns.histplot(df['Amount'], bins=50, kde=True, color="green")
plt.title("Transaction Amount Distribution (Log Scale)")
plt.xlabel("Transaction Amount ($)")
plt.ylabel("Frequency")
plt.yscale("log")  # Apply log scale to handle skewness

plt.tight_layout()

# Save the plot
plt.savefig(os.path.join(visuals_dir, "time_amount_distribution.png"), bbox_inches='tight')

# Show the plot
plt.show()

# 1.4.2 Distribution Plot of Time & Amount for Fraud vs. Non-Fraud
# Why? Helps us understand how transaction time and amount differ for fraudulent vs. non-fraudulent transactions.
# Method:

# Overlay distribution plots of Time and Amount for fraud vs. non-fraud cases.
# Helps detect any patterns in transaction timing or amount anomalies.
# Splitting fraud and non-fraud data for visualization
data_fraud = df[df["Class"] == 1]
data_non_fraud = df[df["Class"] == 0]

# Distribution plot for 'Time'
plt.figure(figsize=(8,5))
sns.kdeplot(data_fraud['Time'], label='Fraudulent', shade=True, color="red")
sns.kdeplot(data_non_fraud['Time'], label='Non-Fraudulent', shade=True, color="blue")
plt.xlabel('Seconds Since First Transaction')
plt.ylabel('Density')
plt.title("Transaction Time Distribution (Fraud vs. Non-Fraud)")
plt.legend()
plt.grid()

# Save the plot
plt.savefig(os.path.join(visuals_dir, "Transaction Time Distribution (Fraud vs. Non-Fraud).png"), bbox_inches='tight')

plt.show()

# Distribution plot for 'Amount' (Log Scale)
plt.figure(figsize=(8,5))
sns.kdeplot(np.log1p(data_fraud['Amount']), label='Fraudulent', shade=True, color="red")
sns.kdeplot(np.log1p(data_non_fraud['Amount']), label='Non-Fraudulent', shade=True, color="blue")
plt.xlabel('Log Transaction Amount ($)')
plt.ylabel('Density')
plt.title("Transaction Amount Distribution (Fraud vs. Non-Fraud) [Log Scale]")
plt.legend()
plt.grid()

# Save the plot
plt.savefig(os.path.join(visuals_dir, "Transaction Amount Distribution (Fraud vs. Non-Fraud) [Log Scale].png"), bbox_inches='tight')

plt.show()

# 1.4.3 Correlation Matrix & Heatmap
# Why? Helps us understand relationships between features and detect highly correlated variables.
# Method:

# Compute the correlation matrix to measure feature relationships.
# Use a heatmap to visualize correlations, making it easier to identify patterns.
# # Computing correlation matrix
correlation_matrix = df.corr()

# Heatmap Plot
plt.figure(figsize=(8, 4))
sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False, linewidths=0.5)
plt.title("Feature Correlation Heatmap")

# Save the plot
plt.savefig(os.path.join(visuals_dir, "Feature Correlation Heatmap.png"), bbox_inches='tight')

plt.show()


### 1.5 Key Insights from Exploratory Data Analysis (EDA)
#### Fraudulent transactions tend to have lower amounts compared to normal transactions.
#### Time variable does not show a strong pattern related to fraud cases.
# Highly correlated features might provide strong fraud detection signals.
# Dataset is highly imbalanced (fraud cases are only 0.172% of total transactions).
# Proper handling of class imbalance is critical for training an effective fraud detection model.