# Step 2. Data Preprocessing
# Before training our machine learning model, we need to preprocess the data. The preprocessing steps include handling missing values, feature scaling, encoding categorical variables, and handling class imbalance.

# 2.1 Checking and Handling for Missing Values
# Why? Identify and handle missing values to prevent errors and ensure data integrity.
# Method:

# Check for missing values in each column.
# If missing values exist, determine an appropriate strategy (e.g., imputation or removal).

# Checking for missing values
missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]

# Handling missing values if found
if not missing_values.empty:
    print("Missing values found in:\n", missing_values)
    
    # Example: Filling missing values with median (modify based on analysis)
    df.fillna(df.median(), inplace=True)
    print("\nMissing values handled using median imputation.")
else:
    print("No missing values in the dataset, No Handling for Missing values Needed.")


# 2.3 Feature Scaling (Standardization)
# Why? Since the dataset contains Amount and Time, which have different scales, we need to normalize them.
# Method:

# We will use StandardScaler to scale the numerical features.
# Standardization transforms features to have a mean of 0 and a standard deviation of 1.
# Effect:

# Ensures that all features contribute equally to the model without bias due to different units.
# Improves the performance of machine learning algorithms, especially those sensitive to feature scales.

from sklearn.preprocessing import StandardScaler

# Initialize StandardScaler
scaler = StandardScaler()

# Apply standardization to 'Time' and 'Amount'
df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])

print("Feature scaling applied using StandardScaler.")

# 2.4 Outlier Detection
# Why? Outliers can distort model performance by introducing extreme values.
# Goal: Identify and analyze outliers in Time and Amount features.
# Methods Used:

# Boxplots – Visualize extreme values in Time and Amount.
# Interquartile Range (IQR) Method – Detect and count the number of outliers.
# Next Step: Decide whether to remove or handle outliers based on their distribution.

# Boxplot for 'Amount' and 'Time' to visualize outliers

plt.figure(figsize=(12, 5))

# Amount Boxplot
plt.subplot(1, 2, 1)
sns.boxplot(x=df["Amount"], color="lightblue")
plt.title("Boxplot of Transaction Amount")

# Save the plot
plt.savefig(os.path.join(visuals_dir, "Boxplot of Transaction Amount.png"), bbox_inches='tight')

# Time Boxplot
plt.subplot(1, 2, 2)
sns.boxplot(x=df["Time"], color="lightgreen")
plt.title("Boxplot of Transaction Time")

# Save the plot
plt.savefig(os.path.join(visuals_dir, "Boxplot of Transaction Time.png"), bbox_inches='tight')

plt.show()

# 2.4.1 Boxplot Analysis for Outliers
# Why? Visualizing key features with boxplots helps identify extreme values.
# Method:

# Plot boxplots for Amount, Time, and selected V1 to V5 features.
# Detect potential outliers by analyzing whiskers and extreme points.
# Next Step: Use the IQR method to quantify and analyze outliers.

# Check for outliers using boxplots
plt.figure(figsize=(8,4))
df[["Amount", "Time", "V1", "V2", "V3", "V4", "V5"]].boxplot()
plt.xticks(rotation=15)
plt.title("Boxplot of Key Features for Outliers")

# Save the plot
plt.savefig(os.path.join(visuals_dir, "Boxplot of Key Features for Outliers.png"), bbox_inches='tight')

plt.show()

# 2.5 Detecting Outliers using IQR Method
# Why? The Interquartile Range (IQR) method helps detect and analyze outliers in numerical features.
# Method:

# Compute the first quartile (Q1) and third quartile (Q3).
# Calculate the Interquartile Range (IQR = Q3 - Q1).
# Define outlier thresholds:
# Lower Bound: Q1 - 1.5 * IQR
# Upper Bound: Q3 + 1.5 * IQR
# Count and analyze outliers in Time and Amount.

# Function to detect outliers using IQR
def detect_outliers(data, column):
    Q1 = data[column].quantile(0.25)  # First Quartile (25%)
    Q3 = data[column].quantile(0.75)  # Third Quartile (75%)
    IQR = Q3 - Q1  # Interquartile Range
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

# Detect outliers in 'Amount' and 'Time'
outliers_amount = detect_outliers(df, "Amount")
outliers_time = detect_outliers(df, "Time")

# Print number of outliers detected
print(f"Outliers detected in 'Amount': {len(outliers_amount)}")
print(f"Outliers detected in 'Time': {len(outliers_time)}")

# Decision: No Outlier Treatment Required

# We do not need to perform outlier removal or transformation.
# Why? The dataset has already been PCA transformed, meaning outliers have been accounted for during feature extraction.
# Removing or modifying outliers might negatively impact the model since PCA has already adjusted for extreme values.