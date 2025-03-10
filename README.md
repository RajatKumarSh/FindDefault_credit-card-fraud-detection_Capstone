# Credit Card Fraud Detection - Machine Learning Capstone  

**Author:** Rajat Kumar  
**Date:** March 2025  
**Project Type:** Data Science / Machine Learning  
**Status:** Completed  

---

## Project Overview  

Credit card fraud is a major concern for financial institutions, leading to **financial losses and security risks**. Traditional fraud detection systems struggle with evolving fraud patterns.  

This project leverages **Machine Learning** to build an **AI-powered fraud detection system** that accurately identifies fraudulent transactions. The model is optimized to balance **precision and recall**, ensuring **minimal false positives** while **maximizing fraud detection**.

### **Key Features**  
**Exploratory Data Analysis (EDA):** Understanding transaction patterns  
**Data Preprocessing:** Handling missing values, scaling features, and balancing classes  
**Feature Engineering:** Selecting the most relevant fraud-detection features  
**Model Training & Evaluation:** Comparing multiple machine learning models  
**Hyperparameter Tuning:** Optimizing the best-performing model  
**Real-time & Batch Fraud Prediction:** Deploying the trained model for real-world usage  
**Deployment Considerations:** Saving the final model for future use  

---

## Dataset Information  

The dataset used for this project is the **Credit Card Fraud Detection Dataset**, which contains **real-world anonymized credit card transactions**.

### **Step 1. Dataset Overview:**  
- **Total Transactions:** 284,807  
- **Fraud Cases:** 492 (~0.17% of total data)  
- **Non-Fraud Cases:** 284,315 (~99.83% of total data)  
- **Features:** 30 numerical variables (`V1` to `V28`), `Time`, and `Amount`  
- **Target Variable:**  
  - `0` → Legitimate Transaction  
  - `1` → Fraudulent Transaction  

### **Data Preprocessing:**  
**Feature Scaling:** Standardized `Amount` & `Time`  
**Class Balancing:** Applied **SMOTE (Synthetic Minority Oversampling Technique)**  
**Feature Selection:** Selected the **13 most important fraud-related features**  

---

## **Step 2. Installation & Setup**  

To run this fraud detection project on your local machine, follow these steps:

### 1. Clone the Repository  
Open your terminal or command prompt and run:  

```bash
git clone https://github.com/RajatKumarSh/credit-card-fraud-detection.git
cd credit-card-fraud-detection

----

#### **2.1 Create a Virtual Environment** 
python -m venv fraud_env  # Create virtual environment
source fraud_env/bin/activate  # Activate (Linux/Mac)
fraud_env\Scripts\activate  # Activate (Windows)

---

#### **2.2 Install Required Dependencies**
pip install -r requirements.txt

---

#### **2.3 Run the Jupyter Notebook**
jupyter notebook

---

#### **2.3 Execute the Notebook**
Navigate to notebooks/ and open fraud_detection.ipynb.

---

## **Step 3. Project Folder Structure**  

The following structure organizes the files and directories within the project:

📦 credit-card-fraud-detection
┣ 📂 data/ # Dataset Storage
┃ ┣ 📄 creditcard.csv # Raw Dataset
┃ ┣ 📄 processed_data.csv # Preprocessed Dataset (Cleaned & Balanced)
┣ 📂 notebooks/ # Jupyter Notebooks (Code & Analysis)
┣ 📂 models/ # Saved Trained Model
┃ ┣ 📄 final_fraud_detection_model.pkl
┣ 📂 visuals/ # Saved Visualizations
┃ ┣ 📄 feature_importance.png
┃ ┣ 📄 precision_recall_curve.png
┃ ┣ 📄 threshold_tuning_plot.png
┣ 📄 fraud_detection_report.pdf # Final Project Report
┣ 📄 README.md # Installation & Execution Guide
┣ 📄 requirements.txt # Dependencies List


#### **Folder Descriptions:**  
- **data/** → Stores raw and preprocessed datasets.  
- **notebooks/** → Contains Jupyter notebooks for data analysis and modeling.  
- **models/** → Stores the final trained machine learning model.  
- **visuals/** → Contains all generated visualizations used in the project.  
- **fraud_detection_report.pdf** → Final report with methodology, results, and insights.  
- **requirements.txt** → Contains a list of required dependencies for easy setup.  

---

## **Step 4. How the Model Works**  

This fraud detection model uses **Machine Learning** to identify fraudulent transactions in real-time.  

#### **4.1. Data Preprocessing & Balancing**  
✔ **Feature Scaling:** Standardized `Amount` & `Time` using `StandardScaler`.  
✔ **Class Balancing:** Applied **SMOTE (Synthetic Minority Oversampling Technique)** to handle the extreme class imbalance.  
✔ **Feature Selection:** Used **Random Forest feature importance** to select the most relevant fraud-detection features.  

#### **4.2. Model Training & Evaluation**  
✔ Trained multiple models: **Logistic Regression, Decision Tree, Random Forest, XGBoost**.  
✔ **Random Forest** was selected as the best-performing model based on **F1-score & AUC-ROC**.  
✔ Applied **Hyperparameter Tuning** to improve model performance.  

#### **4.3. Optimized Threshold for Fraud Detection**  
✔ The default classification threshold (`0.5`) led to high false negatives.  
✔ We optimized the threshold to **0.19**, significantly improving fraud detection recall.  

#### **4.4. Real-Time & Batch Fraud Prediction**  
✔ The trained model is capable of detecting fraud in:  
   - **Real-time transactions** (instant fraud classification).  
   - **Batch transactions** (processing multiple transactions in one go).  
✔ The **final model is saved & deployed for real-world usage**.  

---

## **Step 5: Execution Instructions.**

## Execution Instructions  

To run this fraud detection project on your local machine, follow these steps:  

### **5.1. Clone the Repository**  
Open your terminal or command prompt and run:  

```bash
git clone https://github.com/RajatKumarSh/credit-card-fraud-detection.git
cd credit-card-fraud-detection

#### **5.2. Create a Virtual Environment**
python -m venv fraud_env  # Create virtual environment
source fraud_env/bin/activate  # Activate (Linux/Mac)
fraud_env\Scripts\activate  # Activate (Windows)

#### **5.3. Install Required Dependencies**
pip install -r requirements.txt

#### **5.4. Run the Jupyter Notebook**
jupyter notebook

#### **5.5. Execute the Notebook**
Open notebooks/fraud_detection.ipynb
Run all cells step by step to train the model and evaluate fraud detection.


## **Step 7. Model Deployment Guide**  

After training, the final fraud detection model is saved as **final_fraud_detection_model.pkl** in the `models/` directory.  

### **7.1. Load the Saved Model for Real-Time Predictions**  
To use the trained model for fraud detection on new transactions, follow these steps:  

```python
import joblib
import pandas as pd

# Load the trained model
model_path = "models/final_fraud_detection_model.pkl"
loaded_model = joblib.load(model_path)

# Example: Running a new transaction for fraud detection
new_transaction = pd.DataFrame([[1000, -1.23, 2.34, 0.56, -0.78, 1.45, -2.36, 0.78, -0.99, 1.34, -1.12, 0.45, -1.45]],
                               columns=['Amount', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12'])

# Predict fraud probability
fraud_probability = loaded_model.predict_proba(new_transaction)[:, 1][0]

# Apply the optimal threshold (0.19)
optimal_threshold = 0.19
fraud_label = " Fraud Alert!" if fraud_probability >= optimal_threshold else "Safe Transaction"

# Print results
print(f"Fraud Probability: {fraud_probability:.4f}")
print(f"Prediction: {fraud_label}")

#### **7.2. Batch Prediction for Multiple Transactions**

# Load test dataset (Example batch of transactions)
batch_transactions = pd.read_csv("data/processed_data.csv").sample(n=10, random_state=42)

# Predict fraud probabilities for batch transactions
fraud_probabilities = loaded_model.predict_proba(batch_transactions)[:, 1]

# Classify transactions based on the optimal threshold (0.19)
batch_predictions = (fraud_probabilities >= optimal_threshold).astype(int)

# Create a DataFrame to display results
batch_results = batch_transactions.copy()
batch_results["Fraud Probability"] = fraud_probabilities
batch_results["Prediction"] = batch_predictions
batch_results["Prediction Label"] = batch_results["Prediction"].apply(lambda x: "Fraud Alert!" if x == 1 else "Safe Transaction")

# Display the batch results
print("\nBatch Fraud Prediction Results:")
print(batch_results[["Fraud Probability", "Prediction Label"]])

---

### Deployment Considerations
This model can be deployed using:
Cloud-based API: Flask/FastAPI for real-time fraud detection in banking systems.
On-Premise Solution: Integrated into financial security platforms.
Batch Processing: Daily fraud risk analysis for large transaction volumes.


---
### API Testing Instructions - Fraud Detection Model

This section provides instructions to test the **deployed fraud detection API**.

---

### API Base URL
- **Local Testing:** The API runs on `http://127.0.0.1:5000`
- **Endpoint for Predictions:** `/predict`
- **HTTP Method:** `POST`
- **Content-Type:** `application/json`

---

### Example API Request for Fraud Detection
To test a **fraud transaction**, use the following example:

-- json
{
  "features": [-1.1303, -4.5956, 5.0837, -7.5810, 7.5460, -6.9491, -1.7291, -8.1902, 2.7147, -7.0832, -11.1413, 7.3819, -14.4687]
}

#### Run the following in your Command Prompt
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"features\": [-1.1303, -4.5956, 5.0837, -7.5810, 7.5460, -6.9491, -1.7291, -8.1902, 2.7147, -7.0832, -11.1413, 7.3819, -14.4687]}"

Expected Output:
{
  "fraud_probability": 0.393,
  "prediction": "Fraud Alert!"
}

#### Example API Request for a Safe Transaction:
To test a **non-fraudulent transaction**, use the following:
-- json
{
  "features": [-4.3, 2.1, -1.8, 3.0, -2.5, 1.7, -0.9, 2.3, -1.5, 0.6, -0.7, 1.2, 120.5]
}

Run the following in your Command Prompt:
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"features\": [-4.3, 2.1, -1.8, 3.0, -2.5, 1.7, -0.9, 2.3, -1.5, 0.6, -0.7, 1.2, 120.5]}"

Expected Output:

{
  "fraud_probability": 0.0313,
  "prediction": "Safe Transaction"
}

### Troubleshooting Common Issues
-- If you see a Not Found error → Make sure Deployed.py is running.
-- If you see a Connection Refused error → Restart Flask using:

python Deployed.py

-- If fraud probability seems incorrect → Adjust the fraud threshold in Deployed.py.




