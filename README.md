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
✔ **Exploratory Data Analysis (EDA):** Understanding transaction patterns  
✔ **Data Preprocessing:** Handling missing values, scaling features, and balancing classes  
✔ **Feature Engineering:** Selecting the most relevant fraud-detection features  
✔ **Model Training & Evaluation:** Comparing multiple machine learning models  
✔ **Hyperparameter Tuning:** Optimizing the best-performing model  
✔ **Real-time & Batch Fraud Prediction:** Deploying the trained model for real-world usage  
✔ **Deployment Considerations:** Saving the final model for future use  

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
  **Feature Scaling:** Standardized `Amount` & `Time` using `StandardScaler`  
  **Class Balancing:** Applied **SMOTE (Synthetic Minority Oversampling Technique)**  
  **Feature Selection:** Selected the **13 most important fraud-related features**  

---

## Step 2. Installation & Setup

To run this fraud detection project on your local machine, follow these steps:

### **2.1 Clone the Repository**
```bash
git clone https://github.com/RajatKumarSh/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### **2.2 Create a Virtual Environment**
```bash
python -m venv fraud_env  # Create virtual environment
source fraud_env/bin/activate  # Activate (Linux/Mac)
fraud_env\Scripts\activate  # Activate (Windows)
```

### **2.3 Install Required Dependencies**
```bash
pip install -r requirements.txt
```

### **2.4 Run the Jupyter Notebook**
```bash
jupyter notebook
```

Navigate to `notebooks/` and open `fraud_detection.ipynb`.  

---

## Step 3. Project Folder Structure

**credit-card-fraud-detection**  
  **data/** *(Dataset Storage)*  
    creditcard.csv *(Raw Dataset)*  
    processed_data.csv *(Preprocessed Dataset)*  
  **notebooks/** *(Jupyter Notebooks - Code & Analysis)*  
  **models/** *(Saved Trained Model)*  
    final_fraud_detection_model.pkl  
  **visuals/** *(Generated Visualizations)*  
    feature_importance.png  
    precision_recall_curve.png  
    threshold_tuning_plot.png  
    +20 more
  **fraud_detection_report.pdf** *(Final Project Report)*  
  **README.md** *(Installation & Execution Guide)*  
  **requirements.txt** *(Dependencies List)*  

---

## Step 4. How the Model Works

### **4.1. Data Preprocessing & Balancing**  
  **Feature Scaling:** Standardized `Amount` & `Time`.  
  **Class Balancing:** Applied **SMOTE** to handle the extreme class imbalance.  
  **Feature Selection:** Used **Random Forest feature importance**.  

### **4.2. Model Training & Evaluation**  
  Trained multiple models: **Logistic Regression, Decision Tree, Random Forest, XGBoost**.  
  **Random Forest** was selected as the best-performing model.  
  Applied **Hyperparameter Tuning** to improve performance.  

### **4.3. Optimized Threshold for Fraud Detection**  
  Adjusted threshold to **0.19**, improving fraud detection recall.  

### **4.4. Real-Time & Batch Fraud Prediction**  
  Supports **real-time transactions** & **batch processing**.  
  The **final model is saved & deployed**.  

---

## Step 5. API Deployment Guide

### **5.1 Running the API Locally**
```bash
python deployment/Deployed_app.py
```

### **5.2 API Endpoints**
- **Health Check (GET Request)**
```bash
curl -X GET "https://finddefault-credit-card-fraud-detection.onrender.com/"
```

- **Fraud Prediction (POST Request)**
```bash
curl -X POST "https://finddefault-credit-card-fraud-detection.onrender.com/predict" -H "Content-Type: application/json" -d "{\"features\": [-4.3, 2.1, -1.8, 3.0, -2.5, 1.7, -0.9, 2.3, -1.5, 0.6, -0.7, 1.2, 120.5]}"

```

**Example Response:**
```json
{
  "fraud_probability": 0.0313,
  "prediction": "Safe Transaction"
}
```
---

## Troubleshooting
  If API **does not start**, ensure dependencies are installed with `pip install -r requirements.txt`.  
  If API **returns an error**, verify `Deployed_app.py` is running.  
  If **fraud probability seems incorrect**, adjust the threshold in `Deployed_app.py`.  

---