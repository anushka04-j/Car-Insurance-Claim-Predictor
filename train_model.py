"""
Train logistic regression model for car insurance claim prediction.
Identifies key features and saves the trained model.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib
import os

# Load data
print("Loading dataset...")
df = pd.read_csv('insurance_claims.csv')

# Prepare features
print("Preparing features...")
df_processed = df.copy()

# Encode categorical variables
label_encoders = {}
categorical_cols = ['gender', 'vehicle_type', 'coverage_type']

for col in categorical_cols:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col])
    label_encoders[col] = le

# Select features (excluding target and premium which is derived)
feature_cols = ['age', 'gender', 'driving_experience', 'vehicle_age', 'vehicle_type',
                'annual_mileage', 'credit_score', 'previous_claims', 'traffic_violations',
                'coverage_type', 'deductible']

X = df_processed[feature_cols]
y = df_processed['claim_filed']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
print("Training logistic regression model...")
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
print("\n" + "="*50)
print("FEATURE IMPORTANCE (Coefficients)")
print("="*50)
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': model.coef_[0],
    'Abs_Coefficient': np.abs(model.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

print(feature_importance.to_string(index=False))

# Save model and preprocessors
print("\nSaving model and preprocessors...")
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/insurance_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(label_encoders, 'models/label_encoders.pkl')
joblib.dump(feature_cols, 'models/feature_cols.pkl')

print("\nModel saved successfully!")
print("Files saved in 'models' directory:")

