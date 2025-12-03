"""
Generate synthetic car insurance claims dataset for 'On the Road' insurance company.
This creates a realistic dataset with features that affect claim outcomes.
"""

import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of records
n_samples = 5000

# Generate synthetic data
data = {
    'age': np.random.randint(18, 80, n_samples),
    'gender': np.random.choice(['Male', 'Female'], n_samples),
    'driving_experience': np.random.randint(0, 50, n_samples),
    'vehicle_age': np.random.randint(0, 20, n_samples),
    'vehicle_type': np.random.choice(['Sedan', 'SUV', 'Sports Car', 'Truck', 'Hatchback'], n_samples, p=[0.3, 0.25, 0.15, 0.15, 0.15]),
    'annual_mileage': np.random.randint(5000, 30000, n_samples),
    'credit_score': np.random.randint(300, 850, n_samples),
    'previous_claims': np.random.poisson(0.3, n_samples),
    'traffic_violations': np.random.poisson(0.2, n_samples),
    'coverage_type': np.random.choice(['Basic', 'Standard', 'Premium'], n_samples, p=[0.4, 0.4, 0.2]),
    'deductible': np.random.choice([500, 1000, 2000, 5000], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
}

df = pd.DataFrame(data)

# Create target variable (claim outcome: 1 = claim filed, 0 = no claim)
# Higher probability of claim for:
# - Younger drivers (higher risk)
# - Sports cars (higher risk)
# - Higher mileage (more exposure)
# - Lower credit score (correlation with risk)
# - Previous claims (history of incidents)
# - Traffic violations (risky behavior)
# - Lower deductible (more likely to file small claims)

claim_probability = (
    0.1 +  # base probability
    (df['age'] < 30) * 0.15 +  # young drivers
    (df['age'] > 65) * 0.10 +  # older drivers
    (df['vehicle_type'] == 'Sports Car') * 0.20 +  # sports cars
    (df['annual_mileage'] > 20000) * 0.15 +  # high mileage
    (df['credit_score'] < 600) * 0.12 +  # low credit
    (df['previous_claims'] > 0) * 0.25 +  # previous claims
    (df['traffic_violations'] > 0) * 0.18 +  # violations
    (df['deductible'] == 500) * 0.10 -  # low deductible
    (df['driving_experience'] > 10) * 0.08  # experienced drivers
)

# Normalize probability to [0, 1]
claim_probability = np.clip(claim_probability, 0, 1)

# Generate binary outcome
df['claim_filed'] = np.random.binomial(1, claim_probability, n_samples)

# Add some noise and additional features
df['premium'] = (
    1000 + 
    df['age'].apply(lambda x: -20 if x < 25 else 10 if x > 65 else 0) +
    (df['vehicle_type'] == 'Sports Car') * 500 +
    (df['vehicle_type'] == 'SUV') * 200 +
    df['annual_mileage'] * 0.05 +
    (df['credit_score'] < 600) * 300 +
    df['previous_claims'] * 200 +
    df['traffic_violations'] * 150 +
    np.random.normal(0, 200, n_samples)
)

df['premium'] = df['premium'].clip(lower=500)

# Save to CSV
df.to_csv('insurance_claims.csv', index=False)
print(f"Dataset generated successfully with {n_samples} records!")
print(f"Claim rate: {df['claim_filed'].mean():.2%}")

